import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

from config import Config
from all_defences.splitguard_defense import A_ESTIMATORS


class SplitGuardTrainer:

    def __init__(self, client_model, server_model, train_loader, test_loader,
                 defense, adv_type='honest', stop_policy=None, a_estimator='output',
                 server_attack=None, public_loader=None, on_detection=None):
        assert adv_type in ('honest', 'random', 'fsha'), "adv_type must be 'honest', 'random' or 'fsha'"
        if adv_type == 'fsha':
            assert server_attack is not None and public_loader is not None, \
                "adv_type='fsha' needs server_attack (FSHAAttack) and public_loader"
        self.server_attack = server_attack
        self.public_loader = public_loader
        self.on_detection  = on_detection
        self._pub_iter     = None
        self._reported     = set()

        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {self.device}")

        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.defense      = defense
        self.adv_type     = adv_type
        self.stop_policy  = stop_policy
        self.stopped_at   = None
        assert a_estimator in A_ESTIMATORS, f"a_estimator must be one of {list(A_ESTIMATORS)}"
        self.a_estimator_name = a_estimator
        self.estimator = A_ESTIMATORS[a_estimator]()
        if a_estimator == 'local':
            print("  Estimating A: training a local copy of the full model for one epoch...")
            self.estimator.prepare(self.client_model, self.server_model, self.train_loader, self.device)
            print(f"  Local model accuracy at end of epoch: {self.estimator.curve[-1]*100:.2f}%")
        if stop_policy is not None:
            assert stop_policy in defense.detections, f"unknown policy {stop_policy}"

        self.client_optimizer = optim.Adam(self.client_model.parameters(), lr=0.001, amsgrad=True)
        self.server_optimizer = optim.Adam(self.server_model.parameters(), lr=0.001, amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()

        self.train_losses, self.train_accuracies, self.test_accuracies = [], [], []
        self.score_log = []

        os.makedirs(Config.SAVE_DIR, exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    def _train_one_epoch(self, epoch, epochs):
        self.client_model.train()
        self.server_model.train()

        running_loss, correct, total, n_regular = 0.0, 0, 0, 0
        progress = tqdm(self.train_loader, desc=f"  [{self.adv_type}] Epoch [{epoch+1}/{epochs}]", leave=False)

        for index, (images, labels) in enumerate(progress):
            images, labels = images.to(self.device), labels.to(self.device)
            self.client_optimizer.zero_grad()
            self.server_optimizer.zero_grad()

            send_fakes = self.defense.should_send_fake(index)

            smashed = self.client_model(images)
            smashed_server = smashed.detach().requires_grad_(True)

            LABELS_SENT = labels
            if send_fakes:
                LABELS_SENT = self.defense.make_fake_labels(labels)

            outputs = self.server_model(smashed_server)
            if self.adv_type == 'honest':
                loss = self.criterion(outputs, LABELS_SENT)
            elif self.adv_type == 'random':
                loss = self.criterion(outputs, torch.randint(0, 10, labels.size(), dtype=torch.long, device=self.device))
            elif self.adv_type == 'fsha':
                loss = self._fsha_server_loss(smashed_server)

            loss.backward()
            smashed.backward(smashed_server.grad)

            self.estimator.update(index, smashed, outputs, labels, send_fakes)
            client_grad = list(self.client_model.parameters())[0].grad.detach().clone().flatten()
            score = self.defense.record(index, send_fakes, client_grad,
                                        A=self.estimator.accuracy(index), batch_size=labels.size(0))
            if score is not None:
                self.score_log.append((index, score))
                if self.on_detection is not None:
                    for name, at in self.defense.detections.items():
                        if at is not None and name not in self._reported:
                            self._reported.add(name)
                            self.on_detection(name, index)
                if self.stop_policy and self.defense.attack_detected(self.stop_policy):
                    self.stopped_at = index
                    print(f"\n  [SplitGuard] '{self.stop_policy}' policy reports an attack at "
                          f"batch {index} (SG={score:.4f}). Stopping training.")
                    break

            if not send_fakes:
                self.client_optimizer.step()
            self.server_optimizer.step()

            if not send_fakes:
                running_loss += loss.item()
                n_regular    += 1
                correct += outputs.argmax(1).eq(labels).sum().item()
                total   += labels.size(0)

            progress.set_postfix({'Loss': f'{loss.item():.4f}',
                                  'SG': f'{self.defense.scores[-1]:.3f}' if self.defense.scores else '-'})

        epoch_loss = running_loss / max(n_regular, 1)
        epoch_acc  = 100. * correct / max(total, 1)
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        return epoch_loss, epoch_acc

    def _next_public(self):
        if self._pub_iter is None:
            self._pub_iter = iter(self.public_loader)
        try:
            x, _ = next(self._pub_iter)
        except StopIteration:
            self._pub_iter = iter(self.public_loader)
            x, _ = next(self._pub_iter)
        return x.to(self.device)

    def _fsha_server_loss(self, smashed_server):
        from all_attacks.fsha_attack import gradient_penalty
        atk = self.server_attack
        pub = self._next_public()
        atk.pilot_optimizer.zero_grad()
        recon_loss = atk.mse(atk.decoder(atk.pilot(pub)), pub)
        recon_loss.backward()
        atk.pilot_optimizer.step()

        fixed = smashed_server.detach()
        for _ in range(atk.critic_iters):
            batch = self._next_public()
            with torch.no_grad():
                real = atk.pilot(batch)
            n = min(real.shape[0], fixed.shape[0])
            atk.critic_optimizer.zero_grad()
            critic_loss = (atk.critic(fixed[:n]).mean() - atk.critic(real[:n]).mean()
                           + atk.gp_lambda * gradient_penalty(atk.critic, real[:n], fixed[:n], self.device))
            critic_loss.backward()
            atk.critic_optimizer.step()
        return -atk.critic(smashed_server).mean()

    def _evaluate(self):
        self.client_model.eval()
        self.server_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.server_model(self.client_model(inputs))
                correct += outputs.argmax(1).eq(labels).sum().item()
                total   += labels.size(0)
        acc = 100. * correct / total
        self.test_accuracies.append(acc)
        return acc

    def train(self, epochs=1):
        print("\n" + "=" * 60)
        print(f"   SPLITGUARD -- server: {self.adv_type.upper()}")
        print("=" * 60)
        print(f"  Dataset    : {Config.DATASET}")
        print(f"  Epochs     : {epochs}")
        print(f"  Defense    : {self.defense}")
        print(f"  A estimate : {self.a_estimator_name}")
        print("=" * 60 + "\n")

        for epoch in range(epochs):
            loss, tr_acc = self._train_one_epoch(epoch, epochs)
            te_acc = self._evaluate()
            det = {k: v for k, v in self.defense.detections.items() if v is not None}
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {loss:.4f} | "
                  f"Train: {tr_acc:.2f}% | Test: {te_acc:.2f}% | "
                  f"Fake batches: {len(self.defense.fakes)} | "
                  f"Mean SG: {self.defense.mean_score():.4f}")
            print(f"  Policy detections (batch index): {det if det else 'none'}")
            if self.defense.n_history or self.defense.bf_history:
                print(f"  Algorithm 3 adjustments | B_F: {self.defense.bf_history} | N: {self.defense.n_history}")
            if self.stopped_at is not None:
                break
        return self.defense.scores

    def save_results(self):
        tag = f"splitguard_{self.a_estimator_name}_{self.adv_type}_{Config.DATASET}"
        pd.DataFrame(self.score_log, columns=['batch_index', 'sg_score']).to_csv(
            f"{Config.RESULTS_DIR}/{tag}_scores.csv", index=False)
        pd.DataFrame({'epoch': range(1, len(self.train_losses) + 1),
                      'train_loss': self.train_losses,
                      'train_accuracy': self.train_accuracies,
                      'test_accuracy': self.test_accuracies}).to_csv(
            f"{Config.RESULTS_DIR}/{tag}_training.csv", index=False)
        torch.save({'client_state': self.client_model.state_dict(),
                    'server_state': self.server_model.state_dict(),
                    'adv_type': self.adv_type,
                    'sg_scores': self.defense.scores,
                    'detections': self.defense.detections,
                    'stop_policy': self.stop_policy,
                    'stopped_at': self.stopped_at,
                    'a_estimator': self.a_estimator_name,
                    'b_fake_history': self.defense.bf_history,
                    'n_history': self.defense.n_history},
                   f"{Config.SAVE_DIR}/splitguard_{self.a_estimator_name}_{self.adv_type}_{Config.DATASET}.pth")
        pd.DataFrame(self.defense.decision_log, columns=['batch_index', 'policy', 'decision', 'A', 'A_F']).to_csv(
            f"{Config.RESULTS_DIR}/{tag}_decisions.csv", index=False)
        print(f"  Results saved -> {Config.RESULTS_DIR}/{tag}_*.csv")
