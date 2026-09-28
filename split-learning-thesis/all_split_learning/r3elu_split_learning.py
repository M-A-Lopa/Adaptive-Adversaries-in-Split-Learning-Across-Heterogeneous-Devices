import torch
import pandas as pd
from tqdm import tqdm

from config import Config
from all_split_learning.split_learning import SplitLearningTrainer
from all_defences.r3elu_defense import R3eLU


class R3eLUSplitLearningTrainer(SplitLearningTrainer):
    """
    Vanilla SL training loop with the R3eLU privacy tunnel of Mao et al.:
      guest : smashed = R3eLU-forward( client(x) )                         (Alg. 1, protects the guest)
      host  : grad    = R3eLU-backward( dL/d smashed )  before sending it  (Alg. 2, protects the host)
    The perturbation is also active at test time — the guest never sends raw activations.
    """

    def __init__(self, client_model, server_model, train_loader, test_loader, mechanism, tag="r3elu"):
        super().__init__(client_model, server_model, train_loader, test_loader)
        self.mech = mechanism
        self.r3elu = R3eLU(mechanism)
        self.tag = tag

    def _train_one_epoch(self, epoch):
        self.client_model.train()
        self.server_model.train()
        running_loss, correct, total = 0.0, 0, 0

        bar = tqdm(self.train_loader, desc=f"  R3eLU eps={self.mech.epsilon} [{epoch+1}/{Config.EPOCHS}]", leave=False)
        for inputs, labels in bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.client_optimizer.zero_grad()
            smashed = self.r3elu(self.client_model(inputs))                       # guest side

            smashed_server = smashed.detach().requires_grad_(True)                # host side
            self.server_optimizer.zero_grad()
            outputs = self.server_model(smashed_server)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.server_optimizer.step()

            grad = smashed_server.grad
            if self.mech.protect_backward:
                grad = self.mech.backward_perturb(grad)                           # host perturbs before sending
            smashed.backward(grad)                                                # guest back-propagates
            self.client_optimizer.step()
            self.mech.end_iteration()

            running_loss += loss.item()
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

        self.train_losses.append(running_loss / len(self.train_loader))
        self.train_accuracies.append(100. * correct / total)
        return self.train_losses[-1], self.train_accuracies[-1]

    def _evaluate(self):
        self.client_model.eval()
        self.server_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.server_model(self.r3elu(self.client_model(inputs)))
                correct += outputs.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        self.test_accuracies.append(100. * correct / total)
        return self.test_accuracies[-1]

    def _save_checkpoint(self, epoch, best_acc):
        torch.save({'epoch': epoch, 'client_state': self.client_model.state_dict(),
                    'server_state': self.server_model.state_dict(), 'best_acc': best_acc,
                    'epsilon': self.mech.epsilon, 'K': self.mech.K, 'C': self.mech.C,
                    'importance': self.mech.importance.U.cpu(), 'dataset': Config.DATASET},
                   f"{Config.SAVE_DIR}/best_{self.tag}_{Config.MODEL_NAME.lower()}_{Config.DATASET}.pth")

    def save_results(self):
        pd.DataFrame({'epoch': range(1, len(self.train_losses) + 1), 'train_loss': self.train_losses,
                      'train_accuracy': self.train_accuracies, 'test_accuracy': self.test_accuracies}) \
            .to_csv(f"{Config.RESULTS_DIR}/{self.tag}_{Config.MODEL_NAME.lower()}_training_{Config.DATASET}.csv", index=False)
