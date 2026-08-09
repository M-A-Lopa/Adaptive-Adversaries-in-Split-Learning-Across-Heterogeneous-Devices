# split_learning.py
# Vanilla Split Learning — Training and Evaluation Engine
#
# How the backward pass works:
# 1. Client computes smashed data from input
# 2. Server receives smashed data, computes loss, calls backward
# 3. Gradient of smashed data flows back to client
# 4. Client uses gradient to update its own layers
#
# Attack integration:
# get_smashed_data() exposes intermediate activations for attack evaluation
# evaluate_with_defense() measures accuracy cost of any defense function


import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from config import Config


class SplitLearningTrainer:
    """
    Trains client and server models using Vanilla Split Learning protocol.
    Simulates two-party communication via smashed data and gradient exchange.

    After training, the client model can be passed directly to
    WhiteBoxInversionAttack — the attack calls client_model(inputs)
    to obtain smashed data without needing any changes here.
    """

    def __init__(self, client_model, server_model,
                 train_loader, test_loader):

        self.device = torch.device(
            Config.DEVICE if torch.cuda.is_available() else 'cpu'
        )
        print(f"  Device: {self.device}")

        # Move models to device
        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        self.train_loader = train_loader
        self.test_loader  = test_loader

        # Separate optimizers — simulates client and server
        # updating their own weights independently
        self.client_optimizer = optim.Adam(
            self.client_model.parameters(),
            lr=Config.LEARNING_RATE
        )
        self.server_optimizer = optim.Adam(
            self.server_model.parameters(),
            lr=Config.LEARNING_RATE
        )

        # Reduce LR when accuracy plateaus — prevents overfitting
        self.client_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.client_optimizer, patience=5, factor=0.5
        )
        self.server_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.server_optimizer, patience=5, factor=0.5
        )

        self.criterion = nn.CrossEntropyLoss()

        # Metric history — used for plotting and CSV export
        self.train_losses     = []
        self.train_accuracies = []
        self.test_accuracies  = []

        os.makedirs(Config.SAVE_DIR,    exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # ── Single epoch training ─────────────────────────────────────────
    def _train_one_epoch(self, epoch):
        """
        Runs one full training epoch using the SL communication protocol.

        The detach + requires_grad trick simulates network communication:
        - smashed_data_server is what the server actually receives
        - its .grad after server backward = the gradient message sent back
        - client uses that gradient to update its own layers
        """
        self.client_model.train()
        self.server_model.train()

        running_loss = 0.0
        correct      = 0
        total        = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"  Epoch [{epoch+1}/{Config.EPOCHS}]",
            leave=False
        )

        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # ── CLIENT FORWARD PASS ───────────────────────────────────
            self.client_optimizer.zero_grad()
            smashed_data = self.client_model(inputs)
            # smashed_data shape: [batch, 32, 8, 8] for CIFAR-10
            #                     [batch, 32, 5, 5] for MNIST

            # Detach from client graph — simulates sending over network
            # requires_grad=True allows gradient to flow back from server
            smashed_data_server = smashed_data.detach().requires_grad_(True)

            # ── SERVER FORWARD PASS ───────────────────────────────────
            self.server_optimizer.zero_grad()
            outputs = self.server_model(smashed_data_server)
            loss    = self.criterion(outputs, labels)

            # ── SERVER BACKWARD PASS ──────────────────────────────────
            # Server computes gradient of loss w.r.t. smashed data
            # This gradient is the message sent back to client
            loss.backward()
            self.server_optimizer.step()

            # ── CLIENT BACKWARD PASS ──────────────────────────────────
            # smashed_data_server.grad is the gradient from server
            # Client uses this to backpropagate through its own layers
            smashed_data.backward(smashed_data_server.grad)
            self.client_optimizer.step()

            # ── Track metrics ─────────────────────────────────────────
            running_loss += loss.item()
            _, predicted  = outputs.max(1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc' : f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc  = 100. * correct / total

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)

        return epoch_loss, epoch_acc

    # ── Evaluation ────────────────────────────────────────────────────
    def _evaluate(self):
        """
        Standard accuracy evaluation on test set.
        No attack or defense involved — pure classification performance.
        """
        self.client_model.eval()
        self.server_model.eval()

        correct = 0
        total   = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs  = inputs.to(self.device)
                labels  = labels.to(self.device)
                smashed = self.client_model(inputs)
                outputs = self.server_model(smashed)
                _, predicted = outputs.max(1)
                total   += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        self.test_accuracies.append(test_acc)
        return test_acc

    # ── Full training loop ────────────────────────────────────────────
    def train(self):
        """
        Runs the full training loop for Config.EPOCHS epochs.
        Saves best checkpoint automatically.
        Returns metric histories for plotting.
        """
        print("\n" + "="*60)
        print("   VANILLA SPLIT LEARNING — TRAINING START")
        print("="*60)
        print(f"  Dataset    : {Config.DATASET}")
        print(f"  Epochs     : {Config.EPOCHS}")
        print(f"  LR         : {Config.LEARNING_RATE}")
        print(f"  Cut Layer  : {Config.CUT_LAYER}")
        print(f"  Batch Size : {Config.BATCH_SIZE}")
        print("="*60 + "\n")

        best_acc = 0.0

        for epoch in range(Config.EPOCHS):
            train_loss, train_acc = self._train_one_epoch(epoch)
            test_acc              = self._evaluate()

            # Step schedulers based on test accuracy
            self.client_scheduler.step(test_acc)
            self.server_scheduler.step(test_acc)

            print(
                f"  Epoch {epoch+1:3d}/{Config.EPOCHS} | "
                f"Loss: {train_loss:.4f} | "
                f"Train: {train_acc:.2f}% | "
                f"Test: {test_acc:.2f}%"
            )

            # Save checkpoint when test accuracy improves
            if test_acc > best_acc:
                best_acc = test_acc
                self._save_checkpoint(epoch, best_acc)

        print(f"\n  Training complete. Best Test Accuracy: {best_acc:.2f}%")
        return self.train_losses, self.train_accuracies, self.test_accuracies

    # ── Checkpoint save ───────────────────────────────────────────────
    def _save_checkpoint(self, epoch, best_acc):
        """
        Saves both client and server model states.
        Includes dataset name so attack knows which normalization to use.
        Filename includes dataset name to prevent CIFAR-10 and MNIST
        checkpoints overwriting each other.
        """
        save_path = f"{Config.SAVE_DIR}/best_vanilla_sl_{Config.DATASET}.pth"
        torch.save({
            'epoch'        : epoch,
            'client_state' : self.client_model.state_dict(),
            'server_state' : self.server_model.state_dict(),
            'best_acc'     : best_acc,
            'dataset'      : Config.DATASET,
            'cut_layer'    : Config.CUT_LAYER,
            'lr'           : Config.LEARNING_RATE
        }, save_path)

    # ── Smashed data extraction — used by attack ──────────────────────
    def get_smashed_data(self, inputs):
        """
        Exposes intermediate smashed data from the client model.
        This is what the server receives during normal SL operation
        and what the attacker intercepts during a model inversion attack.

        Used by WhiteBoxInversionAttack in attacks.py:
            smashed = trainer.get_smashed_data(inputs)
            reconstructed = attacker.reconstruct(smashed, inputs.shape)

        Can also be called directly on client_model:
            smashed = client_model(inputs)   ← equivalent, simpler

        inputs: [batch, C, H, W] — normalized image batch on correct device
        returns: [batch, C_smash, H_smash, W_smash] — smashed data tensor
        """
        self.client_model.eval()
        with torch.no_grad():
            return self.client_model(inputs)

    # ── Accuracy with defense — used by defense evaluation ────────────
    def evaluate_with_defense(self, defense_fn):
        """
        Measures classification accuracy when a defense is applied
        to smashed data before it reaches the server.

        defense_fn: callable that takes smashed data tensor and returns
                    a perturbed version of the same shape.
                    Example: pgsl_defense.protect or dpsl_defense.protect

        Used during defense evaluation to measure accuracy cost:
            baseline_acc = trainer.evaluate_with_defense(lambda x: x)
            pgsl_acc     = trainer.evaluate_with_defense(pgsl.protect)
            dpsl_acc     = trainer.evaluate_with_defense(dpsl.protect)

        Returns accuracy as float percentage.
        """
        self.client_model.eval()
        self.server_model.eval()

        correct = 0
        total   = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs  = inputs.to(self.device)
                labels  = labels.to(self.device)

                # Get smashed data
                smashed = self.client_model(inputs)

                # Apply defense — server only sees protected version
                smashed_protected = defense_fn(smashed)

                # Server classifies from protected smashed data
                outputs = self.server_model(smashed_protected)
                _, predicted = outputs.max(1)
                total   += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100.0 * correct / total

    # ── Save results CSV ──────────────────────────────────────────────
    def save_results(self):
        """
        Saves epoch-by-epoch training metrics to CSV.
        Filename includes dataset name so CIFAR-10 and MNIST results
        coexist without overwriting.
        Used by compare_datasets.py for side-by-side comparison plots.
        """
        df = pd.DataFrame({
            'epoch'         : range(1, len(self.train_losses) + 1),
            'train_loss'    : self.train_losses,
            'train_accuracy': self.train_accuracies,
            'test_accuracy' : self.test_accuracies
        })
        path = f"{Config.RESULTS_DIR}/vanilla_sl_results_{Config.DATASET}.csv"
        df.to_csv(path, index=False)
        print(f"  Results saved → {path}")