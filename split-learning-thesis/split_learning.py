# Core Vanilla Split Learning training and evaluation logic
#
# How the backward pass works:
# 1. Server computes loss and calls loss.backward()
# 2. This gives us gradient w.r.t. smashed_data (smashed_data_detached.grad)
# 3. We send this gradient back to the client
# 4. Client calls smashed_data.backward(gradient) to update its layers
# This simulates the gradient communication in real SL deployments


import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
from config import Config


class SplitLearningTrainer:

    def __init__(self, client_model, server_model, train_loader, test_loader):

        # Device setup
        self.device = torch.device(
            Config.DEVICE if torch.cuda.is_available() else 'cpu'
        )
        print(f"Device: {self.device}")

        # Move models to device
        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        self.train_loader = train_loader
        self.test_loader  = test_loader

        # Separate optimizers — simulates client and server updating independently
        self.client_optimizer = optim.Adam(
            self.client_model.parameters(), lr=Config.LEARNING_RATE
        )
        self.server_optimizer = optim.Adam(
            self.server_model.parameters(), lr=Config.LEARNING_RATE
        )

        # Learning rate scheduler — reduces LR when accuracy plateaus
        self.client_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.client_optimizer, patience=5, factor=0.5, verbose=True
        )
        self.server_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.server_optimizer, patience=5, factor=0.5, verbose=True
        )

        self.criterion = nn.CrossEntropyLoss()

        # Result tracking
        self.train_losses      = []
        self.train_accuracies  = []
        self.test_accuracies   = []

        os.makedirs(Config.SAVE_DIR,   exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # --------------------------Single epoch training--------------------------
    def train_one_epoch(self, epoch):
        self.client_model.train()
        self.server_model.train()

        running_loss = 0.0
        correct = 0
        total   = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch+1}/{Config.EPOCHS}]",
            leave=False
        )

        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # --------------------------CLIENT FORWARD PASS--------------------------
            self.client_optimizer.zero_grad()
            smashed_data = self.client_model(inputs)
            # smashed_data shape: [batch, 32, 8, 8] for CIFAR-10

            # Detach from client graph before sending to server
            # requires_grad=True allows gradient to flow back to client
            smashed_data_server = smashed_data.detach().requires_grad_(True)

            # --------------------------SERVER FORWARD PASS--------------------------
            self.server_optimizer.zero_grad()
            outputs = self.server_model(smashed_data_server)
            loss    = self.criterion(outputs, labels)

            # --------------------------SERVER BACKWARD PASS--------------------------
            loss.backward()
            self.server_optimizer.step()

            # --------------------------CLIENT BACKWARD PASS--------------------------
            # smashed_data_server.grad is the gradient the server
            # computed w.r.t. smashed data — this is sent back to client
            smashed_data.backward(smashed_data_server.grad)
            self.client_optimizer.step()

            # --------------------------Track metrics--------------------------
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

    # --------------------------Evaluation--------------------------
    def evaluate(self):
        self.client_model.eval()
        self.server_model.eval()

        correct = 0
        total   = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                smashed_data = self.client_model(inputs)
                outputs      = self.server_model(smashed_data)

                _, predicted = outputs.max(1)
                total       += labels.size(0)
                correct     += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        self.test_accuracies.append(test_acc)
        return test_acc

    # -----------------------Full training loop-------------------------------
    def train(self):
        print("\n" + "="*60)
        print("   VANILLA SPLIT LEARNING — TRAINING START")
        print("="*60)
        print(f"  Dataset  : {Config.DATASET}")
        print(f"  Epochs   : {Config.EPOCHS}")
        print(f"  LR       : {Config.LEARNING_RATE}")
        print(f"  Cut Layer: {Config.CUT_LAYER}")
        print("="*60 + "\n")

        best_acc = 0.0

        for epoch in range(Config.EPOCHS):
            train_loss, train_acc = self.train_one_epoch(epoch)
            test_acc              = self.evaluate()

            # Step schedulers
            self.client_scheduler.step(test_acc)
            self.server_scheduler.step(test_acc)

            print(
                f"Epoch {epoch+1:3d}/{Config.EPOCHS} | "
                f"Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Test Acc: {test_acc:.2f}%"
            )

            # Save best model checkpoint
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch'       : epoch,
                    'client_state': self.client_model.state_dict(),
                    'server_state': self.server_model.state_dict(),
                    'best_acc'    : best_acc,
                    'config'      : {
                        'dataset'   : Config.DATASET,
                        'cut_layer' : Config.CUT_LAYER,
                        'lr'        : Config.LEARNING_RATE
                    }
                }, f"{Config.SAVE_DIR}/best_vanilla_sl.pth")

        print(f"\nTraining complete. Best Test Accuracy: {best_acc:.2f}%")
        return self.train_losses, self.train_accuracies, self.test_accuracies

    # --------------------------Save CSV results---------------------------------
    def save_results(self):
        df = pd.DataFrame({
            'epoch'          : range(1, len(self.train_losses) + 1),
            'train_loss'     : self.train_losses,
            'train_accuracy' : self.train_accuracies,
            'test_accuracy'  : self.test_accuracies
        })
        path = f"{Config.RESULTS_DIR}/vanilla_sl_results.csv"
        df.to_csv(path, index=False)
        print(f"Results saved → {path}")