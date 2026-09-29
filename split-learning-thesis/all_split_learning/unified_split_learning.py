import os
import torch
import numpy as np
from tqdm import tqdm
from config import Config


class DefendedSplitLearningTrainer:

    DECORR_WEIGHT = 0.05   # λ for EAD-Norm decorrelation auxiliary loss
    DCOR_WEIGHT = 0.02     # λ for Distance Correlation penalty (FSHA defense)
    GPI_RESAMPLE_INTERVAL = 50   # Batches before resampling GPI mask

    def __init__(self, client_model, server_model, train_loader, val_loader,
                 client_optimizer, server_optimizer, criterion, device,
                 checkpoint_dir="./checkpoints", results_dir="./results"):
        self.client_model = client_model
        self.server_model = server_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_optimizer = client_optimizer
        self.server_optimizer = server_optimizer
        self.criterion = criterion
        self.device = device

        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        self.train_losses, self.train_accuracies = [], []
        self.val_losses, self.val_accuracies = [], []
        self.best_val_acc = 0.0

    def _train_one_epoch(self, epoch):
        self.client_model.train()
        self.server_model.train()

        running_loss, correct, total = 0.0, 0, 0
        progress = tqdm(self.train_loader, desc=f"  Defended Epoch [{epoch+1}/{Config.EPOCHS}]", leave=False)

        for inputs, labels in progress:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if self.client_model.batches_since_resample >= self.GPI_RESAMPLE_INTERVAL:
                self.client_model.resample_gpi_mask()

            self.client_optimizer.zero_grad()
            z_sent = self.client_model(inputs)
            decorr = self.client_model.decorr_loss()
            dcor   = self.client_model.dcor_loss(inputs)

            z_sent_leaf = z_sent.detach().requires_grad_(True)

            self.server_optimizer.zero_grad()
            outputs = self.server_model(z_sent_leaf)
            task_loss = self.criterion(outputs, labels)

            task_loss.backward()
            self.server_optimizer.step()

            z_sent.backward(z_sent_leaf.grad, retain_graph=True)

            auxiliary_loss = (self.DECORR_WEIGHT * decorr) + (self.DCOR_WEIGHT * dcor)
            auxiliary_loss.backward()

            self.client_optimizer.step()

            running_loss += task_loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            progress.set_postfix({
                'Loss': f'{task_loss.item():.4f}',
                'Decorr': f'{decorr.item():.4f}',
                'dCor': f'{dcor.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        return epoch_loss, epoch_acc

    def evaluate(self, data_loader=None):
        if data_loader is None:
            data_loader = self.val_loader

        self.client_model.eval()
        self.server_model.eval()

        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                z_sent = self.client_model(inputs)
                # eval mode: _GradientDefenseFn is skipped (self.training=False
                # inside DefendedServerModel), dropout also disabled — clean
                # forward pass for accurate accuracy measurement.
                outputs = self.server_model(z_sent)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        eval_loss = running_loss / len(data_loader)
        eval_acc = 100. * correct / total
        return eval_loss, eval_acc

    def train(self, num_epochs=Config.EPOCHS):
        print(f"\n--- Starting Defended Split Learning Training ({num_epochs} Epochs) ---")
        for epoch in range(num_epochs):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.val_loader)

            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, filename="defended_split_model_best.pth")

        self.save_checkpoint(num_epochs - 1, filename="defended_split_model_final.pth")
        self.save_results("defended_training_results.pt")
        print("--- Defended Split Learning Training Complete ---\n")
        return self.train_losses, self.train_accuracies, self.val_accuracies

    def save_checkpoint(self, epoch, filename="defended_checkpoint.pth"):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'client_state_dict': self.client_model.state_dict(),
            'server_state_dict': self.server_model.state_dict(),
            'client_optimizer_state_dict': self.client_optimizer.state_dict(),
            'server_optimizer_state_dict': self.server_optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, path)
        print(f" Saved checkpoint to {path}")

    def load_checkpoint(self, filename="defended_checkpoint.pth"):
        path = os.path.join(self.checkpoint_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.client_model.load_state_dict(checkpoint['client_state_dict'])
        self.server_model.load_state_dict(checkpoint['server_state_dict'])
        self.client_optimizer.load_state_dict(checkpoint['client_optimizer_state_dict'])
        self.server_optimizer.load_state_dict(checkpoint['server_optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f" Loaded checkpoint from {path} (Epoch {checkpoint['epoch']+1})")

    def save_results(self, filename="defended_results.pt"):
        path = os.path.join(self.results_dir, filename)
        torch.save({
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc
        }, path)
        print(f" Saved evaluation metrics to {path}")