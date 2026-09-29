import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import gc
from config import Config
from all_defences.pgsl_defense import PGSLDefenseModules, AdaptiveWeightedDecisionFusion


def _get_norm_stats(dataset, device):
    if dataset == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.1307], device=device).view(1, 1, 1, 1)
        std  = torch.tensor([0.3081], device=device).view(1, 1, 1, 1)
    return mean, std


def denormalize(x, mean, std):
    return torch.clamp(x * std + mean, 0.0, 1.0)


class PGSLSplitLearningTrainer:

    def __init__(self, client_model, server_model, train_loader, test_loader):
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {self.device}")
        print(f"  CUDA available: {torch.cuda.is_available()}") 

        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        self.train_loader = train_loader
        self.test_loader  = test_loader

        self.mean, self.std = _get_norm_stats(Config.DATASET, self.device)

        self.client_optimizer = optim.Adam(self.client_model.parameters(), lr=Config.LEARNING_RATE)
        self.server_optimizer = optim.Adam(self.server_model.parameters(), lr=Config.LEARNING_RATE)

        self.client_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.client_optimizer, patience=5, factor=0.5)
        self.server_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.server_optimizer, patience=5, factor=0.5)

        self.criterion = nn.CrossEntropyLoss()

        self.train_losses     = []
        self.train_accuracies = []
        self.test_accuracies  = []

        os.makedirs(Config.SAVE_DIR,    exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    def _train_one_epoch(self, epoch):
        self.client_model.train()
        self.server_model.train()

        running_loss = 0.0
        correct      = 0
        total        = 0

        progress_bar = tqdm(self.train_loader, desc=f"  PGSL Epoch [{epoch+1}/{Config.EPOCHS}]", leave=False)
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            raw_inputs = denormalize(inputs, self.mean, self.std).detach().requires_grad_(True)

            baseline_tensor = PGSLDefenseModules.space_to_depth_downsample(raw_inputs, saliency_map=None)
            s_baseline = self.client_model(baseline_tensor)
            baseline_logits, _, _ = self.server_model(s_baseline, run_full_pipeline=False)

            x_perturbed, map_tensor = PGSLDefenseModules.generate_faithful_saliency_map(
                inputs=raw_inputs, baseline_logits=baseline_logits, num_classes=Config.NUM_CLASSES
            )

            self.client_optimizer.zero_grad()
            self.server_optimizer.zero_grad()

            client_tensor = PGSLDefenseModules.space_to_depth_downsample(x_perturbed, saliency_map=map_tensor)

            smashed_data = self.client_model(client_tensor)

            s_leaf = smashed_data.detach().requires_grad_(True)

            out_a, out_r, out_f = self.server_model(s_leaf, run_full_pipeline=True)

            loss_a = self.criterion(out_a, labels)
            loss_r = self.criterion(out_r, labels)
            loss_f = self.criterion(out_f, labels)

            loss_a.backward(retain_graph=True)
            loss_r.backward(retain_graph=True)

            if s_leaf.grad is not None:
                s_leaf.grad.zero_()

            loss_f.backward()

            smashed_data.backward(gradient=s_leaf.grad)

            self.server_optimizer.step()
            self.client_optimizer.step()

            with torch.no_grad():
                final_logits = AdaptiveWeightedDecisionFusion.fuse_outputs(out_a, out_r, out_f)
            _, predicted = final_logits.max(1)

            running_loss += loss_f.item()
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({'Loss': f'{loss_f.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
            
            del baseline_tensor, s_baseline, baseline_logits
            del x_perturbed, map_tensor, client_tensor
            del smashed_data, s_leaf, out_a, out_r, out_f, final_logits, predicted

            if (batch_idx + 1) % 100 == 0:
                gc.collect()
                
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc  = 100. * correct / total

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)

        return epoch_loss, epoch_acc

    def evaluate(self):
        self.client_model.eval()
        self.server_model.eval()

        correct = 0
        total   = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                raw_inputs = denormalize(inputs, self.mean, self.std)
                client_tensor = PGSLDefenseModules.space_to_depth_downsample(raw_inputs, saliency_map=None)

                smashed = self.client_model(client_tensor)
                out_a, out_r, out_f = self.server_model(smashed, run_full_pipeline=True)

                final_logits = AdaptiveWeightedDecisionFusion.fuse_outputs(out_a, out_r, out_f)

                _, predicted = final_logits.max(1)
                total   += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        self.test_accuracies.append(test_acc)
        return test_acc

    def train(self):
            print("\n" + "="*60)
            print("   PGSL SPLIT LEARNING — TRAINING START")
            print("="*60)
            print(f"  Model    : {Config.MODEL_NAME}")
            print(f"  Dataset  : {Config.DATASET}")
            print(f"  Cut layer: {Config.CUT_LAYER}")
            print(f"  Epochs   : {Config.EPOCHS}")
            print(f"  LR       : {Config.LEARNING_RATE}")
            print("="*60 + "\n")

            best_acc = 0.0

            for epoch in range(Config.EPOCHS):
                train_loss, train_acc = self._train_one_epoch(epoch)
                test_acc              = self.evaluate()

                self.client_scheduler.step(test_acc)
                self.server_scheduler.step(test_acc)

                print(
                    f"  Epoch {epoch+1:3d}/{Config.EPOCHS} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Train: {train_acc:.2f}% | "
                    f"Test: {test_acc:.2f}%")

                if test_acc > best_acc:
                    best_acc = test_acc
                    self._save_checkpoint(epoch, best_acc)

                self._save_latest_checkpoint(epoch, test_acc)

            print(f"\n  PGSL training complete. Best Accuracy: {best_acc:.2f}%")
            return self.train_losses, self.train_accuracies, self.test_accuracies

    def _save_checkpoint(self, epoch, best_acc):
        save_path = f"{Config.SAVE_DIR}/best_pgsl_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"
        torch.save({
            'epoch'        : epoch,
            'client_state' : self.client_model.state_dict(),
            'server_state' : self.server_model.state_dict(),
            'best_acc'     : best_acc,
            'dataset'      : Config.DATASET,
            'model_type'   : 'PGSL',
            'model_name'   : Config.MODEL_NAME,
            'cut_layer'    : Config.CUT_LAYER,
        }, save_path)
        
        
    def _save_latest_checkpoint(self, epoch, test_acc):
        save_path = f"{Config.SAVE_DIR}/latest_pgsl_{Config.MODEL_NAME.lower()}_sl_{Config.DATASET}.pth"
        torch.save({
            'epoch'        : epoch,
            'client_state' : self.client_model.state_dict(),
            'server_state' : self.server_model.state_dict(),
            'client_optimizer_state': self.client_optimizer.state_dict(),
            'server_optimizer_state': self.server_optimizer.state_dict(),
            'test_acc'     : test_acc,
            'dataset'      : Config.DATASET,
            'model_type'   : 'PGSL',
            'model_name'   : Config.MODEL_NAME,
            'cut_layer'    : Config.CUT_LAYER,
        }, save_path)

    def save_results(self):
        df = pd.DataFrame({
            'epoch'         : range(1, len(self.train_losses) + 1),
            'train_loss'    : self.train_losses,
            'train_accuracy': self.train_accuracies,
            'test_accuracy' : self.test_accuracies})
        path = f"{Config.RESULTS_DIR}/pgsl_sl_results_{Config.MODEL_NAME.lower()}_{Config.DATASET}.csv"
        df.to_csv(path, index=False)
        print(f"  Results saved → {path}")