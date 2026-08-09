# pgsl_split_learning.py
# PGSL Split Learning Trainer
#
# Gradient isolation — paper Section IV:
# "The gradients from the last layer of the second stream
# (fused stream) are backpropagated to the last split layer."
#
# Implementation:
# Step 1: backward streams A and R → server params accumulate gradients
#         zero s_leaf.grad to clear A+R contributions from it
# Step 2: backward stream F → server params accumulate F gradients
#         s_leaf.grad now contains ONLY F's gradient
# Step 3: send s_leaf.grad to client (only fused stream gradient)


import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

from config import Config
from all_defences.pgsl_defense import PGSLDefenseModules, AdaptiveWeightedDecisionFusion


class PGSLSplitLearningTrainer:
    """
    Trains PGSLClientModel and PGSLServerModel using the three-stream
    split learning protocol described in the PGSL paper.

    Key difference from vanilla SplitLearningTrainer:
    - Input images are JSMA-perturbed and space_to_depth downsampled
    - Server runs three streams (attacked, recovered, fused)
    - Only fused stream gradient flows back to client
    - Decision-level fusion used for accuracy evaluation
    """

    def __init__(self, client_model, server_model, train_loader, test_loader):
        self.device = torch.device(
            Config.DEVICE if torch.cuda.is_available() else 'cpu'
        )
        print(f"  Device: {self.device}")

        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        self.train_loader = train_loader
        self.test_loader  = test_loader

        self.client_optimizer = optim.Adam(
            self.client_model.parameters(), lr=Config.LEARNING_RATE
        )
        self.server_optimizer = optim.Adam(
            self.server_model.parameters(), lr=Config.LEARNING_RATE
        )

        self.client_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.client_optimizer, patience=5, factor=0.5
        )
        self.server_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.server_optimizer, patience=5, factor=0.5
        )

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

        progress_bar = tqdm(
            self.train_loader,
            desc=f"  PGSL Epoch [{epoch+1}/{Config.EPOCHS}]",
            leave=False
        )

        for inputs, labels in progress_bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # =================================================================
            # PHASE 1: Compute baseline logits for JSMA Jacobian
            # The graph must connect inputs → baseline_logits for autograd
            # =================================================================
            inputs.requires_grad_(True)

            # Downsample with empty saliency placeholder
            baseline_tensor = PGSLDefenseModules.space_to_depth_downsample(
                inputs, saliency_map=None
            )

            # Forward through client (graph intact)
            s_baseline = self.client_model(baseline_tensor)

            # Forward through attacked stream only (fast, no full pipeline)
            baseline_logits, _, _ = self.server_model(
                s_baseline, run_full_pipeline=False
            )

            # =================================================================
            # PHASE 2: Generate JSMA perturbation
            # =================================================================
            x_perturbed, map_tensor = PGSLDefenseModules.generate_faithful_saliency_map(
                inputs=inputs,
                baseline_logits=baseline_logits,
                num_classes=Config.NUM_CLASSES
            )
            # inputs.requires_grad_(False) is called inside generate_faithful_saliency_map

            # =================================================================
            # PHASE 3: PGSL training forward pass with perturbed images
            # =================================================================
            self.client_optimizer.zero_grad()
            self.server_optimizer.zero_grad()

            # Build 4ch+1 tensor from perturbed image and its saliency map
            client_tensor = PGSLDefenseModules.space_to_depth_downsample(
                x_perturbed, saliency_map=map_tensor
            )

            # Client forward pass
            smashed_data = self.client_model(client_tensor)

            # Detach from client graph — simulate network transmission
            # requires_grad_(True) enables gradient message from server to client
            s_leaf = smashed_data.detach().requires_grad_(True)

            # =================================================================
            # PHASE 4: Three-stream server forward
            # =================================================================
            out_a, out_r, out_f = self.server_model(s_leaf, run_full_pipeline=True)

            loss_a = self.criterion(out_a, labels)
            loss_r = self.criterion(out_r, labels)
            loss_f = self.criterion(out_f, labels)

            # =================================================================
            # PHASE 5: Gradient isolation
            # Paper: only fused stream gradient goes back to client
            #
            # Step A: backward A and R → server params accumulate A+R gradients
            #         s_leaf.grad accumulates A+R contributions (unwanted for client)
            loss_a.backward(retain_graph=True)
            loss_r.backward(retain_graph=True)

            # Clear s_leaf.grad so A and R contributions are discarded
            # Server params retain their accumulated A+R gradients
            if s_leaf.grad is not None:
                s_leaf.grad.zero_()

            # Step B: backward F → server params accumulate F gradients
            #         s_leaf.grad now contains ONLY fused stream gradient
            loss_f.backward()

            # Send ONLY fused stream gradient back to client
            smashed_data.backward(gradient=s_leaf.grad)

            # Update both
            self.server_optimizer.step()
            self.client_optimizer.step()

            # =================================================================
            # Accuracy tracking using decision fusion
            # =================================================================
            with torch.no_grad():
                final_logits = AdaptiveWeightedDecisionFusion.fuse_outputs(
                    out_a, out_r, out_f
                )
            _, predicted = final_logits.max(1)

            running_loss += loss_f.item()
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({
                'Loss': f'{loss_f.item():.4f}',
                'Acc' : f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc  = 100. * correct / total

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)

        return epoch_loss, epoch_acc

    def evaluate(self):
        """
        Evaluation using full PGSL pipeline + adaptive decision fusion.
        Inputs go through space_to_depth (no JSMA in eval — too slow),
        then through all three streams, fused for final prediction.
        """
        self.client_model.eval()
        self.server_model.eval()

        correct = 0
        total   = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # During evaluation: no JSMA perturbation (zero saliency map)
                client_tensor = PGSLDefenseModules.space_to_depth_downsample(
                    inputs, saliency_map=None
                )

                # Full three-stream forward
                smashed = self.client_model(client_tensor)
                out_a, out_r, out_f = self.server_model(
                    smashed, run_full_pipeline=True
                )

                # Fuse for final prediction
                final_logits = AdaptiveWeightedDecisionFusion.fuse_outputs(
                    out_a, out_r, out_f
                )

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
        print(f"  Dataset  : {Config.DATASET}")
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
                f"Test: {test_acc:.2f}%"
            )

            if test_acc > best_acc:
                best_acc = test_acc
                self._save_checkpoint(epoch, best_acc)

        print(f"\n  PGSL training complete. Best Accuracy: {best_acc:.2f}%")
        return self.train_losses, self.train_accuracies, self.test_accuracies

    def _save_checkpoint(self, epoch, best_acc):
        """Saves PGSL-specific checkpoint with separate filename."""
        save_path = f"{Config.SAVE_DIR}/best_pgsl_sl_{Config.DATASET}.pth"
        torch.save({
            'epoch'        : epoch,
            'client_state' : self.client_model.state_dict(),
            'server_state' : self.server_model.state_dict(),
            'best_acc'     : best_acc,
            'dataset'      : Config.DATASET,
            'model_type'   : 'PGSL'
        }, save_path)

    def save_results(self):
        df = pd.DataFrame({
            'epoch'         : range(1, len(self.train_losses) + 1),
            'train_loss'    : self.train_losses,
            'train_accuracy': self.train_accuracies,
            'test_accuracy' : self.test_accuracies
        })
        path = f"{Config.RESULTS_DIR}/pgsl_sl_results_{Config.DATASET}.csv"
        df.to_csv(path, index=False)
        print(f"  Results saved → {path}")