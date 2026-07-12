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
from all_defences.pgsl_defense import AdaptiveWeightedDecisionFusion, PGSLDefenseModules


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
        self.client_optimizer = optim.Adam(self.client_model.parameters(), lr=Config.LEARNING_RATE)
        self.server_optimizer = optim.Adam( self.server_model.parameters(), lr=Config.LEARNING_RATE)

        # Learning rate scheduler — reduces LR when accuracy plateaus
        self.client_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.client_optimizer, patience=5, factor=0.5, verbose=True)
        self.server_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.server_optimizer, patience=5, factor=0.5, verbose=True)

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

            # =========================================================================
            # PHASE 1: Connected Baseline Pass (Graph Intact for Jacobian Calculation)
            # =========================================================================
            inputs.requires_grad_(True)
            
            # Downsample the raw unperturbed image with the empty tracking placeholder channel
            baseline_tensor = PGSLDefenseModules.space_to_depth_downsample(inputs, saliency_map=None)
            
            # Forward pass with autograd ACTIVE so the graph links inputs -> s_baseline -> baseline_logits
            s_baseline = self.client_model(baseline_tensor)
            
            # Fast evaluation pass through the server's attacked stream head
            baseline_logits, _, _ = self.server_model(s_baseline, run_full_pipeline=False)

            # =========================================================================
            # PHASE 2: True Saliency Generation, Disconnection, and Defensive Training
            # =========================================================================
            # Generate the faithful quantitative reversed JSMA map using the active graph
            x_perturbed, map_tensor = PGSLDefenseModules.generate_faithful_saliency_map(
                inputs=inputs,
                baseline_logits=baseline_logits,
                num_classes=Config.NUM_CLASSES
            )

            # Crucial Step: Clear the active optimization gradients across both modules 
            # to completely wipe any intermediate tracking artifacts from the Phase 1 Jacobian computation
            self.client_optimizer.zero_grad()
            self.server_optimizer.zero_grad()

            # Construct the true 4ch + 1 active tensor layout using the newly generated maps
            client_tensor = PGSLDefenseModules.space_to_depth_downsample(x_perturbed, map_tensor)

            # Forward Pass to create actual training intermediate activations
            s_client_output = self.client_model(client_tensor)

            # --- TRUE GRADIENT TRANSMISSION ISOLATION ---
            # Detach to create an explicit server-side leaf node, cutting graph communication back to the client
            s_server_leaf = s_client_output.detach().requires_grad_(True)

            # Process through the server's separate backbones
            out_a, out_r, out_f = self.server_model(s_server_leaf, run_full_pipeline=True)

            # Compute the three distinct structural losses
            loss_a = self.criterion(out_a, labels)
            loss_r = self.criterion(out_r, labels)
            loss_f = self.criterion(out_f, labels)

            # 1. Backpropagate Attacked and Recovered losses. Gradients accumulate on server 
            # parameters, and their impact over s_server_leaf is calculated but trapped there.
            loss_a.backward(retain_graph=True)
            loss_r.backward(retain_graph=True)

            # Zero out any mixed leaf gradients accumulated on s_server_leaf from streams A and R
            s_server_leaf.grad.zero_()

            # 2. Backpropagate Fused stream loss. 
            loss_f.backward() 

            # Extract the true, unpolluted gradient generated EXCLUSIVELY by the fused stream
            fused_gradient_only = s_server_leaf.grad

            # 3. Manually pass this clean gradient block back to the client graph
            s_client_output.backward(gradient=fused_gradient_only)

            # Complete structural optimization parameter update steps
            self.server_optimizer.step()
            self.client_optimizer.step()

            # Adaptive Late Decision Fusion logic for metric tracking and evaluation
            final_logits = AdaptiveWeightedDecisionFusion.fuse_outputs(out_a, out_r, out_f)

            # --------------------------Track metrics--------------------------
            running_loss += loss_f.item()  # Tracking the main defensive fusion loss
            _, predicted  = final_logits.max(1)
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
                }, f"{Config.SAVE_DIR}/best_vanilla_sl_{Config.DATASET}.pth")

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
        # dataset name included in filename
        path = f"{Config.RESULTS_DIR}/vanilla_sl_results_{Config.DATASET}.csv"
        df.to_csv(path, index=False)
        print(f"Results saved → {path}")