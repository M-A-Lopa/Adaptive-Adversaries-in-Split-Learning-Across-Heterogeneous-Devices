# ressfl_split_learning.py
# ResSFL Split Learning Trainer
# Based on: MIA_torch.py from ResSFL repository
#
# Implements the adversarial attacker-aware training (gan_adv regularization).
# The core adversarial game per batch (V2_epoch scheme):
#
#   Step 1 — AE training (maximize reconstruction quality):
#     for _ in range(gan_num_step=3):
#       z = client(x).detach()          # frozen client
#       recon = AE(z)
#       loss = -SSIM(recon, x_denorm)   # maximize SSIM
#       loss.backward(); ae_optimizer.step()
#
#   Step 2 — Client + Server training (minimize classification loss
#             AND minimize reconstruction quality):
#     optimizer.zero_grad(); local_optimizer.zero_grad()
#     z = client(x)                     # connected to graph
#     out = server(z)                   # connected to graph
#     ce_loss = CrossEntropy(out, y)
#
#     AE.eval()                         # freeze AE BN/dropout
#     recon = AE(z)                     # z still connected to client
#     ssim_val = SSIM(recon, x_denorm)
#     if ssim_val > ssim_threshold:     # only penalize above threshold
#       gan_loss = α * (ssim_val - ssim_threshold)
#     total_loss = ce_loss + gan_loss
#     total_loss.backward()             # gradient flows through z to client
#     optimizer.step(); local_optimizer.step()
#
# Optimizers (following original):
#   Client + Server: SGD, lr=0.1, momentum=0.9, weight_decay=5e-4
#   AE:              Adam, lr=1e-3
#   LR schedule:     MultiStepLR at [15, 30, 40] for 50-epoch training
#   Warmup:          1 epoch linear warmup on both SGD optimizers


import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

from config import Config
from all_defences.ressfl_defense import WindowedSSIM, denormalize
from all_model.ressfl_models import build_ae


# ── Warmup LR Scheduler ───────────────────────────────────────────────────────
class WarmUpLR(lr_scheduler._LRScheduler):
    """
    Linear warmup scheduler from utils.py in ResSFL repository.
    Linearly increases LR from 0 to base_lr over total_iters steps.
    Called once per batch during the warmup epoch.

    total_iters = num_batches * warm_epochs (warm_epochs=1 in original)
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


# ── ResSFL Trainer ────────────────────────────────────────────────────────────
class ResSFLTrainer:
    """
    ResSFL adversarial split learning trainer.

    Adapts MIA_torch.MIA for our shallow 3-conv architecture with:
    - Single client (num_client=1)
    - V2_epoch training scheme
    - gan_adv regularization with SSIM loss
    - ssim_threshold=0.4 (prevents accuracy collapse on shallow model)
    - Custom AE auto-sized to match our smashed data dimensions

    Architecture compatibility:
    - ClientModel (from models.py): unchanged, produces smashed [B,32,H,W]
    - ServerModel (from models.py): unchanged, takes smashed data, outputs logits
    - custom_AE: auto-sized by build_ae() to match smashed shape per dataset
    """

    # ── Class-level defaults matching original ResSFL paper ──────────────────
    WARM_EPOCHS  = 1      # epochs of linear warmup
    GAN_NUM_STEP = 3      # AE gradient steps per client step
    AE_INTERVAL  = 1      # train AE every N batches (1 = every batch)
    ALPHA2       = 1.0    # α: weight of SSIM regularization term
    SSIM_THRESH  = 0.4    # ssim_threshold: only penalize if SSIM > threshold
    SGD_LR       = 0.1    # SGD learning rate for client and server
    SGD_MOMENTUM = 0.9
    SGD_WD       = 5e-4
    AE_LR        = 1e-3   # Adam lr for the AE decoder

    def __init__(self, client_model, server_model, train_loader, test_loader):
        self.device = torch.device(
            Config.DEVICE if torch.cuda.is_available() else 'cpu'
        )
        print(f"  Device: {self.device}")

        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.dataset      = Config.DATASET

        # ── AE decoder: attacker model that client trains against ─────────────
        self.local_ae = build_ae(dataset=self.dataset,
                                 activation='sigmoid').to(self.device)
        print(f"  AE decoder parameters: "
              f"{sum(p.numel() for p in self.local_ae.parameters()):,}")

        # ── Optimizers ────────────────────────────────────────────────────────
        # Server and client both use SGD with same hyperparameters
        # (original uses self.optimizer for server, local_optimizer for client)
        self.server_optimizer = optim.SGD(
            self.server_model.parameters(),
            lr=self.SGD_LR,
            momentum=self.SGD_MOMENTUM,
            weight_decay=self.SGD_WD
        )
        self.client_optimizer = optim.SGD(
            self.client_model.parameters(),
            lr=self.SGD_LR,
            momentum=self.SGD_MOMENTUM,
            weight_decay=self.SGD_WD
        )
        self.ae_optimizer = optim.Adam(
            self.local_ae.parameters(), lr=self.AE_LR
        )

        # ── LR schedulers ─────────────────────────────────────────────────────
        # Original: MultiStepLR at [60, 120, 160] / 200 epochs, gamma=0.2
        # Scaled to 50 epochs: [15, 30, 40]
        milestones = [15, 30, 40]
        self.server_scheduler = optim.lr_scheduler.MultiStepLR(
            self.server_optimizer, milestones=milestones, gamma=0.2
        )
        self.client_scheduler = optim.lr_scheduler.MultiStepLR(
            self.client_optimizer, milestones=milestones, gamma=0.2
        )
        self.ae_scheduler = optim.lr_scheduler.MultiStepLR(
            self.ae_optimizer, milestones=milestones, gamma=0.2
        )

        # ── Warmup schedulers ─────────────────────────────────────────────────
        num_batches    = len(train_loader)
        total_warmup   = num_batches * self.WARM_EPOCHS
        self.server_warmup = WarmUpLR(self.server_optimizer, total_warmup)
        self.client_warmup = WarmUpLR(self.client_optimizer, total_warmup)

        self.criterion = nn.CrossEntropyLoss()
        self.ssim_loss = WindowedSSIM()

        # Result tracking
        self.train_losses     = []
        self.train_accuracies = []
        self.test_accuracies  = []

        os.makedirs(Config.SAVE_DIR,    exist_ok=True)
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

    # ── AE pre-training ───────────────────────────────────────────────────────
    def pretrain_ae(self, pre_epochs=30, ae_batch_size=32,
                    collect_batches=80):
        """
        Pre-trains the AE decoder on (denorm_image, smashed_data) pairs
        collected from the frozen client model before main training begins.

        Mirrors pre_GAN_train() from MIA_torch.py:
        1. Collect pairs by running training data through frozen client
        2. Train AE for pre_epochs to learn reconstruction from smashed data
        3. Best checkpoint saved by minimum validation MSE

        collect_batches: number of training batches to collect pairs from
        ae_batch_size: batch size for AE training (32 in original)
        """
        print("\n" + "="*60)
        print("  RESSFL — AE PRE-TRAINING")
        print(f"  Pre-training AE for {pre_epochs} epochs on frozen client")
        print("="*60)

        self.client_model.eval()

        # ── Step 1: Collect (denorm_image, smashed_data) pairs ───────────────
        all_imgs    = []
        all_smashed = []

        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_loader):
                if batch_idx >= collect_batches:
                    break
                inputs = inputs.to(self.device)
                smashed = self.client_model(inputs)
                # Denormalize: must match what train_target_step computes
                imgs_dn = denormalize(inputs, self.dataset)
                all_imgs.append(imgs_dn.cpu())
                all_smashed.append(smashed.cpu())

        all_imgs    = torch.cat(all_imgs,    dim=0)
        all_smashed = torch.cat(all_smashed, dim=0)

        print(f"  Collected {len(all_imgs)} image-activation pairs")
        print(f"  Smashed data shape: {all_smashed.shape}")

        # ── Step 2: Train/val split (90/10 as in original) ───────────────────
        n       = len(all_imgs)
        n_train = int(0.9 * n)

        ae_train = DataLoader(
            TensorDataset(all_imgs[:n_train], all_smashed[:n_train]),
            batch_size=ae_batch_size, shuffle=True
        )
        ae_val = DataLoader(
            TensorDataset(all_imgs[n_train:], all_smashed[n_train:]),
            batch_size=ae_batch_size, shuffle=False
        )

        # ── Step 3: Train AE ──────────────────────────────────────────────────
        pre_criterion = nn.MSELoss()  # MSE for pre-training (as in original)
        pre_optimizer = optim.Adam(self.local_ae.parameters(), lr=self.AE_LR)

        best_val_mse  = float('inf')
        best_state    = None

        for epoch in range(pre_epochs):
            self.local_ae.train()
            train_mse = 0.0

            for imgs, smashed in ae_train:
                imgs    = imgs.to(self.device)
                smashed = smashed.to(self.device)

                pre_optimizer.zero_grad()
                recon = self.local_ae(smashed)
                loss  = pre_criterion(recon, imgs)
                loss.backward()
                pre_optimizer.step()
                train_mse += loss.item()

            train_mse /= len(ae_train)

            # Validation
            self.local_ae.eval()
            val_mse = 0.0
            with torch.no_grad():
                for imgs, smashed in ae_val:
                    imgs    = imgs.to(self.device)
                    smashed = smashed.to(self.device)
                    recon   = self.local_ae(smashed)
                    val_mse += pre_criterion(recon, imgs).item()
            val_mse /= len(ae_val)

            # Save best by minimum validation MSE (mirrors original self.attack())
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state   = {k: v.clone() for k, v in
                                self.local_ae.state_dict().items()}

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Pre-train Epoch {epoch+1:3d}/{pre_epochs} | "
                      f"Train MSE: {train_mse:.5f} | "
                      f"Val MSE: {val_mse:.5f}")

        # Restore best AE weights
        if best_state is not None:
            self.local_ae.load_state_dict(best_state)

        print(f"\n  AE pre-training complete. Best Val MSE: {best_val_mse:.5f}")

    # ── AE training step (gan_train_step) ─────────────────────────────────────
    def _gan_train_step(self, inputs):
        """
        Trains the AE decoder to MAXIMIZE reconstruction quality.
        Called gan_num_step=3 times per batch in main training loop.

        Direct port of gan_train_step() from MIA_torch.py:
          z = client(x).detach()         [frozen client — no gradient to client]
          x_denorm = denormalize(x)
          recon = AE(z)
          loss = -SSIM(recon, x_denorm)  [maximize SSIM = minimize -SSIM]
          backward + step AE only

        AE is in train mode here. Client is in eval mode (frozen).
        z is detached so AE gradient does NOT flow back to client.
        """
        inputs = inputs.to(self.device)

        self.client_model.eval()
        with torch.no_grad():
            z_detached = self.client_model(inputs)

        self.local_ae.train()
        x_denorm = denormalize(inputs, self.dataset)

        self.ae_optimizer.zero_grad()
        recon = self.local_ae(z_detached)
        # Negative SSIM: minimizing this = maximizing reconstruction quality
        ae_loss = -self.ssim_loss(recon, x_denorm)
        ae_loss.backward()
        self.ae_optimizer.step()

        # Return positive SSIM value for logging (as in original ssim_log = -loss)
        return (-ae_loss).item()

    # ── Client + Server training step (train_target_step) ────────────────────
    def _train_target_step(self, inputs, labels):
        """
        Trains client and server with CE loss + SSIM adversarial regularization.
        Direct port of train_target_step() from MIA_torch.py.

        Key: NO detach between client and server — full graph is connected.
        This allows SSIM gradient to flow directly through z to client params.

        The AE is in eval mode here (BN uses running stats, no AE weight updates).
        The SSIM term trains the client to produce smashed data the AE CANNOT
        reconstruct well — the defense mechanism.

        ssim_threshold=0.4: Only penalize when SSIM > 0.4.
        This prevents the client from sacrificing too much accuracy
        when SSIM is already acceptably low (original paper recommendation).
        """
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        self.client_model.train()
        self.server_model.train()

        self.client_optimizer.zero_grad()
        self.server_optimizer.zero_grad()

        # ── Forward pass (graph fully connected) ──────────────────────────────
        z      = self.client_model(inputs)   # [B, 32, H, W] — connected to graph
        output = self.server_model(z)        # [B, num_classes]

        # ── Classification loss ───────────────────────────────────────────────
        ce_loss    = self.criterion(output, labels)
        total_loss = ce_loss

        # ── SSIM adversarial regularization ───────────────────────────────────
        # AE in eval mode: BN uses running stats, weights frozen
        self.local_ae.eval()
        recon    = self.local_ae(z)   # z is still connected to client graph
        x_denorm = denormalize(inputs, self.dataset)
        ssim_val = self.ssim_loss(recon, x_denorm)

        # Only penalize if SSIM exceeds threshold (prevents over-regularization)
        if self.SSIM_THRESH > 0.0:
            if ssim_val.item() > self.SSIM_THRESH:
                # Push SSIM toward threshold, not toward 0
                gan_loss    = self.ALPHA2 * (ssim_val - self.SSIM_THRESH)
                total_loss  = total_loss + gan_loss
        else:
            # No threshold: always minimize SSIM (original default behavior)
            total_loss = total_loss + self.ALPHA2 * ssim_val

        # ── Backward and update ───────────────────────────────────────────────
        total_loss.backward()
        self.server_optimizer.step()
        self.client_optimizer.step()

        return total_loss.item(), ce_loss.item(), ssim_val.item()

    # ── One epoch ─────────────────────────────────────────────────────────────
    def _train_one_epoch(self, epoch):
        """
        V2_epoch training scheme (from MIA_torch.py):
        For each batch:
          1. Train AE (gan_num_step times) to maximize reconstruction
          2. Train client + server to minimize CE + SSIM regularization

        The ordering — AE first, then client — ensures the client always
        defends against the most up-to-date attacker at each step.
        """
        running_ce   = 0.0
        running_ssim = 0.0
        correct      = 0
        total        = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"  ResSFL Epoch [{epoch+1}/{Config.EPOCHS}]",
            leave=False
        )

        for batch_idx, (inputs, labels) in enumerate(progress_bar):

            # ── Step 1: AE training (every AE_INTERVAL batches) ──────────────
            # Original: if self.gan_regularizer and batch % interval == 0
            if batch_idx % self.AE_INTERVAL == 0:
                ssim_log = 0.0
                for _ in range(self.GAN_NUM_STEP):
                    ssim_log = self._gan_train_step(inputs)

            # ── Step 2: Client + Server training ─────────────────────────────
            total_loss, ce_loss, ssim_val = self._train_target_step(
                inputs, labels
            )

            # ── Track metrics ─────────────────────────────────────────────────
            running_ce   += ce_loss
            running_ssim += ssim_val
            total        += labels.size(0)

            # Accuracy from classification output
            with torch.no_grad():
                inputs_d = inputs.to(self.device)
                labels_d = labels.to(self.device)
                self.client_model.eval()
                self.server_model.eval()
                z_eval   = self.client_model(inputs_d)
                out_eval = self.server_model(z_eval)
                _, predicted = out_eval.max(1)
                correct += predicted.eq(labels_d).sum().item()
                self.client_model.train()
                self.server_model.train()

            progress_bar.set_postfix({
                'CE':   f'{ce_loss:.4f}',
                'SSIM': f'{ssim_val:.4f}',
                'Acc':  f'{100.*correct/total:.1f}%'
            })

        epoch_loss = running_ce / len(self.train_loader)
        epoch_acc  = 100. * correct / total

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)

        return epoch_loss, epoch_acc

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate(self):
        """
        Standard classification accuracy on test set.
        Mirrors validate_target() in MIA_torch.py.
        No defense applied during evaluation — clean forward pass.
        """
        self.client_model.eval()
        self.server_model.eval()

        correct = 0
        total   = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                z      = self.client_model(inputs)
                output = self.server_model(z)
                _, predicted = output.max(1)
                total   += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        self.test_accuracies.append(test_acc)
        return test_acc

    # ── Full training loop ────────────────────────────────────────────────────
    def train(self):
        """
        Full training pipeline:
        1. Pre-train AE (30 epochs on frozen client)
        2. Main V2_epoch adversarial training (Config.EPOCHS epochs)

        LR warmup applied during epoch 1 (per-batch).
        MultiStepLR scheduler applied from epoch 2 onward (per-epoch).
        """
        print("\n" + "="*60)
        print("   RESSFL SPLIT LEARNING — TRAINING")
        print("="*60)
        print(f"  Dataset       : {Config.DATASET}")
        print(f"  Epochs        : {Config.EPOCHS}")
        print(f"  SGD LR        : {self.SGD_LR}")
        print(f"  AE LR         : {self.AE_LR}")
        print(f"  α (SSIM reg.) : {self.ALPHA2}")
        print(f"  SSIM threshold: {self.SSIM_THRESH}")
        print(f"  AE steps/batch: {self.GAN_NUM_STEP}")
        print("="*60 + "\n")

        # Phase 1: AE pre-training (mirrors pre_GAN_train(30) in original)
        self.pretrain_ae(pre_epochs=30)

        # Phase 2: Main adversarial training
        best_acc = 0.0

        for epoch in range(Config.EPOCHS):

            # Warmup: step per-batch during epoch 1
            # MultiStepLR: step per-epoch after warmup
            use_warmup = (epoch < self.WARM_EPOCHS)

            train_loss, train_acc = self._train_one_epoch(epoch)
            test_acc              = self.evaluate()

            # Scheduler steps
            if epoch >= self.WARM_EPOCHS:
                self.server_scheduler.step()
                self.client_scheduler.step()
                self.ae_scheduler.step()

            print(
                f"  Epoch {epoch+1:3d}/{Config.EPOCHS} | "
                f"Loss: {train_loss:.4f} | "
                f"Train: {train_acc:.2f}% | "
                f"Test: {test_acc:.2f}%"
            )

            if test_acc > best_acc:
                best_acc = test_acc
                self._save_checkpoint(epoch, best_acc)

        print(f"\n  ResSFL training complete. Best Accuracy: {best_acc:.2f}%")
        return self.train_losses, self.train_accuracies, self.test_accuracies

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch, best_acc):
        """
        Saves client, server, and AE decoder states.
        AE is saved separately since it is not part of main inference.
        Separate filename prevents overwriting PGSL or vanilla checkpoints.
        """
        path = f"{Config.SAVE_DIR}/best_ressfl_{Config.DATASET}.pth"
        torch.save({
            'epoch'        : epoch,
            'client_state' : self.client_model.state_dict(),
            'server_state' : self.server_model.state_dict(),
            'ae_state'     : self.local_ae.state_dict(),
            'best_acc'     : best_acc,
            'dataset'      : Config.DATASET,
            'model_type'   : 'ResSFL',
            'ssim_threshold': self.SSIM_THRESH,
            'alpha2'       : self.ALPHA2
        }, path)

    def save_results(self):
        df = pd.DataFrame({
            'epoch'         : range(1, len(self.train_losses) + 1),
            'train_loss'    : self.train_losses,
            'train_accuracy': self.train_accuracies,
            'test_accuracy' : self.test_accuracies
        })
        path = f"{Config.RESULTS_DIR}/ressfl_results_{Config.DATASET}.csv"
        df.to_csv(path, index=False)
        print(f"  Results saved → {path}")