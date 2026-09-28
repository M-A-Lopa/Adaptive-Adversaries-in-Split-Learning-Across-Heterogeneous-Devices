"""
Private vanilla SL training of Khan et al. (Alg. 1 client / Alg. 2 servers), following the official
procedure.py: MSE loss on one-hot targets, SGD + momentum on both sides, retry a batch when the revealed
loss exceeds 15 (fixed-point truncation failure guard).
Paper hyper-parameters (Sec. 8.2): E=10, lr=0.002, momentum=0.9, batch=128, last incomplete batch dropped.
The secret-shared server runs on CPU (int64 matmul is not supported on CUDA; the paper also used CPU).
"""
import math
import time
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

from config import Config
from all_defences.splitfss_defense import SplitFSSServer


class SplitFSSTrainer:
    PAPER_EPOCHS, PAPER_LR, PAPER_MOMENTUM, PAPER_BATCH = 10, 0.002, 0.9, 128
    MAX_RETRIES = 5

    def __init__(self, client_model, plain_server_linears, train_loader, test_loader,
                 epochs=PAPER_EPOCHS, lr=PAPER_LR, momentum=PAPER_MOMENTUM, tag="splitfss"):
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.client_model = client_model.to(self.device)
        self.server = SplitFSSServer(plain_server_linears, final_relu=True, precision_fractional=5,
                                     lr=lr, momentum=momentum)
        self.client_optimizer = torch.optim.SGD(self.client_model.parameters(), lr=lr, momentum=momentum)
        self.train_loader, self.test_loader = train_loader, test_loader
        self.epochs, self.tag = epochs, tag
        self.num_classes = Config.NUM_CLASSES
        self.train_losses, self.train_accuracies, self.test_accuracies, self.epoch_times = [], [], [], []
        self.retries = 0

    def _secure_step(self, smashed, labels):
        onehot = F.one_hot(labels.cpu(), self.num_classes).double()
        for attempt in range(self.MAX_RETRIES):
            X = self.server.client_share(smashed.detach().cpu())
            out = self.server.forward(X)
            loss, dout = self.server.mse_loss_grad(out, onehot)
            if math.isfinite(loss) and abs(loss) <= 15:
                break
            self.retries += 1
            print(f"  ⚠️ loss:{loss:.3e} RETRY ({attempt+1}/{self.MAX_RETRIES})")
        dx = self.server.backward_and_step(dout)
        grad_client = self.server.reveal(dx).reshape(smashed.shape).to(self.device)   # client adds both shares
        return loss, self.server.reveal(out), grad_client

    def _train_one_epoch(self, epoch):
        self.client_model.train()
        running, correct, total = 0.0, 0, 0
        bar = tqdm(self.train_loader, desc=f"  SplitFSS [{epoch+1}/{self.epochs}]", leave=False)
        for inputs, labels in bar:
            inputs = inputs.to(self.device)
            self.client_optimizer.zero_grad()
            smashed = self.client_model(inputs)
            loss, preds, grad = self._secure_step(smashed, labels)
            smashed.backward(grad)
            self.client_optimizer.step()
            running += loss
            correct += preds.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
            bar.set_postfix({'Loss': f'{loss:.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
        self.train_losses.append(running / len(self.train_loader))
        self.train_accuracies.append(100. * correct / total)
        return self.train_losses[-1], self.train_accuracies[-1]

    def evaluate(self):
        self.client_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                smashed = self.client_model(inputs.to(self.device))
                preds = self.server.reveal(self.server.forward(self.server.client_share(smashed.cpu())))
                correct += preds.argmax(1).eq(labels).sum().item()
                total += labels.size(0)
        return 100. * correct / total

    def train(self):
        print("\n" + "=" * 60)
        print("   SplitFSS (SL + FSS) — PRIVATE VANILLA SL TRAINING")
        print("=" * 60)
        print(f"  Dataset: {Config.DATASET} | Epochs: {self.epochs} | lr: {self.server.lr} | "
              f"momentum: {self.server.momentum} | fixed-point: 10^{self.server.fp.prec}")
        best = 0.0
        for epoch in range(self.epochs):
            t0 = time.time()
            loss, acc = self._train_one_epoch(epoch)
            test_acc = self.evaluate()
            self.test_accuracies.append(test_acc)
            self.epoch_times.append(time.time() - t0)
            print(f"  Epoch {epoch+1:3d}/{self.epochs} | Loss: {loss:.4f} | Train: {acc:.2f}% | "
                  f"Test: {test_acc:.2f}% | {self.epoch_times[-1]/60:.1f} min")
            if test_acc > best:
                best = test_acc
                self._save_checkpoint(epoch, best)
        print(f"\n  Training complete. Best Test Accuracy: {best:.2f}% | batch retries: {self.retries}")
        return best

    def _save_checkpoint(self, epoch, best_acc):
        torch.save({'epoch': epoch, 'client_state': self.client_model.state_dict(),
                    'server_plain_state_ANALYSIS_ONLY': self.server.export_plain_state(),
                    'best_acc': best_acc, 'dataset': Config.DATASET},
                   f"{Config.SAVE_DIR}/best_{self.tag}_{Config.DATASET}.pth")

    def save_results(self):
        comm = self.server.comm_report()
        df = pd.DataFrame({'epoch': range(1, len(self.train_losses) + 1), 'train_loss': self.train_losses,
                           'train_accuracy': self.train_accuracies, 'test_accuracy': self.test_accuracies,
                           'epoch_time_s': self.epoch_times})
        for k, v in comm.items():
            df[k] = v
        df.to_csv(f"{Config.RESULTS_DIR}/{self.tag}_training_{Config.DATASET}.csv", index=False)
        return comm
