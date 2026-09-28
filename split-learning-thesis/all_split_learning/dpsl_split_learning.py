

import torch
import torch.optim as optim
import pandas as pd

from config import Config
from all_split_learning.split_learning import SplitLearningTrainer


class DPSLSplitLearningTrainer(SplitLearningTrainer):

    def __init__(self, client_model, server_model, train_loader, test_loader,
                 epochs=100, run_name='dpsl'):
        self.epochs = epochs
        self.run_name = run_name
        Config.EPOCHS = epochs                      # used by the parent's progress bar

        # Materialise LazyLinear in the server BEFORE the optimiser is created
        device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
        client_model.to(device).eval(); server_model.to(device).eval()
        x0 = next(iter(train_loader))[0][:2].to(device)
        with torch.no_grad():
            server_model(client_model(x0))

        super().__init__(client_model, server_model, train_loader, test_loader)

        # Paper: Adam (default params) + cosine annealing, T_max = number of epochs
        self.client_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.client_optimizer, T_max=epochs)
        self.server_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.server_optimizer, T_max=epochs)

    def train(self):
        print("\n" + "=" * 60)
        print(f"   DP-SL (Pham et al.) — {Config.MODEL_NAME} / {Config.DATASET}  [{self.run_name}]")
        print("=" * 60)
        print(f"  Cut layer  : {Config.CUT_LAYER}")
        print(f"  Epochs     : {self.epochs}  (Adam lr={Config.LEARNING_RATE}, cosine annealing)")
        print(f"  Batch size : {Config.BATCH_SIZE}")
        eps = getattr(self.client_model, 'epsilon', None)
        sig = getattr(self.client_model, 'sigma', 0.0)
        print(f"  Epsilon    : {'no noise' if eps is None else eps}   sigma = {sig:.4f}")
        print("=" * 60 + "\n")

        best_acc = 0.0
        for epoch in range(self.epochs):
            train_loss, train_acc = self._train_one_epoch(epoch)
            test_acc = self._evaluate()
            self.client_scheduler.step()
            self.server_scheduler.step()
            print(f"  Epoch {epoch+1:3d}/{self.epochs} | Loss: {train_loss:.4f} | "
                  f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
            if test_acc > best_acc:
                best_acc = test_acc
                self._save_checkpoint(epoch, best_acc)

        self.final_test_acc = self.test_accuracies[-1]
        print(f"\n  Done. Final test acc: {self.final_test_acc:.2f}%  |  Best: {best_acc:.2f}%")
        return self.train_losses, self.train_accuracies, self.test_accuracies

    def _save_checkpoint(self, epoch, best_acc):
        path = f"{Config.SAVE_DIR}/best_{Config.MODEL_NAME.lower()}_{self.run_name}_{Config.DATASET}.pth"
        torch.save({'epoch': epoch,
                    'client_state': self.client_model.state_dict(),
                    'server_state': self.server_model.state_dict(),
                    'best_acc': best_acc,
                    'dataset': Config.DATASET,
                    'cut_layer': Config.CUT_LAYER,
                    'epsilon': getattr(self.client_model, 'epsilon', None)}, path)

    def save_results(self):
        df = pd.DataFrame({'epoch': range(1, len(self.train_losses) + 1),
                           'train_loss': self.train_losses,
                           'train_accuracy': self.train_accuracies,
                           'test_accuracy': self.test_accuracies})
        path = f"{Config.RESULTS_DIR}/{Config.MODEL_NAME}_{self.run_name}_results_{Config.DATASET}.csv"
        df.to_csv(path, index=False)
        print(f"  Results saved → {path}")