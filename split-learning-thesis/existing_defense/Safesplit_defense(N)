

import torch
import torch.nn as nn
import numpy as np
from scipy.fft import dct
from typing import Dict, List, Optional, Tuple
from copy import deepcopy


# ─────────────────────────────────────────────────────────────
# Component 1: DCT-based Static Frequency Analysis
# ─────────────────────────────────────────────────────────────

class FrequencyAnalyzer:
    
    
    def __init__(self,
                 low_freq_ratio: float = 0.1,
                 anomaly_threshold: float = 2.5,
                 window_size: int = 10):
        
        self.low_freq_ratio = low_freq_ratio
        self.anomaly_threshold = anomaly_threshold
        self.window_size = window_size
        
        self._checkpoint = None
        self._distance_history = []
        
        print(f"[SafeSplit] FrequencyAnalyzer: "
              f"low_freq={low_freq_ratio*100:.0f}%, "
              f"threshold={anomaly_threshold}")
    
    def _get_dct_low_freq(self, 
                           model_params: torch.Tensor) -> np.ndarray:
        
        params_np = model_params.detach().cpu().float().numpy().flatten()
        
        # 2D DCT এর জন্য reshape
        n = len(params_np)
        side = int(np.sqrt(n))
        if side * side < n:
            # Pad করে square বানাও
            padded = np.zeros(side * side + side)
            padded[:n] = params_np
            params_2d = padded[:side*(side+1)].reshape(side, side+1)
        else:
            params_2d = params_np[:side*side].reshape(side, side)
        
        # 2D DCT
        dct_2d = dct(dct(params_2d.T, norm='ortho').T, norm='ortho')
        
        # Low-frequency component (top-left corner)
        k = max(1, int(min(params_2d.shape) * self.low_freq_ratio))
        low_freq = dct_2d[:k, :k].flatten()
        
        return low_freq
    
    def save_checkpoint(self, server_model: nn.Module):
        
        # Key layer parameters collect করো
        checkpoint_params = {}
        for name, param in server_model.named_parameters():
            if 'weight' in name:  # Weight layers only
                checkpoint_params[name] = param.data.clone()
        
        self._checkpoint = checkpoint_params
    
    def analyze(self, server_model: nn.Module) -> dict:
        
        if self._checkpoint is None:
            return {
                'frequency_distance': 0.0,
                'is_anomalous': False,
                'anomaly_score': 0.0
            }
        
        total_distance = 0.0
        n_layers = 0
        
        for name, param in server_model.named_parameters():
            if name not in self._checkpoint or 'weight' not in name:
                continue
            
            # Current DCT low-freq
            curr_low = self._get_dct_low_freq(param.data)
            
            # Checkpoint DCT low-freq
            prev_low = self._get_dct_low_freq(self._checkpoint[name])
            
            # Ensure same length
            min_len = min(len(curr_low), len(prev_low))
            curr_low = curr_low[:min_len]
            prev_low = prev_low[:min_len]
            
            # Euclidean distance in frequency domain
            dist = np.linalg.norm(curr_low - prev_low)
            total_distance += dist
            n_layers += 1
        
        avg_distance = total_distance / max(1, n_layers)
        self._distance_history.append(avg_distance)
        
        # Z-score based anomaly detection
        anomaly_score = 0.0
        is_anomalous = False
        
        if len(self._distance_history) >= 3:
            recent = self._distance_history[-self.window_size:]
            mean_d = np.mean(recent[:-1])  # Exclude current
            std_d = np.std(recent[:-1]) + 1e-8
            anomaly_score = (avg_distance - mean_d) / std_d
            
            is_anomalous = anomaly_score > self.anomaly_threshold
        
        return {
            'frequency_distance': avg_distance,
            'is_anomalous': is_anomalous,
            'anomaly_score': anomaly_score
        }


# ─────────────────────────────────────────────────────────────
# Component 2: Rotational Distance Metric (Dynamic Analysis)
# ─────────────────────────────────────────────────────────────

class RotationalDistanceAnalyzer:
    
    
    def __init__(self,
                 threshold: float = 2.0,
                 window_size: int = 10):
        
        self.threshold = threshold
        self.window_size = window_size
        self._checkpoint = None
        self._rotation_history = []
        
        print(f"[SafeSplit] RotationalAnalyzer: "
              f"threshold={threshold}°")
    
    def save_checkpoint(self, server_model: nn.Module):
        
        checkpoint = {}
        for name, param in server_model.named_parameters():
            if 'weight' in name:
                checkpoint[name] = param.data.clone().float()
        self._checkpoint = checkpoint
    
    def _cosine_angle(self,
                      v1: torch.Tensor,
                      v2: torch.Tensor) -> float:
        
        v1_flat = v1.flatten().float()
        v2_flat = v2.flatten().float()
        
        cos_sim = torch.dot(v1_flat, v2_flat) / (
            torch.norm(v1_flat) * torch.norm(v2_flat) + 1e-8
        )
        
        # Clamp to [-1, 1] for numerical stability
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        angle = torch.acos(cos_sim).item() * 180 / np.pi
        
        return angle
    
    def analyze(self, server_model: nn.Module) -> dict:
        
        if self._checkpoint is None:
            return {
                'mean_rotation': 0.0,
                'max_rotation': 0.0,
                'is_anomalous': False,
                'anomaly_score': 0.0
            }
        
        rotations = []
        
        for name, param in server_model.named_parameters():
            if name not in self._checkpoint or 'weight' not in name:
                continue
            
            angle = self._cosine_angle(
                self._checkpoint[name], param.data)
            rotations.append(angle)
        
        if not rotations:
            return {
                'mean_rotation': 0.0,
                'max_rotation': 0.0,
                'is_anomalous': False,
                'anomaly_score': 0.0
            }
        
        mean_rotation = np.mean(rotations)
        max_rotation = np.max(rotations)
        
        self._rotation_history.append(mean_rotation)
        
        # Z-score anomaly detection
        anomaly_score = 0.0
        is_anomalous = False
        
        if len(self._rotation_history) >= 3:
            recent = self._rotation_history[-self.window_size:]
            mean_r = np.mean(recent[:-1])
            std_r = np.std(recent[:-1]) + 1e-8
            anomaly_score = (mean_rotation - mean_r) / std_r
            is_anomalous = anomaly_score > self.threshold
        
        return {
            'mean_rotation': mean_rotation,
            'max_rotation': max_rotation,
            'is_anomalous': is_anomalous,
            'anomaly_score': anomaly_score
        }


# ─────────────────────────────────────────────────────────────
# Component 3: Circular Backward Analysis (Rollback)
# ─────────────────────────────────────────────────────────────

class CircularBackwardAnalysis:
    
    
    def __init__(self, max_checkpoints: int = 5):
        self.max_checkpoints = max_checkpoints
        self._checkpoints = []  # [(epoch, client_id, state_dict)]
        self._flagged_clients = {}  # client_id → flagged_epoch
        self._rollback_count = 0
        
        print(f"[SafeSplit] CircularBackward: "
              f"max_checkpoints={max_checkpoints}")
    
    def save_checkpoint(self,
                        epoch: int,
                        client_id: int,
                        server_model: nn.Module):
        
        state = deepcopy(server_model.state_dict())
        self._checkpoints.append((epoch, client_id, state))
        
        # Keep only recent checkpoints
        if len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints.pop(0)
    
    def rollback(self, server_model: nn.Module,
                 epoch: int, client_id: int) -> bool:
        
        # Find checkpoint before this client's training
        target_checkpoint = None
        for ckpt_epoch, ckpt_client, state in reversed(self._checkpoints):
            if ckpt_client == client_id and ckpt_epoch == epoch:
                target_checkpoint = state
                break
        
        if target_checkpoint is None:
            print(f"  ⚠️  [SafeSplit] No checkpoint found for "
                  f"client {client_id} epoch {epoch}")
            return False
        
        # Restore server model
        server_model.load_state_dict(target_checkpoint)
        self._rollback_count += 1
        
        print(f"  🔄 [SafeSplit] Rolled back server model "
              f"(client={client_id}, epoch={epoch})")
        return True
    
    def flag_client(self, client_id: int, epoch: int):
        """Client কে flag করো — এই epoch skip করবে।"""
        self._flagged_clients[client_id] = epoch
        print(f"  🚩 [SafeSplit] Client {client_id} flagged "
              f"for epoch {epoch} (will retry next epoch)")
    
    def unflag_client(self, client_id: int):
        """Previous flag clear করো।"""
        if client_id in self._flagged_clients:
            del self._flagged_clients[client_id]
    
    def is_flagged(self, client_id: int, epoch: int) -> bool:
        """Client এই epoch-এ flagged কিনা।"""
        return (client_id in self._flagged_clients and 
                self._flagged_clients[client_id] == epoch)


# ─────────────────────────────────────────────────────────────
# Main SafeSplit Defense
# ─────────────────────────────────────────────────────────────

class SafeSplitDefense:
    
    
    def __init__(self,
                 server_model: nn.Module,
                 freq_low_ratio: float = 0.1,
                 freq_threshold: float = 2.5,
                 rot_threshold: float = 2.0,
                 combined_threshold: float = 1.5,
                 max_checkpoints: int = 10,
                 device: str = 'cpu'):
        
        self.device = device
        self.combined_threshold = combined_threshold
        
        # Three components
        self.freq_analyzer = FrequencyAnalyzer(
            low_freq_ratio=freq_low_ratio,
            anomaly_threshold=freq_threshold
        )
        
        self.rot_analyzer = RotationalDistanceAnalyzer(
            threshold=rot_threshold
        )
        
        self.rollback = CircularBackwardAnalysis(
            max_checkpoints=max_checkpoints
        )
        
        # Statistics
        self._total_clients = 0
        self._flagged_count = 0
        self._rollback_count = 0
        self._detection_history = []
        
        print(f"\n{'='*55}")
        print(f"  SafeSplit Defense Initialized")
        print(f"  Freq threshold     : {freq_threshold}")
        print(f"  Rotation threshold : {rot_threshold}°")
        print(f"  Combined threshold : {combined_threshold}")
        print(f"  Max checkpoints    : {max_checkpoints}")
        print(f"{'='*55}\n")
    
    def before_client_training(self,
                                epoch: int,
                                client_id: int,
                                server_model: nn.Module):
      
        # Save checkpoint for rollback
        self.rollback.save_checkpoint(epoch, client_id, server_model)
        
        # Save checkpoint for analysis
        self.freq_analyzer.save_checkpoint(server_model)
        self.rot_analyzer.save_checkpoint(server_model)
    
    def after_client_training(self,
                               epoch: int,
                               client_id: int,
                               server_model: nn.Module) -> bool:
        
        self._total_clients += 1
        
        # Static Analysis: DCT frequency
        freq_result = self.freq_analyzer.analyze(server_model)
        
        # Dynamic Analysis: Rotational distance
        rot_result = self.rot_analyzer.analyze(server_model)
        
        # Combined decision
        # Both analyses must agree (AND logic for lower false positive)
        freq_anomalous = freq_result['is_anomalous']
        rot_anomalous = rot_result['is_anomalous']
        
        # Combined score (average of z-scores)
        combined_score = (freq_result['anomaly_score'] + 
                         rot_result['anomaly_score']) / 2
        
        is_poisoned = (freq_anomalous and rot_anomalous) or \
                      (combined_score > self.combined_threshold * 2)
        
        # Record detection
        detection = {
            'epoch': epoch,
            'client_id': client_id,
            'freq_distance': freq_result['frequency_distance'],
            'freq_anomaly_score': freq_result['anomaly_score'],
            'rotation': rot_result['mean_rotation'],
            'rot_anomaly_score': rot_result['anomaly_score'],
            'combined_score': combined_score,
            'is_poisoned': is_poisoned
        }
        self._detection_history.append(detection)
        
        if is_poisoned:
            self._flagged_count += 1
            
            print(f"\n  🚨 [SafeSplit] BACKDOOR DETECTED!")
            print(f"     Client ID    : {client_id}")
            print(f"     Epoch        : {epoch}")
            print(f"     Freq score   : {freq_result['anomaly_score']:.2f}")
            print(f"     Rot score    : {rot_result['anomaly_score']:.2f}")
            print(f"     Combined     : {combined_score:.2f}")
            
            # Rollback server model
            rolled_back = self.rollback.rollback(
                server_model, epoch, client_id)
            
            if rolled_back:
                self._rollback_count += 1
            
            # Flag client (skip this epoch, retry next)
            self.rollback.flag_client(client_id, epoch)
            
        else:
            # Unflag if previously flagged
            self.rollback.unflag_client(client_id)
            
            if len(self._detection_history) % 10 == 0:
                print(f"  ✅ [SafeSplit] Client {client_id} "
                      f"(epoch {epoch}) — benign "
                      f"[freq={freq_result['anomaly_score']:.2f}, "
                      f"rot={rot_result['anomaly_score']:.2f}]")
        
        return is_poisoned
    
    def should_skip_client(self,
                           client_id: int,
                           epoch: int) -> bool:
        
        return self.rollback.is_flagged(client_id, epoch - 1)
    
    def get_stats(self) -> dict:
        """Defense statistics।"""
        detection_rate = (self._flagged_count / 
                         max(1, self._total_clients) * 100)
        return {
            'total_clients_analyzed': self._total_clients,
            'flagged_as_poisoned': self._flagged_count,
            'rollbacks_performed': self._rollback_count,
            'detection_rate_pct': detection_rate,
            'checkpoints_stored': len(self.rollback._checkpoints)
        }
    
    def print_stats(self):
        stats = self.get_stats()
        print(f"\n[SafeSplit Statistics]")
        print(f"  Clients analyzed : {stats['total_clients_analyzed']}")
        print(f"  Flagged (poison) : {stats['flagged_as_poisoned']}")
        print(f"  Rollbacks done   : {stats['rollbacks_performed']}")
        print(f"  Detection rate   : {stats['detection_rate_pct']:.1f}%")
        print(f"  Checkpoints      : {stats['checkpoints_stored']}\n")


# ─────────────────────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  SafeSplit Defense — Quick Test")
    print("=" * 55)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Simple server model (PyramidCNN server side)
    server_model = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(256, 10)
    ).to(device)
    
    optimizer = torch.optim.SGD(
        server_model.parameters(), lr=0.01)
    
    # Initialize SafeSplit
    defense = SafeSplitDefense(
        server_model=server_model,
        freq_threshold=2.0,
        rot_threshold=1.5,
        device=str(device)
    )
    
    print("--- Simulating Multi-Client Training ---\n")
    
    n_epochs = 3
    n_clients = 4  # Client 0,1,2 = benign, Client 3 = malicious
    
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}:")
        
        for client_id in range(n_clients):
            # Skip if flagged from previous epoch
            if defense.should_skip_client(client_id, epoch):
                print(f"  ⏭️  Skipping client {client_id} "
                      f"(flagged prev epoch)")
                continue
            
            # BEFORE training
            defense.before_client_training(
                epoch, client_id, server_model)
            
            # Simulate training
            fake_smashed = torch.randn(16, 64, 16, 16).to(device)
            fake_labels = torch.randint(0, 10, (16,)).to(device)
            
            optimizer.zero_grad()
            output = server_model(fake_smashed)
            loss = nn.CrossEntropyLoss()(output, fake_labels)
            
            # Malicious client 3 injects backdoor
            # (makes large parameter updates)
            if client_id == 3 and epoch >= 2:
                backdoor_loss = loss * 8.0  # Simulate aggressive update
                backdoor_loss.backward()
                # Extra perturbation to simulate backdoor
                with torch.no_grad():
                    for param in server_model.parameters():
                        param.data += torch.randn_like(param) * 0.5
            else:
                loss.backward()
            
            optimizer.step()
            
            # AFTER training
            is_poisoned = defense.after_client_training(
                epoch, client_id, server_model)
            
            status = "🚨 POISONED" if is_poisoned else "✅ Clean"
            print(f"  Client {client_id}: {status}")
        
        print()
    
    defense.print_stats()
    
    print("\n--- Testing All Cut Layers ---")
    
    # PyramidCNN cut layer 1, 2, 3 server models
    for cl, in_ch, name in [
        (1, 32, "Weak Device"),
        (2, 64, "Medium Device"),
        (3, 128, "Strong Device")
    ]:
        srv = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 10)
        ).to(device)
        
        d = SafeSplitDefense(srv, device=str(device))
        
        # One round test
        d.before_client_training(1, 0, srv)
        opt = torch.optim.SGD(srv.parameters(), lr=0.01)
        
        inp_shape = {1: (8,32,32,32), 2: (8,64,16,16), 3: (8,128,8,8)}
        fake = torch.randn(*inp_shape[cl]).to(device)
        lbl = torch.randint(0, 10, (8,)).to(device)
        
        opt.zero_grad()
        out = srv(fake)
        nn.CrossEntropyLoss()(out, lbl).backward()
        opt.step()
        
        poisoned = d.after_client_training(1, 0, srv)
        print(f"  {name} (cut={cl}): "
              f"result={'poisoned' if poisoned else 'benign'} ✓")
    
    print("\n" + "=" * 55)
    print("  SafeSplit Defense Test Complete! ✓")
    print("=" * 55)