
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from typing import Tuple, Optional, List


# ─────────────────────────────────────────────────────────────
# Step 1: Dimensionality Transformation
# ─────────────────────────────────────────────────────────────

class DimensionalityTransformer:
    
    def __init__(self, 
                 method: str = 'umap_style',
                 target_dim: int = 64,
                 random_state: int = 42):
        
        self.method = method
        self.target_dim = target_dim
        self.random_state = random_state
        self.is_fitted = False
        
        # PCA for all methods (base)
        self.pca = PCA(
            n_components=min(target_dim, 50),
            random_state=random_state
        )
        self.scaler = StandardScaler()
        
        # PKT-style kernel matrix
        self._kernel_matrix = None
        
        print(f"[SecureSplit] Transformer: method={method}, "
              f"target_dim={target_dim}")
    
    def fit(self, embeddings: np.ndarray):
        
        # Standardize first
        emb_scaled = self.scaler.fit_transform(embeddings)
        
        if self.method == 'pca':
            # Simple PCA reduction
            n_comp = min(self.target_dim, 
                        min(embeddings.shape))
            self.pca = PCA(n_components=n_comp,
                          random_state=self.random_state)
            self.pca.fit(emb_scaled)
            
        elif self.method == 'umap_style':
           
            n_comp = min(self.target_dim, min(embeddings.shape))
            self.pca = PCA(n_components=n_comp,
                          random_state=self.random_state)
            self.pca.fit(emb_scaled)
            
        elif self.method == 'pkt_style':
        
            np.random.seed(self.random_state)
            input_dim = embeddings.shape[1]
            
            # Random projection matrix for RFF
            self._omega = np.random.randn(
                input_dim, self.target_dim // 2) / np.sqrt(input_dim)
            self._bias = np.random.uniform(
                0, 2 * np.pi, self.target_dim // 2)
        
        self.is_fitted = True
        print(f"  [Transformer] Fitted on {embeddings.shape[0]} samples")
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        
        if not self.is_fitted:
            raise RuntimeError("Call fit() first!")
        
        emb_scaled = self.scaler.transform(embeddings)
        
        if self.method == 'pca':
            return self.pca.transform(emb_scaled)
        
        elif self.method == 'umap_style':
            # PCA → nonlinear activation
            pca_out = self.pca.transform(emb_scaled)
            # Tanh nonlinearity — separates clusters better
            transformed = np.tanh(pca_out)
            return transformed
        
        elif self.method == 'pkt_style':
            # Random Fourier Features (kernel expansion)
            proj = emb_scaled @ self._omega + self._bias
            # cos and sin features
            rff = np.concatenate([
                np.cos(proj), np.sin(proj)
            ], axis=1)
            return rff * np.sqrt(2.0 / self.target_dim)
        
        return emb_scaled
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit + transform একসাথে।"""
        self.fit(embeddings)
        return self.transform(embeddings)


# ─────────────────────────────────────────────────────────────
# Step 2: Adaptive Majority-Based Voting Filter
# ─────────────────────────────────────────────────────────────

class MajorityVotingFilter:
    
    
    def __init__(self,
                 n_clusters: int = 2,
                 voting_rounds: int = 3,
                 rejection_threshold: float = 0.1):
        
        self.n_clusters = n_clusters
        self.voting_rounds = voting_rounds
        self.rejection_threshold = rejection_threshold
        
        print(f"[SecureSplit] MajorityVotingFilter: "
              f"clusters={n_clusters}, rounds={voting_rounds}")
    
    def _kmeans_cluster(self, 
                        embeddings: np.ndarray,
                        n_clusters: int,
                        n_iter: int = 100) -> np.ndarray:
        """
        Simple K-means clustering (sklearn ছাড়া)।
        
        Returns: cluster assignments (N,)
        """
        N, D = embeddings.shape
        n_clusters = min(n_clusters, N)
        
        if n_clusters <= 1:
            return np.zeros(N, dtype=int)
        
        # Random initialization
        np.random.seed(42)
        idx = np.random.choice(N, n_clusters, replace=False)
        centers = embeddings[idx].copy()
        
        assignments = np.zeros(N, dtype=int)
        
        for _ in range(n_iter):
            # Assign to nearest center
            dists = np.array([
                np.linalg.norm(embeddings - c, axis=1) 
                for c in centers
            ]).T  # (N, K)
            
            new_assignments = np.argmin(dists, axis=1)
            
            if np.all(new_assignments == assignments):
                break
            
            assignments = new_assignments
            
            # Update centers
            for k in range(n_clusters):
                mask = (assignments == k)
                if mask.sum() > 0:
                    centers[k] = embeddings[mask].mean(axis=0)
        
        return assignments
    
    def filter_class(self, 
                     embeddings: np.ndarray) -> np.ndarray:
        
        N = len(embeddings)
        rejection_votes = np.zeros(N)
        
        for round_idx in range(self.voting_rounds):
            # K-means clustering
            assignments = self._kmeans_cluster(
                embeddings, self.n_clusters)
            
            # Count cluster sizes
            cluster_sizes = np.bincount(
                assignments, minlength=self.n_clusters)
            
            # Majority cluster = largest cluster
            majority_cluster = np.argmax(cluster_sizes)
            
            # Minority clusters = potentially poisoned
            for k in range(self.n_clusters):
                if k != majority_cluster:
                    minority_mask = (assignments == k)
                    minority_ratio = minority_mask.sum() / N
                    
                    # Only vote to reject if minority is small enough
                    if minority_ratio <= 0.5 - self.rejection_threshold:
                        rejection_votes[minority_mask] += 1
        
        # Reject if majority of rounds voted to reject
        rejection_mask = rejection_votes > (self.voting_rounds / 2)
        
        return rejection_mask
    
    def filter(self,
               transformed_emb: np.ndarray,
               labels: np.ndarray) -> np.ndarray:
        
        N = len(labels)
        rejection_mask = np.zeros(N, dtype=bool)
        
        unique_classes = np.unique(labels)
        
        for c in unique_classes:
            class_mask = (labels == c)
            class_embs = transformed_emb[class_mask]
            
            if len(class_embs) < self.n_clusters * 2:
                # Too few samples — skip filtering
                continue
            
            # Get rejection decisions for this class
            class_rejection = self.filter_class(class_embs)
            
            # Map back to original indices
            class_indices = np.where(class_mask)[0]
            rejection_mask[class_indices] = class_rejection
        
        return rejection_mask


# ─────────────────────────────────────────────────────────────
# Main SecureSplit Defense
# ─────────────────────────────────────────────────────────────

class SecureSplitDefense:
    
    
    def __init__(self,
                 embedding_dim: int,
                 transform_method: str = 'umap_style',
                 target_dim: int = 64,
                 n_clusters: int = 2,
                 voting_rounds: int = 3,
                 rejection_threshold: float = 0.1,
                 warmup_batches: int = 30,
                 cut_layer: int = 2,
                 device: str = 'cpu'):
        
        self.embedding_dim = embedding_dim
        self.cut_layer = cut_layer
        self.warmup_batches = warmup_batches
        self.device = device
        
        self._batch_count = 0
        self._is_fitted = False
        self._fit_buffer_emb = []
        self._fit_buffer_lbl = []
        self._total_rejected = 0
        self._total_processed = 0
        
        # Step 1: Transformer
        self.transformer = DimensionalityTransformer(
            method=transform_method,
            target_dim=target_dim
        )
        
        # Step 2: Filter
        self.voting_filter = MajorityVotingFilter(
            n_clusters=n_clusters,
            voting_rounds=voting_rounds,
            rejection_threshold=rejection_threshold
        )
        
        print(f"\n{'='*55}")
        print(f"  SecureSplit Defense Initialized")
        print(f"  Embedding dim      : {embedding_dim}")
        print(f"  Transform method   : {transform_method}")
        print(f"  Target dim         : {target_dim}")
        print(f"  Cut layer          : {cut_layer}")
        print(f"  Warmup batches     : {warmup_batches}")
        print(f"  N clusters         : {n_clusters}")
        print(f"  Voting rounds      : {voting_rounds}")
        print(f"{'='*55}\n")
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Tensor → numpy (flatten if needed)."""
        if tensor.dim() > 2:
            tensor = tensor.view(tensor.shape[0], -1)
        return tensor.detach().cpu().numpy()
    
    def _fit_transformer(self):
        """Collected buffer দিয়ে transformer fit করে।"""
        if not self._fit_buffer_emb:
            return
        
        all_embs = np.concatenate(self._fit_buffer_emb, axis=0)
        self.transformer.fit(all_embs)
        self._is_fitted = True
        
        print(f"  ✅ [SecureSplit] Transformer fitted on "
              f"{len(all_embs)} samples. Defense active!")
    
    def filter(self,
               embeddings: torch.Tensor,
               labels: torch.Tensor) -> Tuple[torch.Tensor,
                                              torch.Tensor,
                                              torch.Tensor]:
        
        self._batch_count += 1
        emb_np = self._to_numpy(embeddings)
        lbl_np = labels.detach().cpu().numpy()
        
        # ── WARMUP PHASE ─────────────────────────────────────
        if not self._is_fitted:
            # Collect data for fitting
            self._fit_buffer_emb.append(emb_np)
            self._fit_buffer_lbl.append(lbl_np)
            
            if self._batch_count >= self.warmup_batches:
                self._fit_transformer()
            
            # During warmup: accept all
            identity_mask = torch.zeros(
                len(labels), dtype=torch.bool)
            return embeddings, labels, identity_mask
        
        # ── ACTIVE DEFENSE PHASE ─────────────────────────────
        # Step 1: Transform embeddings
        try:
            transformed = self.transformer.transform(emb_np)
        except Exception as e:
            print(f"  ⚠️  [SecureSplit] Transform failed: {e}")
            identity_mask = torch.zeros(len(labels), dtype=torch.bool)
            return embeddings, labels, identity_mask
        
        # Step 2: Majority voting filter
        rejection_np = self.voting_filter.filter(transformed, lbl_np)
        rejection_mask = torch.tensor(rejection_np, dtype=torch.bool)
        
        # Keep non-rejected samples
        keep_mask = ~rejection_mask
        
        if keep_mask.sum() == 0:
            # Safety: if all rejected, keep all
            keep_mask = torch.ones(len(labels), dtype=torch.bool)
            rejection_mask = torch.zeros(len(labels), dtype=torch.bool)
        
        clean_embeddings = embeddings[keep_mask]
        clean_labels = labels[keep_mask]
        
        # Track statistics
        n_rejected = rejection_mask.sum().item()
        self._total_rejected += n_rejected
        self._total_processed += len(labels)
        
        if n_rejected > 0:
            rate = n_rejected / len(labels) * 100
            if rate > 15:
                print(f"  ⚠️  [SecureSplit] High rejection: "
                      f"{n_rejected}/{len(labels)} ({rate:.1f}%) "
                      f"— Backdoor attack suspected!")
        
        return clean_embeddings, clean_labels, rejection_mask
    
    def is_active(self) -> bool:
        """Defense active কিনা।"""
        return self._is_fitted
    
    def get_stats(self) -> dict:
        """Defense statistics।"""
        rejection_rate = (self._total_rejected / 
                         max(1, self._total_processed) * 100)
        return {
            'total_processed': self._total_processed,
            'total_rejected': self._total_rejected,
            'rejection_rate_pct': rejection_rate,
            'defense_active': self._is_fitted,
            'cut_layer': self.cut_layer,
            'batches_processed': self._batch_count
        }
    
    def print_stats(self):
        stats = self.get_stats()
        print(f"\n[SecureSplit Statistics]")
        print(f"  Total processed  : {stats['total_processed']}")
        print(f"  Total rejected   : {stats['total_rejected']}")
        print(f"  Rejection rate   : {stats['rejection_rate_pct']:.2f}%")
        print(f"  Defense active   : {stats['defense_active']}")
        print(f"  Cut layer        : {stats['cut_layer']}")
        print(f"  Batches seen     : {stats['batches_processed']}\n")


# ─────────────────────────────────────────────────────────────
# Multi-Cut-Layer SecureSplit
# ─────────────────────────────────────────────────────────────

class SecureSplitManager:
    
    
    def __init__(self,
                 cut_layers: list = [1, 2, 3],
                 transform_method: str = 'umap_style',
                 warmup_batches: int = 30,
                 device: str = 'cpu'):
        
        # Embedding dims for PyramidCNN
        emb_dims = {
            1: 32 * 32 * 32,   # 32768
            2: 64 * 16 * 16,   # 16384
            3: 128 * 8 * 8     # 8192
        }
        
        self.defenses = {}
        
        print("\n[SecureSplit Manager] Initializing...")
        
        for cl in cut_layers:
            dim = emb_dims.get(cl, 16384)
            # Target dim scales with cut layer
            target_dim = max(32, 64 // cl)
            
            self.defenses[cl] = SecureSplitDefense(
                embedding_dim=dim,
                transform_method=transform_method,
                target_dim=target_dim,
                warmup_batches=warmup_batches,
                cut_layer=cl,
                device=device
            )
    
    def get_defense(self, cut_layer: int) -> SecureSplitDefense:
        """Get defense for specific cut layer."""
        if cut_layer not in self.defenses:
            raise ValueError(f"No defense for cut_layer={cut_layer}")
        return self.defenses[cut_layer]


# ─────────────────────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  SecureSplit Defense — Quick Test")
    print("=" * 55)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    num_classes = 10
    batch_size = 32
    cut_layer = 2
    embedding_dim = 64 * 16 * 16  # PyramidCNN cut layer 2
    
    # Initialize defense
    defense = SecureSplitDefense(
        embedding_dim=embedding_dim,
        transform_method='umap_style',
        target_dim=64,
        n_clusters=2,
        voting_rounds=3,
        rejection_threshold=0.1,
        warmup_batches=5,
        cut_layer=cut_layer,
        device=str(device)
    )
    
    print("--- Phase 1: Warmup ---")
    for i in range(6):
        fake_emb = torch.randn(batch_size, 64, 16, 16)
        fake_labels = torch.randint(0, num_classes, (batch_size,))
        
        clean_emb, clean_lbl, rejected = defense.filter(
            fake_emb, fake_labels)
        
        print(f"  Batch {i+1}: active={defense.is_active()}, "
              f"accepted={len(clean_lbl)}/{batch_size}, "
              f"rejected={rejected.sum().item()}")
    
    print("\n--- Phase 2: Normal vs Backdoor Detection ---")
    
    # Normal batch
    normal_emb = torch.randn(batch_size, 64, 16, 16)
    normal_labels = torch.randint(0, num_classes, (batch_size,))
    clean_emb, clean_lbl, rejected = defense.filter(
        normal_emb, normal_labels)
    print(f"  Normal batch  → rejected: {rejected.sum().item()}/{batch_size}")
    
    # Simulated VILLAIN backdoor
    # Attacker adds trigger pattern to some embeddings
    villain_emb = torch.randn(batch_size, 64, 16, 16)
    villain_emb[:6] = villain_emb[:6] + 3.5  # Trigger injection
    villain_labels = torch.randint(0, num_classes, (batch_size,))
    
    clean_emb, clean_lbl, rejected = defense.filter(
        villain_emb, villain_labels)
    print(f"  Backdoor batch→ rejected: {rejected.sum().item()}/{batch_size} "
          f"(6 poisoned injected)")
    
    print("\n--- Phase 3: All Cut Layers ---")
    manager = SecureSplitManager(
        cut_layers=[1, 2, 3],
        transform_method='umap_style',
        warmup_batches=3,
        device=str(device)
    )
    
    test_cases = [
        (1, (batch_size, 32, 32, 32), "Weak Device"),
        (2, (batch_size, 64, 16, 16), "Medium Device"),
        (3, (batch_size, 128, 8, 8),  "Strong Device"),
    ]
    
    for cl, shape, name in test_cases:
        d = manager.get_defense(cl)
        emb = torch.randn(*shape)
        lbl = torch.randint(0, 10, (batch_size,))
        
        # Warmup
        for _ in range(4):
            d.filter(emb, lbl)
        
        # Test with backdoor
        emb_poisoned = emb.clone()
        emb_poisoned[:5] += 4.0
        _, _, rej = d.filter(emb_poisoned, lbl)
        print(f"  {name} (cut={cl}): shape={shape} "
              f"→ rejected={rej.sum().item()}/{batch_size} ✓")
    
    defense.print_stats()
    
    print("=" * 55)
    print("  SecureSplit Defense Test Complete! ✓")
    print("=" * 55)