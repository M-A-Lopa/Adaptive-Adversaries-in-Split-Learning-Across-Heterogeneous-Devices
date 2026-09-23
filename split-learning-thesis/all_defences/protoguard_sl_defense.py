import torch
import torch.nn.functional as F


class ProtoGuardSLDefense:
    
    def __init__(self, num_classes, alpha=0.5):
        assert 0.0 < alpha < 1.0, "alpha must be in (0, 1); paper default is 0.5"
        self.num_classes = num_classes
        self.alpha = alpha

        self.prototypes = {}          
        self.reference_patterns = {}  
        self.calibration_scores = {}  
        self._proto_classes = []
        self._proto_matrix = None
        self.fitted = False

    @staticmethod
    def _flatten(smashed_data):
        if smashed_data.dim() > 2:
            return smashed_data.mean(dim=list(range(2, smashed_data.dim())))
        return smashed_data

    def _consistency_vectors(self, flat_embeddings, proto_matrix):
        """Eq. 2: cosine similarity of each embedding to every class
        prototype -> shape [N, num_classes_seen]."""
        e_norm = F.normalize(flat_embeddings, dim=1)
        p_norm = F.normalize(proto_matrix, dim=1)
        return e_norm @ p_norm.T


    def fit(self, client_model, calibration_loader, device):

        client_model.eval()

        embeddings_by_class = {c: [] for c in range(self.num_classes)}
        with torch.no_grad():
            for images, labels in calibration_loader:
                images = images.to(device)
                labels = labels.to(device)
                flat = self._flatten(client_model(images)).cpu()
                for c in range(self.num_classes):
                    mask = (labels.cpu() == c)
                    if mask.any():
                        embeddings_by_class[c].append(flat[mask])

        self.prototypes = {}
        for c, chunks in embeddings_by_class.items():
            if len(chunks) == 0:
                continue
            E_c = torch.cat(chunks, dim=0)
            self.prototypes[c] = E_c.median(dim=0).values  

        self._proto_classes = sorted(self.prototypes)
        self._proto_matrix = torch.stack([self.prototypes[c] for c in self._proto_classes])

        self.reference_patterns = {}
        self.calibration_scores = {}
        for c, chunks in embeddings_by_class.items():
            if c not in self.prototypes or len(chunks) == 0:
                continue
            E_c = torch.cat(chunks, dim=0)
            v_c = self._consistency_vectors(E_c, self._proto_matrix)  
            mu_c = v_c.median(dim=0).values                           
            s_c = (v_c - mu_c).norm(dim=1)                            
            self.reference_patterns[c] = mu_c
            self.calibration_scores[c] = s_c.sort().values

        self.fitted = True
        return self


    def protect(self, smashed_data, labels=None):

        if not self.fitted or labels is None or self._proto_matrix is None:
            return smashed_data

        device = smashed_data.device
        flat = self._flatten(smashed_data).detach().cpu()
        v = self._consistency_vectors(flat, self._proto_matrix)

        protected = smashed_data.clone()
        labels_cpu = labels.detach().cpu()

        for c in self._proto_classes:
            mask = (labels_cpu == c)
            if not mask.any() or c not in self.reference_patterns:
                continue

            mu_c = self.reference_patterns[c]
            s = (v[mask] - mu_c).norm(dim=1)  

            calib = self.calibration_scores[c]
            less_than = torch.searchsorted(calib, s, right=False)
            ge_count = len(calib) - less_than
            p_vals = (ge_count.float() + 1.0) / (len(calib) + 1.0)  

            flagged = p_vals <= self.alpha
            if flagged.any():
                idx = mask.nonzero(as_tuple=True)[0][flagged]
                proto_c = self.prototypes[c].to(device=device, dtype=protected.dtype)
                if smashed_data.dim() > 2:
                    view_shape = (1, -1) + (1,) * (smashed_data.dim() - 2)
                    proto_c = proto_c.view(*view_shape).expand(
                        len(idx), -1, *smashed_data.shape[2:]
                    )
                protected[idx] = proto_c

        return protected

    def __repr__(self):
        status = "fitted" if self.fitted else "NOT fitted"
        return (f"ProtoGuardSLDefense(alpha={self.alpha}, num_classes={self.num_classes}, "
                f"{status})  [re-implemented from Shui et al. 2026, arXiv:2604.03595]")


if __name__ == "__main__":
    torch.manual_seed(0)
    num_classes = 5
    defense = ProtoGuardSLDefense(num_classes=num_classes, alpha=0.5)

    class DummyLoader:
        def __iter__(self):
            for _ in range(10):
                yield torch.randn(32, 16, 4, 4), torch.randint(0, num_classes, (32,))

    class DummyClient(torch.nn.Module):
        def forward(self, x):
            return x

    defense.fit(DummyClient(), DummyLoader(), device='cpu')
    print(defense)

    smashed = torch.randn(8, 16, 4, 4)
    labels = torch.randint(0, num_classes, (8,))
    out = defense.protect(smashed, labels)
    print("Output shape:", out.shape, "matches input:", out.shape == smashed.shape)