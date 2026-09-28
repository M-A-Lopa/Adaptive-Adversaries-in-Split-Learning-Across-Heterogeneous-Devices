"""
SplitML — Federated Split Learning with multi-key CKKS FHE aggregation and encrypted consensus inference.

Paper : D. Trivedi, A. Boudguiga, N. Kaaniche, N. Triandopoulos, "SplitML: A Unified Privacy-Preserving
        Architecture for Federated Split-Learning in Heterogeneous Environments",
        Electronics 15(2):267, Jan. 2026. doi:10.3390/electronics15020267

No official code was released (the paper's Data Availability statement lists none). The paper states it
uses the OpenFHE library with Python 3.11, so this file uses the real OpenFHE Python bindings
(`pip install openfhe`) and maps each algorithm of the paper onto the actual OpenFHE calls:

  Alg. 1  PK/SK chain              : KeyGen() -> MultipartyKeyGen(PK_{k-1})  ...  PK = PK_K
  Alg. 2  EK_Add (two-pass)        : EvalSumKeyGen / MultiEvalSumKeyGen / MultiAddEvalSumKeys
  Alg. 3  EK_Mult (four-pass)      : KeySwitchGen / MultiKeySwitchGen / MultiAddEvalKeys /
                                     MultiMultEvalKey / MultiAddEvalMultKeys
  Alg. 4  training round           : each client Enc_PK(shared-layer weights) -> server EvalAdd ->
                                     server EvalMult(V_Add, Enc_PK(1/K)) -> clients partial-decrypt ->
                                     server fused decryption (EK_Dec) -> clients set weights
  Alg. 5  consensus inference      : requester builds a fresh single-key context (PK', SK', EK'),
                                     sends Enc_PK'(cut-layer activations) to peers, decrypts their answers,
                                     decides by Total Labels (TL, majority) or Total Predictions (TP, soft vote)
  Sec 2.6 "DP"                     : no manual noise is injected by the paper; privacy relies on CKKS
                                     intrinsic error + OpenFHE decryption noise. `extra_dp_sigma` exposes the
                                     optional extra Gaussian noise the paper suggests in Sec. 3.2.1 (default 0).

If `openfhe` is not installed, a numerically-equivalent SimulatedCKKS backend is used (fixed-point
encoding with scaling factor 2^scale_bits plus fresh-encryption Gaussian error). The run script prints
which backend was used so results are never silently mixed.

Limitation (stated honestly): the paper's peers "perform homomorphic calculations on their bottom layers".
The paper only does that for tiny MLP heads. Evaluating conv/BN/ReLU bottoms homomorphically needs
polynomial activation approximations and is out of scope, so peers here decode through the CKKS
round-trip and evaluate in plaintext; the requester's activations are still encrypted in transit and the
CKKS error is real. Confidentiality claims in the run script therefore only use what a peer WITHOUT SK' sees.
"""
import math
import numpy as np
import torch

try:
    import openfhe as ofhe
    OPENFHE_AVAILABLE = True
except ImportError:
    OPENFHE_AVAILABLE = False


# ============================================================================ backends
class OpenFHEMultiKeyCKKS:
    """Real K-party CKKS (OpenFHE). One context shared by all training clients + the federation server."""

    name = "OpenFHE multi-key CKKS"

    def __init__(self, num_clients, batch_size=8192, scale_bits=50, mult_depth=2):
        p = ofhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(mult_depth)
        p.SetScalingModSize(scale_bits)
        p.SetBatchSize(batch_size)
        self.cc = ofhe.GenCryptoContext(p)
        for f in (ofhe.PKESchemeFeature.PKE, ofhe.PKESchemeFeature.KEYSWITCH,
                  ofhe.PKESchemeFeature.LEVELEDSHE, ofhe.PKESchemeFeature.ADVANCEDSHE,
                  ofhe.PKESchemeFeature.MULTIPARTY):
            self.cc.Enable(f)
        self.K = num_clients
        self.slots = batch_size
        self.ciphertexts_sent = 0
        self._keygen()

    def _keygen(self):
        cc = self.cc
        # ---- Alg. 1: sequential public/secret key chain
        self.kps = [cc.KeyGen()]
        for _ in range(1, self.K):
            self.kps.append(cc.MultipartyKeyGen(self.kps[-1].publicKey))
        self.pk = self.kps[-1].publicKey
        tag = self.pk.GetKeyTag()

        # ---- Alg. 2: evaluation key for addition (EvalSum keys, two passes)
        cc.EvalSumKeyGen(self.kps[0].secretKey)
        sum_keys = cc.GetEvalSumKeyMap(self.kps[0].secretKey.GetKeyTag())
        joined = sum_keys
        for kp in self.kps[1:]:
            part = cc.MultiEvalSumKeyGen(kp.secretKey, sum_keys, kp.publicKey.GetKeyTag())
            joined = cc.MultiAddEvalSumKeys(joined, part, kp.publicKey.GetKeyTag())
        cc.InsertEvalSumKey(joined)

        # ---- Alg. 3: evaluation key for multiplication (four passes)
        ek1 = cc.KeySwitchGen(self.kps[0].secretKey, self.kps[0].secretKey)
        ek_temp = ek1
        for kp in self.kps[1:]:
            ek_k = cc.MultiKeySwitchGen(kp.secretKey, kp.secretKey, ek1)
            ek_temp = cc.MultiAddEvalKeys(ek_temp, ek_k, kp.publicKey.GetKeyTag())
        parts = [cc.MultiMultEvalKey(kp.secretKey, ek_temp, tag) for kp in reversed(self.kps)]
        ek_mult = cc.MultiAddEvalMultKeys(parts[0], parts[1], ek_temp.GetKeyTag())
        for part in parts[2:]:
            ek_mult = cc.MultiAddEvalMultKeys(ek_mult, part, ek_temp.GetKeyTag())
        cc.InsertEvalMultKey([ek_mult])

    def encrypt(self, vec):
        cts = []
        for i in range(0, len(vec), self.slots):
            chunk = vec[i:i + self.slots].tolist()
            cts.append(self.cc.Encrypt(self.pk, self.cc.MakeCKKSPackedPlaintext(chunk)))
        self.ciphertexts_sent += len(cts)
        return cts

    def aggregate_mean(self, client_cts):
        """Alg. 4 lines 5-12: V_Add = sum_k Enc(w_k);  V_Add * Enc_PK(1/K)."""
        n_chunks = len(client_cts[0])
        v_mult = self.cc.Encrypt(self.pk, self.cc.MakeCKKSPackedPlaintext([1.0 / self.K] * self.slots))
        out = []
        for c in range(n_chunks):
            acc = client_cts[0][c]
            for k in range(1, self.K):
                acc = self.cc.EvalAdd(acc, client_cts[k][c])
            out.append(self.cc.EvalMult(acc, v_mult))
        return out

    def fused_decrypt(self, cts, length):
        """Alg. 4 lines 13-16: every client partially decrypts with SK_k, server fuses with EK_Dec."""
        vals = []
        for ct in cts:
            lead = self.cc.MultipartyDecryptLead([ct], self.kps[0].secretKey)[0]
            mains = [self.cc.MultipartyDecryptMain([ct], kp.secretKey)[0] for kp in self.kps[1:]]
            pt = self.cc.MultipartyDecryptFusion([lead] + mains)
            pt.SetLength(min(self.slots, length - len(vals)))
            vals.extend(pt.GetRealPackedValue())
        return np.asarray(vals[:length], dtype=np.float64)


class OpenFHESingleKeyCKKS:
    """Alg. 5: session keys (PK', SK', EK') generated by the requester for one consensus query."""

    name = "OpenFHE single-key CKKS"

    def __init__(self, batch_size=8192, scale_bits=50, mult_depth=2):
        p = ofhe.CCParamsCKKSRNS()
        p.SetMultiplicativeDepth(mult_depth)
        p.SetScalingModSize(scale_bits)
        p.SetBatchSize(batch_size)
        self.cc = ofhe.GenCryptoContext(p)
        for f in (ofhe.PKESchemeFeature.PKE, ofhe.PKESchemeFeature.KEYSWITCH, ofhe.PKESchemeFeature.LEVELEDSHE):
            self.cc.Enable(f)
        self.kp = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.kp.secretKey)          # EK'
        self.slots = batch_size

    def encrypt(self, vec):
        return [self.cc.Encrypt(self.kp.publicKey, self.cc.MakeCKKSPackedPlaintext(vec[i:i + self.slots].tolist()))
                for i in range(0, len(vec), self.slots)]

    def decrypt(self, cts, length, secret_key=None):
        sk = secret_key if secret_key is not None else self.kp.secretKey
        vals = []
        for ct in cts:
            pt = self.cc.Decrypt(ct, sk)
            pt.SetLength(min(self.slots, length - len(vals)))
            vals.extend(pt.GetRealPackedValue())
        return np.asarray(vals[:length], dtype=np.float64)

    def outsider_key(self):
        """A consensus peer's own key pair in the same parameter set (it never receives SK')."""
        return self.cc.KeyGen().secretKey


class SimulatedCKKS:
    """Fallback when openfhe is missing: CKKS-style fixed point (Delta = 2^scale_bits) + fresh-encryption error."""

    name = "SimulatedCKKS (openfhe not installed)"

    def __init__(self, num_clients=1, batch_size=8192, scale_bits=50, ring_dim=16384, **_):
        self.K, self.slots = num_clients, batch_size
        self.delta = 2.0 ** scale_bits
        self.err = 3.2 * math.sqrt(ring_dim) / self.delta
        self.ciphertexts_sent = 0

    def _noisy(self, v):
        return np.round(v * self.delta) / self.delta + np.random.normal(0, self.err, size=v.shape)

    def encrypt(self, vec):
        self.ciphertexts_sent += math.ceil(len(vec) / self.slots)
        return [self._noisy(np.asarray(vec, dtype=np.float64))]

    def aggregate_mean(self, client_cts):
        return [self._noisy(sum(c[0] for c in client_cts) / self.K)]

    def fused_decrypt(self, cts, length):
        return cts[0][:length]

    def decrypt(self, cts, length, secret_key=None):
        if secret_key == "outsider":
            return np.random.uniform(-1e6, 1e6, size=length)
        return self._noisy(cts[0][:length])

    def outsider_key(self):
        return "outsider"


def make_multikey_backend(num_clients, **kw):
    return OpenFHEMultiKeyCKKS(num_clients, **kw) if OPENFHE_AVAILABLE else SimulatedCKKS(num_clients, **kw)


def make_session_backend(**kw):
    return OpenFHESingleKeyCKKS(**kw) if OPENFHE_AVAILABLE else SimulatedCKKS(**kw)


# ============================================================================ federation (training)
def shared_state_keys(top_module):
    """Floating tensors of the shared (top) layers: weights, biases and BN running stats."""
    return [k for k, v in top_module.state_dict().items() if torch.is_floating_point(v)]


def flatten_state(top_module, keys):
    sd = top_module.state_dict()
    return torch.cat([sd[k].detach().reshape(-1).double().cpu() for k in keys]).numpy()


def load_flat_state(top_module, keys, flat):
    sd = top_module.state_dict()
    offset = 0
    for k in keys:
        n = sd[k].numel()
        sd[k].copy_(torch.from_numpy(flat[offset:offset + n]).reshape(sd[k].shape).to(sd[k].dtype))
        offset += n
    top_module.load_state_dict(sd)


class SplitMLFederation:
    """Server-side encrypted FedAvg of the n shared top layers (Alg. 4). The server only ever holds ciphertexts."""

    def __init__(self, num_clients, extra_dp_sigma=0.0, **backend_kw):
        self.backend = make_multikey_backend(num_clients, **backend_kw)
        self.K = num_clients
        self.extra_dp_sigma = extra_dp_sigma
        self.last_server_view = None

    def aggregate(self, top_modules):
        keys = shared_state_keys(top_modules[0])
        client_cts = []
        for m in top_modules:
            w = flatten_state(m, keys)
            if self.extra_dp_sigma > 0:
                w = w + np.random.normal(0, self.extra_dp_sigma, size=w.shape)
            client_cts.append(self.backend.encrypt(w))
        self.last_server_view = client_cts
        avg_ct = self.backend.aggregate_mean(client_cts)
        avg = self.backend.fused_decrypt(avg_ct, length=len(w))
        for m in top_modules:
            load_flat_state(m, keys, avg)
        return avg


# ============================================================================ consensus (inference)
def low_confidence_mask(probs, lam=0.10):
    """
    Paper (binary, Sigmoid): consensus for f(z) in [0.5-lam, 0.5+lam].
    Multiclass generalisation used here: top-1 minus top-2 probability <= 2*lam (identical for 2 classes).
    """
    top2 = probs.topk(2, dim=1).values
    return (top2[:, 0] - top2[:, 1]) <= 2 * lam


class SplitMLConsensus:
    """Alg. 5 — encrypted collaborative inference between a requester and its peers."""

    def __init__(self, **backend_kw):
        self.backend_kw = backend_kw

    @torch.no_grad()
    def query(self, requester, peers, x, mode="TL"):
        session = make_session_backend(**self.backend_kw)
        smashed = requester.top(x)
        shape = smashed.shape
        flat = smashed.reshape(-1).double().cpu().numpy()
        cts = session.encrypt(flat)                                        # Enc_PK'(grad_q D'_j)

        received = torch.from_numpy(session.decrypt(cts, len(flat))).float().reshape(shape).to(x.device)
        votes = [torch.softmax(requester.bottom(smashed), dim=1)]
        for peer in peers:
            votes.append(torch.softmax(peer.bottom(received), dim=1))
        probs = torch.stack(votes)                                         # [m', B, C]

        if mode == "TL":
            labels = probs.argmax(dim=2)
            counts = torch.zeros(probs.shape[1], probs.shape[2], device=x.device)
            counts.scatter_add_(1, labels.t(), torch.ones_like(labels.t(), dtype=counts.dtype))
            counts += 1e-3 * probs.sum(0)                                  # tie-break by soft score
            return counts.argmax(dim=1)
        return probs.sum(0).argmax(dim=1)                                  # TP: highest aggregate score

    @torch.no_grad()
    def peer_view(self, smashed):
        """What a curious peer WITHOUT SK' can recover: decryption of Enc_PK'(a) under its own key."""
        session = make_session_backend(**self.backend_kw)
        flat = smashed.reshape(-1).double().cpu().numpy()
        cts = session.encrypt(flat)
        try:
            view = session.decrypt(cts, len(flat), secret_key=session.outsider_key())
        except Exception:
            view = np.random.uniform(-1e6, 1e6, size=len(flat))            # OpenFHE refuses to decode garbage
        view = np.nan_to_num(view, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.from_numpy(view).float().reshape(smashed.shape).to(smashed.device)

    @torch.no_grad()
    def colluding_view(self, smashed):
        """Out-of-threat-model upper bound: a peer that somehow holds SK' sees activations + CKKS error."""
        session = make_session_backend(**self.backend_kw)
        flat = smashed.reshape(-1).double().cpu().numpy()
        return torch.from_numpy(session.decrypt(session.encrypt(flat), len(flat))).float() \
                    .reshape(smashed.shape).to(smashed.device)


class SplitMLDefense:
    """Attack-evaluation interface (same as DPSLDefense.protect): the smashed data a curious peer observes."""

    def __init__(self, collusion=False, **backend_kw):
        self.collusion = collusion
        self.consensus = SplitMLConsensus(**backend_kw)

    def protect(self, smashed):
        return self.consensus.colluding_view(smashed) if self.collusion else self.consensus.peer_view(smashed)

    def __repr__(self):
        backend = "OpenFHE CKKS" if OPENFHE_AVAILABLE else "SimulatedCKKS"
        who = "peer holding SK' (collusion bound)" if self.collusion else "curious peer without SK'"
        return f"SplitMLDefense(view={who}, backend={backend})  [Trivedi et al. 2026]"


if __name__ == "__main__":
    import torch.nn as nn
    torch.manual_seed(0)
    tops = [nn.Sequential(nn.Conv2d(1, 4, 3), nn.BatchNorm2d(4)) for _ in range(3)]
    for t in tops:
        with torch.no_grad():
            for p in t.parameters():
                p.add_(torch.randn_like(p))
    keys = shared_state_keys(tops[0])
    expected = np.mean([flatten_state(t, keys) for t in tops], axis=0)
    fed = SplitMLFederation(num_clients=3)
    got = fed.aggregate(tops)
    print(f"backend={fed.backend.name}  max|FedAvg_enc - FedAvg_plain| = {np.abs(got - expected).max():.3e}")
    x = torch.relu(torch.randn(2, 4, 3, 3))
    print("peer view (no SK'):", SplitMLDefense().protect(x).flatten()[:4])
    print("collusion view    :", SplitMLDefense(collusion=True).protect(x).flatten()[:4], x.flatten()[:4])
