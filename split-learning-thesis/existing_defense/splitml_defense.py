
import math
import numpy as np
import torch

try:
    import openfhe as ofhe
    OPENFHE_AVAILABLE = True
except ImportError:
    OPENFHE_AVAILABLE = False


class OpenFHEMultiKeyCKKS:

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
        self.kps = [cc.KeyGen()]
        for _ in range(1, self.K):
            self.kps.append(cc.MultipartyKeyGen(self.kps[-1].publicKey))
        self.pk = self.kps[-1].publicKey
        tag = self.pk.GetKeyTag()

        cc.EvalSumKeyGen(self.kps[0].secretKey)
        sum_keys = cc.GetEvalSumKeyMap(self.kps[0].secretKey.GetKeyTag())
        joined = sum_keys
        for kp in self.kps[1:]:
            part = cc.MultiEvalSumKeyGen(kp.secretKey, sum_keys, kp.publicKey.GetKeyTag())
            joined = cc.MultiAddEvalSumKeys(joined, part, kp.publicKey.GetKeyTag())
        cc.InsertEvalSumKey(joined)

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
        vals = []
        for ct in cts:
            lead = self.cc.MultipartyDecryptLead([ct], self.kps[0].secretKey)[0]
            mains = [self.cc.MultipartyDecryptMain([ct], kp.secretKey)[0] for kp in self.kps[1:]]
            pt = self.cc.MultipartyDecryptFusion([lead] + mains)
            pt.SetLength(min(self.slots, length - len(vals)))
            vals.extend(pt.GetRealPackedValue())
        return np.asarray(vals[:length], dtype=np.float64)


class OpenFHESingleKeyCKKS:

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
        return self.cc.KeyGen().secretKey


class SimulatedCKKS:

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


def shared_state_keys(top_module):
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


def low_confidence_mask(probs, lam=0.10):
    top2 = probs.topk(2, dim=1).values
    return (top2[:, 0] - top2[:, 1]) <= 2 * lam


class SplitMLConsensus:

    def __init__(self, **backend_kw):
        self.backend_kw = backend_kw

    @torch.no_grad()
    def query(self, requester, peers, x, mode="TL"):
        session = make_session_backend(**self.backend_kw)
        smashed = requester.top(x)
        shape = smashed.shape
        flat = smashed.reshape(-1).double().cpu().numpy()
        cts = session.encrypt(flat)                                        

        received = torch.from_numpy(session.decrypt(cts, len(flat))).float().reshape(shape).to(x.device)
        votes = [torch.softmax(requester.bottom(smashed), dim=1)]
        for peer in peers:
            votes.append(torch.softmax(peer.bottom(received), dim=1))
        probs = torch.stack(votes)                                         

        if mode == "TL":
            labels = probs.argmax(dim=2)
            counts = torch.zeros(probs.shape[1], probs.shape[2], device=x.device)
            counts.scatter_add_(1, labels.t(), torch.ones_like(labels.t(), dtype=counts.dtype))
            counts += 1e-3 * probs.sum(0)                                  
            return counts.argmax(dim=1)
        return probs.sum(0).argmax(dim=1)                                  

    @torch.no_grad()
    def peer_view(self, smashed):
        session = make_session_backend(**self.backend_kw)
        flat = smashed.reshape(-1).double().cpu().numpy()
        cts = session.encrypt(flat)
        try:
            view = session.decrypt(cts, len(flat), secret_key=session.outsider_key())
        except Exception:
            view = np.random.uniform(-1e6, 1e6, size=len(flat))            
        view = np.nan_to_num(view, nan=0.0, posinf=1e6, neginf=-1e6)
        return torch.from_numpy(view).float().reshape(smashed.shape).to(smashed.device)

    @torch.no_grad()
    def colluding_view(self, smashed):
        session = make_session_backend(**self.backend_kw)
        flat = smashed.reshape(-1).double().cpu().numpy()
        return torch.from_numpy(session.decrypt(session.encrypt(flat), len(flat))).float() \
                    .reshape(smashed.shape).to(smashed.device)


class SplitMLDefense:

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
