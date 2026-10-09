import math
import hashlib
from dataclasses import dataclass, field

import torch


def flatten_params(*modules):
    return torch.cat([p.detach().reshape(-1).float().cpu()
                      for m in modules for p in m.parameters()])


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def state_hash(*states):
    h = hashlib.sha256()
    for st in states:
        for k in sorted(st.keys()):
            h.update(k.encode())
            h.update(st[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


_DCT_CACHE = {}


def dct_matrix(n, device='cpu'):
    key = (n, str(device))
    if key not in _DCT_CACHE:
        idx = torch.arange(n, dtype=torch.float64, device=device)
        C = torch.cos(math.pi * (2 * idx[None, :] + 1) * idx[:, None] / (2 * n)) * math.sqrt(2.0 / n)
        C[0] /= math.sqrt(2.0)
        _DCT_CACHE[key] = C.float()
    return _DCT_CACHE[key]


def to_square(vec):
    n = int(math.ceil(math.sqrt(vec.numel())))
    out = torch.zeros(n * n, dtype=vec.dtype, device=vec.device)
    out[:vec.numel()] = vec
    return out.view(n, n)


def band_mask(n, band, device='cpu'):
    u = torch.arange(n, device=device)
    low = (u[:, None] + u[None, :]) < (n / 2.0)
    if band == 'low':
        return low
    if band == 'high':
        return ~low
    return torch.ones(n, n, dtype=torch.bool, device=device)


@dataclass
class ZorroEntry:
    turn: int
    client_state: dict
    server_state: dict
    update: torch.Tensor
    poisoned: bool
    commitment: str = ""
    features: torch.Tensor = field(default=None, repr=False)


class ZORRODefense:

    def __init__(self, k=3, beta=0.7, transform='dct', band='low', metric='l1',
                 rollback=True, commit=True, zkp=True, inspect='client', device='cpu'):
        assert transform in ('dct', 'none')
        assert band in ('low', 'all', 'high')
        assert metric in ('l1', 'l2', 'cosine')
        assert inspect in ('client', 'client+server')
        assert 0.0 < beta <= 1.0 and k >= 1
        self.k, self.beta = int(k), float(beta)
        self.transform, self.band, self.metric = transform, band, metric
        self.rollback, self.commit, self.zkp, self.inspect = rollback, commit, zkp, inspect
        self.device = device

        self.queue = []
        self.bm_index = None
        self._continue_override = None
        self.banned = set()
        self.log = []
        self.stats = {'turns': 0, 'poison_present': 0, 'poison_removed': 0,
                      'benign_new': 0, 'benign_new_not_selected': 0,
                      'proof_rejected': 0, 'integrity_failures': 0, 'skipped_banned': 0}


    def config(self):
        return dict(k=self.k, beta=self.beta, transform=self.transform, band=self.band,
                    metric=self.metric, rollback=self.rollback, commit=self.commit,
                    zkp=self.zkp, inspect=self.inspect)

    def inspected_vector(self, client, server):
        return flatten_params(client, server) if self.inspect == 'client+server' else flatten_params(client)


    def _features(self, update):
        x = update.to(self.device)
        if self.transform == 'none':
            return x
        M = to_square(x)
        C = dct_matrix(M.shape[0], self.device)
        D = C @ M @ C.T
        return D[band_mask(M.shape[0], self.band, self.device)]

    def _feat(self, entry):
        if entry.features is None:
            entry.features = self._features(entry.update).cpu()
        return entry.features


    def _raw_scores(self, entries):
        feats = [self._feat(e) for e in entries]
        if self.metric == 'l1':
            return [float(f.abs().sum()) for f in feats]
        if self.metric == 'l2':
            return [float(f.norm()) for f in feats]
        if len(feats) == 1:
            return [0.0]

        scores = []
        for i, f in enumerate(feats):
            others = torch.stack([g for j, g in enumerate(feats) if j != i]).mean(0)
            cos = torch.nn.functional.cosine_similarity(f[None], others[None]).item()
            scores.append(1.0 - cos)
        return scores


    def decide(self, entries):
        S = self._raw_scores(entries)
        adj = list(S)
        if len(entries) == self.k + 1:
            adj[0] = adj[0] / self.beta
            adj[-1] = adj[-1] * self.beta
            j = max(range(len(adj)), key=lambda i: adj[i])
        else:
            j = None
            return j, len(entries) - 1, S, adj
        kept = [i for i in range(len(entries)) if i != j]
        bm_kept = min(range(len(kept)), key=lambda t: adj[kept[t]])
        return j, bm_kept, S, adj


    def seed(self, entry):
        if self.commit:
            entry.commitment = state_hash(entry.client_state, entry.server_state)
        self.queue.append(entry)
        self.queue = self.queue[-self.k:]
        self.bm_index = len(self.queue) - 1

    def current_bm(self):
        if self._continue_override is not None:
            return self._continue_override
        return self.queue[self.bm_index]

    def is_banned(self, client_id):
        return client_id in self.banned


    def verify_integrity(self):
        if not self.commit:
            return True
        ok = all(state_hash(e.client_state, e.server_state) == e.commitment for e in self.queue)
        if not ok:
            self.stats['integrity_failures'] += 1
        return ok


    def process_turn(self, turn, client_id, new_entry, cheat=False):
        self.stats['turns'] += 1
        self.verify_integrity()
        if self.commit:
            new_entry.commitment = state_hash(new_entry.client_state, new_entry.server_state)

        entries = self.queue + [new_entry]
        j, bm_kept, S, adj = self.decide(entries)


        if cheat:
            others = list(range(len(entries) - 1))
            claim_j = max(others, key=lambda i: adj[i]) if j is not None else None
            claim_kept = [i for i in range(len(entries)) if i != claim_j]
            claim_bm_kept = claim_kept.index(len(entries) - 1)
        else:
            claim_j, claim_bm_kept = j, bm_kept


        honest = (claim_j == j) and (claim_bm_kept == bm_kept)
        accepted = True
        if self.zkp and not honest:
            accepted = False
            self.stats['proof_rejected'] += 1
            self.banned.add(client_id)

        if accepted:
            kept = [entries[i] for i in range(len(entries)) if i != claim_j]
            removed = entries[claim_j] if claim_j is not None else None
            self.queue = kept
            self.bm_index = claim_bm_kept
            self._continue_override = None if self.rollback else new_entry
        else:
            removed = new_entry


        if any(e.poisoned for e in entries) and (j is not None or not accepted):
            self.stats['poison_present'] += 1
            if removed is not None and removed.poisoned:
                self.stats['poison_removed'] += 1
        bm = self.current_bm()
        if not new_entry.poisoned:
            self.stats['benign_new'] += 1
            if accepted and bm is not new_entry:
                self.stats['benign_new_not_selected'] += 1

        rec = {'turn': turn, 'client': client_id, 'new_poisoned': new_entry.poisoned, 'cheat': cheat,
               'accepted': accepted, 'removed_turn': removed.turn if removed is not None else None,
               'removed_poisoned': removed.poisoned if removed is not None else None,
               'bm_turn': bm.turn, 'bm_poisoned': bm.poisoned,
               'scores': [round(s, 4) for s in S], 'adj_scores': [round(s, 4) for s in adj],
               'queue_turns': [e.turn for e in self.queue]}
        self.log.append(rec)
        return rec


    def summary(self):
        s = dict(self.stats)
        s['PRR'] = 100.0 * s['poison_removed'] / s['poison_present'] if s['poison_present'] else float('nan')
        s['BBR'] = 100.0 * s['benign_new_not_selected'] / s['benign_new'] if s['benign_new'] else float('nan')
        s.update({f'cfg_{k}': v for k, v in self.config().items()})
        return s


    def export_queue(self):
        return {'bm_index': self.bm_index,
                'entries': [{'turn': e.turn, 'client_state': e.client_state, 'server_state': e.server_state,
                             'update': e.update, 'poisoned': e.poisoned} for e in self.queue]}

    def __repr__(self):
        return f"ZORRODefense({self.config()}) [Sheybani et al., CCS 2025]"


class LegacyHighFreqZORRO:

    def __init__(self, high_freq_split=0.5, std_multiplier=3.0, min_history=5):
        assert 0.0 < high_freq_split < 1.0
        self.high_freq_split = high_freq_split
        self.std_multiplier = std_multiplier
        self.min_history = min_history
        self.history = []

    @staticmethod
    def _dct(x):
        n = x.shape[-1]
        v = torch.cat([x[..., 0::2], x[..., 1::2].flip(-1)], dim=-1)
        Vc = torch.fft.fft(v, dim=-1)
        k = -torch.arange(n, dtype=x.dtype, device=x.device) * math.pi / (2 * n)
        return 2 * (Vc.real * torch.cos(k) - Vc.imag * torch.sin(k))

    def analyze_update(self, update_vector):
        coeffs = self._dct(update_vector)
        split = int(coeffs.shape[-1] * (1 - self.high_freq_split))
        ratio = ((coeffs[split:] ** 2).sum() / ((coeffs ** 2).sum() + 1e-12)).item()
        flagged = False
        if len(self.history) >= self.min_history:
            m = sum(self.history) / len(self.history)
            sd = (sum((h - m) ** 2 for h in self.history) / len(self.history)) ** 0.5
            flagged = ratio > m + self.std_multiplier * sd
        if not flagged:
            self.history.append(ratio)
        return flagged, ratio