"""
ZORRO — Zero-Knowledge Robustness and Privacy for Split Learning
Sheybani, Pegoraro, Knauer, Rieger, Mollakuqe, Koushanfar, Sadeghi — CCS 2025 (arXiv:2509.09787)

Paper-faithful re-implementation of Algorithm 1 (poisoning detection) plus a simulated
version of the Algorithm 2 attestation, with EVERY component switchable for ablation.

  Component (paper section)                 Flag                Paper default
  ---------------------------------------   -----------------   -------------
  C1  frequency transform (2D DCT, Sec 4.3)  transform           'dct'
  C2  low-frequency band  (App. A, Eq. 4)    band                'low'
  C3  Taxicab / L1 poison score              metric              'l1'
  C4  beta scaling of oldest/newest          beta                0.7
  C5  queue of k checkpoints                 k                   3
  C6  prune argmax + continue from BM        rollback            True
  C7  hash commitments of forwarded models   commit              True
  C8  proof of correct execution (ZKP)       zkp                 True
  --  which partition is inspected           inspect             'client'

Adaptation notes (vanilla SL in this repo vs. U-shaped SL in the paper):
  * The paper inspects head+tail, both client-owned. In this repo's vanilla SL the client owns
    only the head; the "tail" (classifier) is on the server. inspect='client' is the faithful
    setting. inspect='client+server' is an explicit DEVIATION, kept only to test whether ZORRO
    fails here because the poisoned partition is simply not inspected.
  * One client turn = one training epoch (same protocol as run_safesplit.py).
  * C8 is SIMULATED. Real ZORRO uses a VOLE-based ZKP (emp-zk / Wolverine). Here the verifier
    re-executes the committed computation and compares it with the client's claim. This has the
    same accept/reject behaviour as a sound and complete proof, but no zero-knowledge and no
    cryptographic cost. Report it as a simulation.

The earlier repo version (high-frequency energy ratio + z-score, no queue) is kept as
LegacyHighFreqZORRO so earlier result tables remain reproducible.
"""
import math
import hashlib
from dataclasses import dataclass, field

import torch


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════
def flatten_params(*modules):
    """Concatenate trainable parameters (no BN buffers) of the given modules into one vector."""
    return torch.cat([p.detach().reshape(-1).float().cpu()
                      for m in modules for p in m.parameters()])


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def state_hash(*states):
    """SHA-256 commitment over one or more state dicts (deterministic key order)."""
    h = hashlib.sha256()
    for st in states:
        for k in sorted(st.keys()):
            h.update(k.encode())
            h.update(st[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


_DCT_CACHE = {}


def dct_matrix(n, device='cpu'):
    """Orthonormal DCT-II matrix C_N (Sec 4.7: D = C_N M C_N^T)."""
    key = (n, str(device))
    if key not in _DCT_CACHE:
        idx = torch.arange(n, dtype=torch.float64, device=device)
        C = torch.cos(math.pi * (2 * idx[None, :] + 1) * idx[:, None] / (2 * n)) * math.sqrt(2.0 / n)
        C[0] /= math.sqrt(2.0)
        _DCT_CACHE[key] = C.float()
    return _DCT_CACHE[key]


def to_square(vec):
    """Zero-pad a flat vector into a ceil(sqrt(n)) x ceil(sqrt(n)) matrix (Sec 4.7)."""
    n = int(math.ceil(math.sqrt(vec.numel())))
    out = torch.zeros(n * n, dtype=vec.dtype, device=vec.device)
    out[:vec.numel()] = vec
    return out.view(n, n)


def band_mask(n, band, device='cpu'):
    """App. A, Eq. 4: low band = {(u, v) : u + v < n/2}. 'high' is its complement."""
    u = torch.arange(n, device=device)
    low = (u[:, None] + u[None, :]) < (n / 2.0)
    if band == 'low':
        return low
    if band == 'high':
        return ~low
    return torch.ones(n, n, dtype=torch.bool, device=device)


# ═════════════════════════════════════════════════════════════════════════════
#  Queue entry
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class ZorroEntry:
    turn: int
    client_state: dict
    server_state: dict
    update: torch.Tensor          # flattened update of the inspected partition(s), CPU
    poisoned: bool                # ground truth, used ONLY for PRR/BBR bookkeeping
    commitment: str = ""
    features: torch.Tensor = field(default=None, repr=False)   # selected coefficients (cache)


# ═════════════════════════════════════════════════════════════════════════════
#  ZORRO (paper-faithful, ablatable)
# ═════════════════════════════════════════════════════════════════════════════
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

        self.queue = []                 # oldest first, at most k entries
        self.bm_index = None            # index into self.queue
        self._continue_override = None  # used only when rollback is ablated
        self.banned = set()             # client ids excluded after a failed proof
        self.log = []                   # one dict per turn
        self.stats = {'turns': 0, 'poison_present': 0, 'poison_removed': 0,
                      'benign_new': 0, 'benign_new_not_selected': 0,
                      'proof_rejected': 0, 'integrity_failures': 0, 'skipped_banned': 0}

    # ── configuration helpers ────────────────────────────────────────────
    def config(self):
        return dict(k=self.k, beta=self.beta, transform=self.transform, band=self.band,
                    metric=self.metric, rollback=self.rollback, commit=self.commit,
                    zkp=self.zkp, inspect=self.inspect)

    def inspected_vector(self, client, server):
        return flatten_params(client, server) if self.inspect == 'client+server' else flatten_params(client)

    # ── C1 + C2: frequency features ──────────────────────────────────────
    def _features(self, update):
        x = update.to(self.device)
        if self.transform == 'none':
            return x                                         # raw parameter-space update
        M = to_square(x)
        C = dct_matrix(M.shape[0], self.device)
        D = C @ M @ C.T                                      # 2D DCT-II
        return D[band_mask(M.shape[0], self.band, self.device)]

    def _feat(self, entry):
        if entry.features is None:
            entry.features = self._features(entry.update).cpu()
        return entry.features

    # ── C3: poison risk scores ───────────────────────────────────────────
    def _raw_scores(self, entries):
        feats = [self._feat(e) for e in entries]
        if self.metric == 'l1':
            return [float(f.abs().sum()) for f in feats]
        if self.metric == 'l2':
            return [float(f.norm()) for f in feats]
        if len(feats) == 1:
            return [0.0]
        # cosine: distance of each update's spectrum to the mean spectrum of the others
        scores = []
        for i, f in enumerate(feats):
            others = torch.stack([g for j, g in enumerate(feats) if j != i]).mean(0)
            cos = torch.nn.functional.cosine_similarity(f[None], others[None]).item()
            scores.append(1.0 - cos)
        return scores

    # ── Algorithm 1 (lines 6-19) ─────────────────────────────────────────
    def decide(self, entries):
        """entries = queue (oldest first) + [new].
        Returns (removed index j or None, BM index within the kept list, raw scores, adjusted scores)."""
        S = self._raw_scores(entries)
        adj = list(S)
        if len(entries) == self.k + 1:                      # full queue: apply C4 and prune
            adj[0] = adj[0] / self.beta                     # oldest: S[0] / beta
            adj[-1] = adj[-1] * self.beta                   # newest: S[k] * beta
            j = max(range(len(adj)), key=lambda i: adj[i])
        else:                                               # queue still filling: nothing pruned,
            j = None                                        # nothing to compare -> continue from newest
            return j, len(entries) - 1, S, adj
        kept = [i for i in range(len(entries)) if i != j]
        bm_kept = min(range(len(kept)), key=lambda t: adj[kept[t]])
        return j, bm_kept, S, adj

    # ── initialisation (Sec 4.4: pretrained model + benign turns) ─────────
    def seed(self, entry):
        """Insert a benign entry while the queue is being initialised."""
        if self.commit:
            entry.commitment = state_hash(entry.client_state, entry.server_state)
        self.queue.append(entry)
        self.queue = self.queue[-self.k:]
        self.bm_index = len(self.queue) - 1

    def current_bm(self):
        """Model the next turn continues from. With C6 ablated this is the newest trained model,
        even if the detector pruned it from the queue (detection without enforcement)."""
        if self._continue_override is not None:
            return self._continue_override
        return self.queue[self.bm_index]

    def is_banned(self, client_id):
        return client_id in self.banned

    # ── C7: integrity of the received checkpoints ────────────────────────
    def verify_integrity(self):
        if not self.commit:
            return True
        ok = all(state_hash(e.client_state, e.server_state) == e.commitment for e in self.queue)
        if not ok:
            self.stats['integrity_failures'] += 1
        return ok

    # ── one client turn: Steps 4-8 ───────────────────────────────────────
    def process_turn(self, turn, client_id, new_entry, cheat=False):
        """
        new_entry : ZorroEntry of the model the client just trained.
        cheat     : malicious client does NOT run Algorithm 1 honestly; it claims its own model
                    is the best and drops the riskiest *other* model instead.
        """
        self.stats['turns'] += 1
        self.verify_integrity()
        if self.commit:
            new_entry.commitment = state_hash(new_entry.client_state, new_entry.server_state)

        entries = self.queue + [new_entry]
        j, bm_kept, S, adj = self.decide(entries)           # what an honest client computes

        # what the client CLAIMS
        if cheat:
            others = list(range(len(entries) - 1))
            claim_j = max(others, key=lambda i: adj[i]) if j is not None else None
            claim_kept = [i for i in range(len(entries)) if i != claim_j]
            claim_bm_kept = claim_kept.index(len(entries) - 1)
        else:
            claim_j, claim_bm_kept = j, bm_kept

        # C8: verifier (next client + server) checks the claim against the committed computation
        honest = (claim_j == j) and (claim_bm_kept == bm_kept)
        accepted = True
        if self.zkp and not honest:
            accepted = False
            self.stats['proof_rejected'] += 1
            self.banned.add(client_id)                       # Sec 3.2: failing proof -> excluded

        if accepted:
            kept = [entries[i] for i in range(len(entries)) if i != claim_j]
            removed = entries[claim_j] if claim_j is not None else None
            self.queue = kept
            self.bm_index = claim_bm_kept
            self._continue_override = None if self.rollback else new_entry   # C6 ablated
        else:
            removed = new_entry                              # rejected turn: model discarded

        # ── bookkeeping (ground-truth labels only used here) ─────────────
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

    # ── metrics ──────────────────────────────────────────────────────────
    def summary(self):
        s = dict(self.stats)
        s['PRR'] = 100.0 * s['poison_removed'] / s['poison_present'] if s['poison_present'] else float('nan')
        s['BBR'] = 100.0 * s['benign_new_not_selected'] / s['benign_new'] if s['benign_new'] else float('nan')
        s.update({f'cfg_{k}': v for k, v in self.config().items()})
        return s

    # ── persistence (benign queue saved with the training checkpoint) ─────
    def export_queue(self):
        return {'bm_index': self.bm_index,
                'entries': [{'turn': e.turn, 'client_state': e.client_state, 'server_state': e.server_state,
                             'update': e.update, 'poisoned': e.poisoned} for e in self.queue]}

    def __repr__(self):
        return f"ZORRODefense({self.config()}) [Sheybani et al., CCS 2025]"


# ═════════════════════════════════════════════════════════════════════════════
#  Legacy implementation (kept for reproducibility of earlier tables)
# ═════════════════════════════════════════════════════════════════════════════
class LegacyHighFreqZORRO:
    """Earlier repo version: DCT HIGH-frequency energy ratio + mean+3std threshold, no queue/rollback.
    NOT the paper's algorithm (the paper uses the LOW band, an L1 norm, a top-k queue and rollback)."""

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
