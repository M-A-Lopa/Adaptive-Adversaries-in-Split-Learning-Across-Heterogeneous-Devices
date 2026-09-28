"""
SplitFSS — Split Learning + Function Secret Sharing (private vanilla SL).

Paper : T. Khan, M. Budzys, A. Michalas, "Make Split, not Hijack: Preventing Feature-Space
        Hijacking Attacks in Split Learning", ACM SACMAT 2024 / arXiv:2404.09265.
Code  : https://github.com/UnoriginalOrigi/SplitFSS   (official, PySyft 0.2.x / AriaNN, Python 3.7, torch 1.4)

The official code cannot run next to PyTorch 2.x, so this file re-implements the exact protocol the
official code executes, in pure PyTorch:

  * ring            : Z_2^64 (torch.int64, dtype="long")               -> main.py: dtype = "long"
  * fixed precision : base 10, precision_fractional 5 (train) / 4 (test) -> main.py: precision_fractional = 5 if train else 4
  * parties         : client (plaintext conv layers), servers alice (P0) & bob (P1), crypto_provider (dealer)
  * client -> servers: additive secret shares of the activation map    -> procedure.py: output.encrypt(**kwargs)
  * FC layers       : Beaver-triple matrix multiplication              -> paper Sec 5.4 / 6.1
  * ReLU            : FSS comparison on the public masked input x_pub = x + alpha  (paper Sec 3, Alg. 1 line 13)
  * loss            : ((target - output)**2).sum() / batch_size, one-hot labels secret-shared
  * retry guard     : `while loss_dec.abs() > 15: RETRY`               -> procedure.py (truncation-failure guard)

FSS evaluation note: the comparison keys are evaluated through a dealer-held ideal DCF oracle.
Each server's view is exactly the view of the real AriaNN protocol — the opened x_pub (uniformly
masked) and a uniformly random output share — which is what matters for evaluating attacks by a
single corrupted server. Beaver triples, sharing, truncation and openings are real.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RING_LOW, RING_HIGH = -(2 ** 63), 2 ** 63 - 1
BYTES = 8


def rand_ring(shape):
    return torch.randint(RING_LOW, RING_HIGH, tuple(shape), dtype=torch.int64)


class FixedPoint:
    def __init__(self, base=10, precision_fractional=5):
        self.base = base
        self.prec = precision_fractional
        self.scale = base ** precision_fractional

    def encode(self, x):
        return torch.round(x.detach().double().cpu() * self.scale).to(torch.int64)

    def decode(self, v):
        return v.double() / self.scale


class CryptoProvider:
    """The `crypto_provider` VirtualWorker: generates correlated randomness (offline phase)."""

    def __init__(self):
        self.preprocessing_bytes = 0
        self._fss_masks = {}
        self._next_key = 0

    def share(self, secret):
        s0 = rand_ring(secret.shape)
        return s0, secret - s0

    def _count(self, *tensors):
        self.preprocessing_bytes += sum(2 * t.numel() * BYTES for t in tensors)

    def beaver_matmul(self, shape_a, shape_b):
        a, b = rand_ring(shape_a), rand_ring(shape_b)
        c = a @ b
        self._count(a, b, c)
        return self.share(a), self.share(b), self.share(c)

    def beaver_mul(self, shape):
        a, b = rand_ring(shape), rand_ring(shape)
        c = a * b
        self._count(a, b, c)
        return self.share(a), self.share(b), self.share(c)

    def fss_keygen_comparison(self, shape):
        """KeyGen(1^lambda, f_alpha) for f_alpha(x_pub) = [x_pub - alpha >= 0]. Returns alpha shares + key id."""
        alpha = rand_ring(shape)
        key_id = self._next_key
        self._next_key += 1
        self._fss_masks[key_id] = alpha
        self._count(alpha)
        return self.share(alpha), key_id

    def fss_eval_all(self, key_id, x_pub):
        """EvalAll(j, f_j, x_pub) for j in {0,1}: additive shares of [x >= 0] with x = x_pub - alpha."""
        alpha = self._fss_masks.pop(key_id)
        bit = ((x_pub - alpha) >= 0).to(torch.int64)
        return self.share(bit)


class TwoServerEngine:
    """Online phase run jointly by P0 (alice) and P1 (bob). Shares are always (share_P0, share_P1)."""

    def __init__(self, fp, dealer):
        self.fp = fp
        self.dealer = dealer
        self.online_bytes = 0

    def open(self, x0, x1):
        self.online_bytes += 2 * x0.numel() * BYTES
        return x0 + x1

    def truncate(self, z0, z1):
        s = self.fp.scale
        return torch.div(z0, s, rounding_mode='floor'), -torch.div(-z1, s, rounding_mode='floor')

    def matmul(self, X, Y, truncate=True):
        (a0, a1), (b0, b1), (c0, c1) = self.dealer.beaver_matmul(X[0].shape, Y[0].shape)
        E = self.open(X[0] - a0, X[1] - a1)
        F = self.open(Y[0] - b0, Y[1] - b1)
        z0 = E @ F + E @ b0 + a0 @ F + c0
        z1 = E @ b1 + a1 @ F + c1
        return self.truncate(z0, z1) if truncate else (z0, z1)

    def mul(self, X, Y, truncate=True):
        (a0, a1), (b0, b1), (c0, c1) = self.dealer.beaver_mul(X[0].shape)
        E = self.open(X[0] - a0, X[1] - a1)
        F = self.open(Y[0] - b0, Y[1] - b1)
        z0 = E * F + E * b0 + a0 * F + c0
        z1 = E * b1 + a1 * F + c1
        return self.truncate(z0, z1) if truncate else (z0, z1)

    def scalar(self, X, c):
        """Public float constant * shared value (lr, momentum, 2/B)."""
        k = int(round(c * self.fp.scale))
        return self.truncate(X[0] * k, X[1] * k)

    def relu_bit(self, X):
        """Paper Alg. 1 l.13 / Sec. 3: x_pub = x + alpha is opened, then FSS comparison keys are evaluated."""
        (al0, al1), key_id = self.dealer.fss_keygen_comparison(X[0].shape)
        x_pub = self.open(X[0] + al0, X[1] + al1)
        return self.dealer.fss_eval_all(key_id, x_pub), x_pub

    def relu(self, X):
        bit, _ = self.relu_bit(X)
        return self.mul(X, bit, truncate=False), bit


class SecureLinear:
    """nn.Linear whose weight/bias/momentum live only as secret shares on P0/P1."""

    def __init__(self, linear, fp, dealer):
        self.W = dealer.share(fp.encode(linear.weight.data.t().contiguous()))   # [in, out]
        self.b = dealer.share(fp.encode(linear.bias.data))
        self.vW = (torch.zeros_like(self.W[0]), torch.zeros_like(self.W[1]))
        self.vb = (torch.zeros_like(self.b[0]), torch.zeros_like(self.b[1]))


class SplitFSSServer:
    """
    Private server-side head of the paper (Network1 / SplitLeNet2):
        FC(in,100) -> ReLU -> FC(100,out) -> ReLU      (final ReLU kept as in the official code)
    Built from a plaintext nn.Sequential/Module that the model owner secret-shares at setup
    (procedure: modelPriv.encrypt(**kwargs)). Training = SGD with momentum, as in the paper.
    """

    def __init__(self, plain_linears, final_relu=True, precision_fractional=5, lr=0.002, momentum=0.9):
        self.fp = FixedPoint(10, precision_fractional)
        self.dealer = CryptoProvider()
        self.eng = TwoServerEngine(self.fp, self.dealer)
        self.layers = [SecureLinear(l, self.fp, self.dealer) for l in plain_linears]
        self.final_relu = final_relu
        self.lr, self.momentum = lr, momentum
        self.client_to_server_bytes = 0
        self.server_to_client_bytes = 0
        self._cache = None
        self.last_x_pub = []

    # ---------------------------------------------------------------- client-side helpers
    def client_share(self, smashed):
        """Client encrypts the activation map: additive shares sent to alice / bob."""
        flat = smashed.reshape(smashed.shape[0], -1)
        x0, x1 = self.dealer.share(self.fp.encode(flat))
        self.client_to_server_bytes += 2 * x0.numel() * BYTES
        return x0, x1

    def server_view(self, smashed, party=0):
        """What ONE corrupted server sees of the smashed data (decoded share). Same shape as smashed."""
        shares = self.client_share(smashed)
        return self.fp.decode(shares[party]).float().reshape(smashed.shape).to(smashed.device)

    # ---------------------------------------------------------------- secure forward / backward
    def forward(self, X):
        cache, h, self.last_x_pub = [], X, []
        for i, L in enumerate(self.layers):
            z = self.eng.matmul(h, L.W)
            z = (z[0] + L.b[0], z[1] + L.b[1])
            is_last = i == len(self.layers) - 1
            if not is_last or self.final_relu:
                bit, x_pub = self.eng.relu_bit(z)
                self.last_x_pub.append(x_pub)
                out = self.eng.mul(z, bit, truncate=False)
            else:
                bit, out = None, z
            cache.append((h, bit))
            h = out
        self._cache = cache
        return h

    def backward_and_step(self, dout):
        grads = []
        for L, (h_in, bit) in zip(reversed(self.layers), reversed(self._cache)):
            dz = self.eng.mul(dout, bit, truncate=False) if bit is not None else dout
            dW = self.eng.matmul((h_in[0].t().contiguous(), h_in[1].t().contiguous()), dz)
            db = (dz[0].sum(0), dz[1].sum(0))
            Wt = (L.W[0].t().contiguous(), L.W[1].t().contiguous())
            dout = self.eng.matmul(dz, Wt)
            grads.append((L, dW, db))
        for L, dW, db in grads:
            L.vW = self._add(self.eng.scalar(L.vW, self.momentum), dW)
            L.vb = self._add(self.eng.scalar(L.vb, self.momentum), db)
            L.W = self._sub(L.W, self.eng.scalar(L.vW, self.lr))
            L.b = self._sub(L.b, self.eng.scalar(L.vb, self.lr))
        self.server_to_client_bytes += 2 * dout[0].numel() * BYTES
        return dout

    @staticmethod
    def _add(a, b):
        return a[0] + b[0], a[1] + b[1]

    @staticmethod
    def _sub(a, b):
        return a[0] - b[0], a[1] - b[1]

    def mse_loss_grad(self, out, target_onehot):
        """loss = ((target-output)**2).sum()/B ;  dL/dout = 2(output-target)/B  (all on shares)."""
        B = target_onehot.shape[0]
        T = self.dealer.share(self.fp.encode(target_onehot))
        diff = (out[0] - T[0], out[1] - T[1])
        sq = self.eng.mul(diff, diff)
        loss = self.fp.decode(self.eng.open(sq[0].sum(), sq[1].sum())).item() / B
        return loss, self.eng.scalar(diff, 2.0 / B)

    def reveal(self, shares):
        return self.fp.decode(shares[0] + shares[1]).float()

    def export_plain_state(self):
        """Only for checkpoints/analysis: requires BOTH servers to cooperate (never happens in the protocol)."""
        return [(self.reveal(L.W).t().contiguous(), self.reveal(L.b)) for L in self.layers]

    def comm_report(self):
        mb = 1024 ** 2
        return {'client_to_server_MB': self.client_to_server_bytes / mb,
                'server_to_client_MB': self.server_to_client_bytes / mb,
                'server_online_MB': self.eng.online_bytes / mb,
                'preprocessing_MB': self.dealer.preprocessing_bytes / mb}


class SplitFSSDefense:
    """
    Attack-evaluation interface (same shape as DPSLDefense.protect):
      protect(smashed)  -> what a single corrupted server (default P0/alice) observes.
    """

    def __init__(self, party=0, precision_fractional=5):
        self.party = party
        self.fp = FixedPoint(10, precision_fractional)
        self.dealer = CryptoProvider()

    def protect(self, smashed):
        flat = smashed.reshape(smashed.shape[0], -1)
        shares = self.dealer.share(self.fp.encode(flat))
        return self.fp.decode(shares[self.party]).float().reshape(smashed.shape).to(smashed.device)

    def __repr__(self):
        return (f"SplitFSSDefense(corrupted_server={'alice/P0' if self.party == 0 else 'bob/P1'}, "
                f"ring=Z_2^64, fixed_point=10^{self.fp.prec})  [Khan et al. 2024]")


# ============================================================================ models (official repo models.py)
# Network2 / SplitLeNet1 = client, Network1 / SplitLeNet2 = FSS server head (flatten size inferred).


class SplitFSSMiniONNClient(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        return x


class SplitFSSLeNetClient(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        return x


def build_fss_server_head(in_features, num_classes=10, arch="minionn"):
    """Plaintext nn.Linear layers that the model owner secret-shares at setup (modelPriv.encrypt)."""
    if arch == "lenet":
        return [nn.Linear(in_features, 120), nn.Linear(120, 84), nn.Linear(84, num_classes)]
    return [nn.Linear(in_features, 100), nn.Linear(100, num_classes)]


def smashed_features(client, in_channels, img_size):
    with torch.no_grad():
        return client(torch.zeros(1, in_channels, img_size, img_size)).numel()


if __name__ == "__main__":
    torch.manual_seed(0)
    lin = [torch.nn.Linear(256, 100), torch.nn.Linear(100, 10)]
    srv = SplitFSSServer(lin)
    x = torch.randn(8, 256)
    X = srv.client_share(x)
    out = srv.reveal(srv.forward(X))
    ref = torch.relu(lin[1](torch.relu(lin[0](x))))
    print("max |secure - plaintext| :", (out - ref).abs().max().item())
    print("server view sample       :", SplitFSSDefense().protect(x)[0, :4])
