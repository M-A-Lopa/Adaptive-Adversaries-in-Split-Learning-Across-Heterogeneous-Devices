import copy
import random
import numpy as np
import torch
import torch.nn as nn


def angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


def sigmoid(x, shift=0, mult=1, exp=1):
    x_p = (x - shift) * mult
    return (1 / (1 + np.exp(-x_p))) ** exp


def sg_score(fakes, controls, regulars, shift=0, mult=1, exp=1, raw=False):
    f_mean = sum(fakes) / len(fakes)
    c_mean = sum(controls) / len(controls)
    r_mean = sum(regulars) / len(regulars)
    cr_mean = (c_mean + r_mean) / 2

    f_mean_mag = sum([np.linalg.norm(v) for v in fakes]) / len(fakes)
    c_mean_mag = sum([np.linalg.norm(v) for v in controls]) / len(controls)
    r_mean_mag = sum([np.linalg.norm(v) for v in regulars]) / len(regulars)
    cr_mean_mag = (c_mean_mag + r_mean_mag) / 2

    mag_div = (abs(f_mean_mag - cr_mean_mag) + abs(c_mean_mag - r_mean_mag))

    x = angle(f_mean, cr_mean) * (abs(f_mean_mag - cr_mean_mag) / mag_div) - angle(c_mean, r_mean) * (abs(r_mean_mag - c_mean_mag) / mag_div)

    if raw:
        return x
    else:
        return sigmoid(x, shift=shift, mult=mult, exp=exp)


SG_THRESHOLD = 0.9


def policy_fast(S, threshold=SG_THRESHOLD):
    return S[-1] < threshold


def policy_avg_k(S, k, threshold=SG_THRESHOLD):
    return np.mean(S[-k:]) < threshold


def policy_voting(S, c=10, n=5, threshold=SG_THRESHOLD):
    votes = 0
    for i in range(c):
        group = S[i * n:(i + 1) * n]
        if np.mean(group) < threshold:
            votes += 1
    return votes > c / 2


PAPER_POLICIES = {
    'fast'  : (lambda S: policy_fast(S),         1),
    'avg-10': (lambda S: policy_avg_k(S, 10),    10),
    'avg-20': (lambda S: policy_avg_k(S, 20),    20),
    'avg-50': (lambda S: policy_avg_k(S, 50),    50),
    'voting': (lambda S: policy_voting(S, 10, 5), 50),
}


def expected_fake_accuracy(A, B_F, L):
    return A * (1 - B_F) + B_F * (1 - A) / L


AF_TOLERANCE = 0.01


class OutputAccuracyEstimator:

    name = 'output'

    def __init__(self):
        self.correct, self.total = 0, 0

    def prepare(self, *args, **kwargs):
        pass

    def update(self, index, smashed, outputs, labels, is_fake):
        if is_fake:
            return
        self.correct += outputs.detach().argmax(1).eq(labels).sum().item()
        self.total   += labels.size(0)

    def accuracy(self, index):
        return self.correct / self.total if self.total else 0.0


class LocalModelAccuracyEstimator:

    name = 'local'

    def __init__(self, lr=0.001):
        self.lr = lr
        self.curve = []

    def prepare(self, client_model, server_model, train_loader, device):
        client = copy.deepcopy(client_model).to(device).train()
        server = copy.deepcopy(server_model).to(device).train()
        opt = torch.optim.Adam(list(client.parameters()) + list(server.parameters()), lr=self.lr, amsgrad=True)
        criterion = nn.CrossEntropyLoss()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            outputs = server(client(images))
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            correct += outputs.detach().argmax(1).eq(labels).sum().item()
            total   += labels.size(0)
            self.curve.append(correct / total)
        del client, server

    def update(self, index, smashed, outputs, labels, is_fake):
        pass

    def accuracy(self, index):
        if not self.curve:
            return 0.0
        return self.curve[min(index, len(self.curve) - 1)]


class LinearClassifierAccuracyEstimator:

    name = 'linear'

    def __init__(self, num_classes=10, lr=0.001):
        self.num_classes = num_classes
        self.lr = lr
        self.clf, self.opt = None, None
        self.criterion = nn.CrossEntropyLoss()
        self.correct, self.total = 0, 0

    def prepare(self, *args, **kwargs):
        pass

    def update(self, index, smashed, outputs, labels, is_fake):
        if is_fake:
            return
        x = smashed.detach().flatten(1)
        if self.clf is None:
            self.clf = nn.Linear(x.shape[1], self.num_classes).to(x.device)
            self.opt = torch.optim.Adam(self.clf.parameters(), lr=self.lr)
        logits = self.clf(x)
        self.correct += logits.detach().argmax(1).eq(labels).sum().item()
        self.total   += labels.size(0)
        self.opt.zero_grad()
        self.criterion(logits, labels).backward()
        self.opt.step()

    def accuracy(self, index):
        return self.correct / self.total if self.total else 0.0


A_ESTIMATORS = {
    'output': OutputAccuracyEstimator,
    'local' : LocalModelAccuracyEstimator,
    'linear': LinearClassifierAccuracyEstimator,
}


class SplitGuardDefense:

    def __init__(self, b_fake=64, p_fake=0.1, N=20, mult=5, exp=2,
                 num_classes=10, af_tolerance=AF_TOLERANCE, increase_n=True):
        self.b_fake = b_fake
        self.p_fake = p_fake
        self.N      = N
        self.N_initial  = N
        self.increase_n = increase_n
        self.mult   = mult
        self.exp    = exp
        self.num_classes  = num_classes
        self.af_tolerance = af_tolerance
        self.reset()

    def reset(self):
        self.fakes, self.r_1, self.r_2 = [], [], []
        self.fake_indices, self.r_1_indices, self.r_2_indices = [], [], []
        self.scores = []
        self.N = self.N_initial
        self.n_history = []
        self.bf_history = []
        self.detections = {name: None for name in PAPER_POLICIES}
        self.decision_log = []

    def should_send_fake(self, index):
        return index > self.N and random.random() <= self.p_fake

    def make_fake_labels(self, labels):
        rand_labels = (labels + random.randint(1, 8)) % 10
        return torch.cat((rand_labels[:self.b_fake], labels[self.b_fake:]))

    def record(self, index, send_fakes, client_grad, A=None, batch_size=None):
        client_grad = client_grad.detach().cpu()
        if send_fakes:
            self.fakes.append(client_grad)
            self.fake_indices.append(index)
            F, R1, R2 = self._active_sets()
            if len(F) > 0 and len(R1) > 0 and len(R2) > 0:
                sg = sg_score(F, R1, R2, mult=self.mult, exp=self.exp, raw=False)
                self.scores.append(sg)
                self._apply_policies(index, A, batch_size)
                return sg
        else:
            if index > self.N_initial:
                if random.random() <= 0.5:
                    self.r_1.append(client_grad)
                    self.r_1_indices.append(index)
                else:
                    self.r_2.append(client_grad)
                    self.r_2_indices.append(index)
        return None

    def _active_sets(self):
        keep = lambda grads, idxs: [g for g, i in zip(grads, idxs) if i > self.N]
        return (keep(self.fakes, self.fake_indices),
                keep(self.r_1, self.r_1_indices),
                keep(self.r_2, self.r_2_indices))

    def make_decision(self, scores_are_high, A, B_F):
        if scores_are_high:
            return 'keep'
        if A is None:
            return 'attack'
        A_F = expected_fake_accuracy(A, B_F, self.num_classes)
        if round(abs(A - A_F), 9) <= self.af_tolerance:
            if B_F >= 1:
                return 'wait'
            return 'increase_bf'
        return 'attack'

    def _apply_policies(self, index, A, batch_size):
        B_F = min(self.b_fake / batch_size, 1.0) if batch_size else 1.0
        increase_bf = False
        for name, (fn, min_scores) in PAPER_POLICIES.items():
            if self.detections[name] is not None or len(self.scores) < min_scores:
                continue
            decision = self.make_decision(not fn(self.scores), A, B_F)
            self.decision_log.append((index, name, decision, A,
                                      None if A is None else expected_fake_accuracy(A, B_F, self.num_classes)))
            if decision == 'attack':
                self.detections[name] = index
            elif decision == 'increase_bf':
                increase_bf = True
        if increase_bf and batch_size:
            self.b_fake = min(self.b_fake * 2, batch_size)
            self.bf_history.append((index, self.b_fake))
            if self.increase_n:
                self.N = self.N * 2
                self.n_history.append((index, self.N))

    def attack_detected(self, policy):
        return self.detections[policy] is not None

    def mean_score(self):
        return float(np.mean(self.scores)) if self.scores else float('nan')

    def __repr__(self):
        return (f"SplitGuardDefense(b_fake={self.b_fake}, p_fake={self.p_fake}, "
                f"N={self.N}, mult={self.mult}, exp={self.exp}, "
                f"af_tolerance={self.af_tolerance}, increase_n={self.increase_n})  [official code: Erdogan et al. 2022]")


if __name__ == "__main__":
    torch.manual_seed(0); random.seed(0)
    base = torch.randn(200)
    regs = [base + 0.1 * torch.randn(200) for _ in range(20)]
    r1, r2 = regs[:10], regs[10:]

    honest_fakes = [-2 * base + 0.1 * torch.randn(200) for _ in range(5)]
    hijack_fakes = [base + 0.1 * torch.randn(200) for _ in range(5)]

    print("honest-like score :", sg_score(honest_fakes, r1, r2, mult=5, exp=2))
    print("hijack-like score :", sg_score(hijack_fakes, r1, r2, mult=5, exp=2))

    honest_S = [0.99] * 60
    attack_S = [0.3, 0.95, 0.5, 0.2] * 15
    for name, (fn, _) in PAPER_POLICIES.items():
        print(f"{name:<7} honest -> {fn(honest_S)!s:<5}  attack -> {fn(attack_S)}")
    print("A_F (A=0.98, B_F=4/64, L=10):", round(expected_fake_accuracy(0.98, 4/64, 10), 4),
          "(paper: ~91.8%)")

    sg = SplitGuardDefense()
    for A, B_F, high in [(0.98, 1.0, True), (0.98, 1.0, False), (0.10, 1.0, False),
                         (0.10, 4/64, False), (0.98, 4/64, False)]:
        print(f"Algorithm 3 | high={high!s:<5} A={A:.2f} B_F={B_F:.3f} "
              f"A_F={expected_fake_accuracy(A, B_F, 10):.4f} -> {sg.make_decision(high, A, B_F)}")

    random.seed(1); torch.manual_seed(1)
    d = SplitGuardDefense(b_fake=4, N=20, increase_n=True)
    g = lambda: torch.randn(50)
    for i in range(21, 200):
        fake = (i % 10 == 0)
        d.record(i, fake, g(), A=0.10, batch_size=64)
    print(f"Increase-N test | b_fake history: {d.bf_history}")
    print(f"Increase-N test | N history     : {d.n_history}")
    F, R1, R2 = d._active_sets()
    print(f"Increase-N test | stored F/R1/R2: {len(d.fakes)}/{len(d.r_1)}/{len(d.r_2)} "
          f"-> active after N={d.N}: {len(F)}/{len(R1)}/{len(R2)}")
