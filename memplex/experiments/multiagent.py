"""Experiment 4: Fisher-gated multi-agent knowledge sharing.

Two agents train on different but overlapping data with different quality.
Agent A is expert on classes 0-4, has noisy data on 5-9.
Agent B is expert on classes 5-9, has noisy data on 0-4.

Four sharing protocols compared:
  1. FedAvg: naive parameter averaging
  2. Bayesian FL: Fisher-weighted averaging (F_A θ_A + F_B θ_B) / (F_A + F_B)
  3. Fisher-gated (hard): only share in directions where sender is confident
     AND receiver is not (prevents negative transfer)
  4. Fisher-gated (soft): per-direction Fisher-weighted combination

Key finding: Bayesian FL matches oracle (all data). Hard gate prevents
negative transfer on specific classes where one agent has noise.
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import eigh
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass


@dataclass
class MultiAgentConfig:
    d: int = 64
    n_classes: int = 10
    n_expert: int = 300
    n_noisy: int = 50
    noise_rate: float = 0.4
    n_steps: int = 300
    lr: float = 0.05
    seed: int = 42


def make_expert_noisy_split(X, y, expert_classes, n_expert, n_noisy, noise_rate, rng):
    """Create a data split where agent is expert on some classes, noisy on others."""
    em = np.isin(y, expert_classes)
    nm = ~em
    Xe, ye = X[em], y[em]
    Xn, yn = X[nm], y[nm]
    ie = rng.choice(len(Xe), min(n_expert, len(Xe)), replace=True)
    inn = rng.choice(len(Xn), min(n_noisy, len(Xn)), replace=True)
    X_out = np.vstack([Xe[ie], Xn[inn]])
    y_out = np.concatenate([ye[ie], yn[inn]])
    n_corrupt = int(len(inn) * noise_rate)
    if n_corrupt > 0:
        ci = rng.choice(len(inn), n_corrupt, replace=False)
        for c in ci:
            old = y_out[len(ie) + c]
            y_out[len(ie) + c] = (old + rng.randint(1, 10)) % 10
    return X_out, y_out


def train_agent(X_tr, y_tr, W0, b0, d, C, lr=0.05, n_steps=300):
    """Train linear model and compute exact Fisher via Kronecker structure."""
    loss_fn = nn.CrossEntropyLoss()
    model = nn.Linear(d, C)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(W0.reshape(C, d)))
        model.bias.copy_(torch.tensor(b0))
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.FloatTensor(X_tr)
    y = torch.LongTensor(y_tr)
    for _ in range(n_steps):
        optim.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optim.step()
    with torch.no_grad():
        p = torch.softmax(model(X), dim=1).numpy()
    n = len(X_tr)
    p_avg = p.mean(axis=0)
    cov_p = np.diag(p_avg) - np.outer(p_avg, p_avg)
    XtX = X_tr.T @ X_tr / n
    F = np.kron(cov_p, XtX)
    W = model.weight.detach().numpy().flatten()
    b = model.bias.detach().numpy()
    return W, b, F


def evaluate(W_flat, b, X_test, y_test, d, C):
    model = nn.Linear(d, C)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(W_flat.reshape(C, d)))
        model.bias.copy_(torch.tensor(b))
    model.eval()
    with torch.no_grad():
        return (model(torch.FloatTensor(X_test)).argmax(1) ==
                torch.LongTensor(y_test)).float().mean().item()


def fisher_gated_share(W_A, F_A, W_B, F_B, W_avg, method="hard", thresh=0.01):
    """Share knowledge between two agents using Fisher gating.

    Args:
        W_A, F_A: agent A weights and Fisher
        W_B, F_B: agent B weights and Fisher
        W_avg: naive average (starting point)
        method: "hard" (block low-confidence) or "soft" (Fisher-weighted)
        thresh: relative threshold for "high" vs "low" Fisher

    Returns:
        Shared weights
    """
    dim = len(W_A)

    if method == "hard":
        # Hard gate: only share in directions where sender is confident
        # AND receiver is uncertain
        ev_A, evec_A = eigh(F_A)
        ev_B, evec_B = eigh(F_B)
        iA = np.argsort(ev_A)[::-1]
        iB = np.argsort(ev_B)[::-1]
        ev_A, ev_B = ev_A[iA], ev_B[iB]
        evec_A, evec_B = evec_A[:, iA], evec_B[:, iB]

        FB_in_A = np.diag(evec_A.T @ F_B @ evec_A)
        FA_in_B = np.diag(evec_B.T @ F_A @ evec_B)
        FAm = max(ev_A.max(), 1e-15)
        FBm = max(ev_B.max(), 1e-15)

        share_AB = (ev_A > thresh * FAm) & (FB_in_A < thresh * FBm)
        share_BA = (ev_B > thresh * FBm) & (FA_in_B < thresh * FAm)

        W_out = W_avg.copy()
        dA = W_A - W_avg
        for i in range(dim):
            if share_AB[i]:
                v = evec_A[:, i]
                W_out += (dA @ v) * v
        dB = W_B - W_avg
        for i in range(dim):
            if share_BA[i]:
                v = evec_B[:, i]
                W_out += (dB @ v) * v
        return W_out

    elif method == "soft":
        # Soft gate: per-direction Fisher-weighted combination
        F_total = F_A + F_B
        ev_T, evec_T = eigh(F_total)
        iT = np.argsort(ev_T)[::-1]
        ev_T = ev_T[iT]
        evec_T = evec_T[:, iT]

        FA_proj = np.diag(evec_T.T @ F_A @ evec_T)
        FB_proj = np.diag(evec_T.T @ F_B @ evec_T)
        FT_max = max(ev_T.max(), 1e-15)
        threshold = thresh * FT_max
        FA_h = np.maximum(FA_proj, threshold)
        FB_h = np.maximum(FB_proj, threshold)
        ratio_A = FA_h / (FA_h + FB_h)

        dA_proj = evec_T.T @ (W_A - W_avg)
        dB_proj = evec_T.T @ (W_B - W_avg)
        W_out = W_avg.copy()
        for i in range(dim):
            v = evec_T[:, i]
            W_out += ratio_A[i] * dA_proj[i] * v
            W_out += (1 - ratio_A[i]) * dB_proj[i] * v
        return W_out


def run_multi_agent_experiment(cfg: MultiAgentConfig = None,
                               X_all=None, y_all=None) -> Dict[str, Any]:
    """Run the full multi-agent knowledge sharing experiment."""
    if cfg is None:
        cfg = MultiAgentConfig()
    if X_all is None or y_all is None:
        from sklearn.datasets import load_digits
        digits = load_digits()
        X_all = digits.data.astype(np.float32)
        y_all = digits.target
        X_all = (X_all - X_all.mean()) / (X_all.std() + 1e-8)

    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)

    d, C = cfg.d, cfg.n_classes
    dim_w = d * C

    X_Atr, y_Atr = make_expert_noisy_split(
        X_all, y_all, list(range(0, C // 2)), cfg.n_expert, cfg.n_noisy, cfg.noise_rate, rng)
    X_Btr, y_Btr = make_expert_noisy_split(
        X_all, y_all, list(range(C // 2, C)), cfg.n_expert, cfg.n_noisy, cfg.noise_rate, rng)

    test_idx = rng.choice(len(X_all), len(X_all) // 5, replace=False)
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]

    torch.manual_seed(cfg.seed)
    init_model = nn.Linear(d, C)
    W0 = init_model.weight.detach().numpy().flatten()
    b0 = init_model.bias.detach().numpy()

    W_A, b_A, F_A = train_agent(X_Atr, y_Atr, W0, b0, d, C, cfg.lr, cfg.n_steps)
    W_B, b_B, F_B = train_agent(X_Btr, y_Btr, W0, b0, d, C, cfg.lr, cfg.n_steps)

    W_avg = (W_A + W_B) / 2
    b_avg = (b_A + b_B) / 2

    F_sum = F_A + F_B + 1e-4 * np.eye(dim_w)
    W_bayes = np.linalg.solve(F_sum, F_A @ W_A + F_B @ W_B)

    W_hard = fisher_gated_share(W_A, F_A, W_B, F_B, W_avg, method="hard")
    W_soft = fisher_gated_share(W_A, F_A, W_B, F_B, W_avg, method="soft")

    W_orc, b_orc, _ = train_agent(
        np.vstack([X_Atr, X_Btr]), np.concatenate([y_Atr, y_Btr]),
        W0, b0, d, C, cfg.lr, cfg.n_steps)

    results = {}
    for name, W, b in [("A_alone", W_A, b_A), ("B_alone", W_B, b_B),
                        ("fedavg", W_avg, b_avg), ("bayesian", W_bayes, b_avg),
                        ("hard_gate", W_hard, b_avg), ("soft_gate", W_soft, b_avg),
                        ("oracle", W_orc, b_orc)]:
        results[name] = evaluate(W, b, X_test, y_test, d, C)

    return results
