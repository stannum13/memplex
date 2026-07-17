"""Experiment 2: RICH -> BLIND regime transition with adaptive switching.

A problem that naturally transitions from Fisher-RICH (full-rank, well-conditioned)
to Fisher-BLIND (rank-deficient) during training, with a task shift at step 100.

Shows that:
  - Fixed NG diverges in BLIND (F^{-1} amplifies noise from rank-deficient Fisher)
  - Fixed Adam is steady but slow
  - Spectral-entropy-adaptive NG (continuous damping) outperforms both

This is the main result: continuous damping via spectral entropy beats
both fixed strategies and hard switching.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

from memplex.fisher.spectrum import adaptive_damping


@dataclass
class TransitionConfig:
    d: int = 20
    n: int = 500
    n_steps: int = 200
    shift_step: int = 100
    fisher_samples_blind: int = 5  # k samples for Fisher estimation in BLIND phase
    lr_ng: float = 0.5
    lr_adam: float = 0.05
    eps_min: float = 1e-8
    eps_max: float = 0.5
    seed: int = 42


def run_transition_experiment(cfg: TransitionConfig = None) -> Dict[str, Any]:
    """Run the RICH->BLIND transition experiment.

    Phase 1 (steps 0 to shift_step-1): RICH data, task A, exact Fisher
    Phase 2 (steps shift_step to n_steps-1): same data distribution, task B,
        Fisher estimated from k=fisher_samples_blind samples (rank-deficient)

    Returns dict with loss curves, cumulative loss, and damping trajectory.
    """
    if cfg is None:
        cfg = TransitionConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Same data distribution throughout, different tasks
    X_np = np.random.randn(cfg.n, cfg.d) / np.sqrt(cfg.d)
    W_A = np.random.randn(cfg.d, 1)
    W_B = np.random.randn(cfg.d, 1)

    X = torch.FloatTensor(X_np)
    y1 = torch.FloatTensor(X_np @ W_A)
    y2 = torch.FloatTensor(X_np @ W_B)
    loss_fn = nn.MSELoss()

    # Precompute exact Fisher (RICH) and rank-deficient Fisher (BLIND)
    X_aug = np.hstack([X_np, np.ones((cfg.n, 1))])
    F_rich = X_aug.T @ X_aug / cfg.n

    # For BLIND: re-estimate Fisher from k samples each step (noisy, rank-deficient)
    def get_blind_fisher():
        idx = np.random.choice(cfg.n, cfg.fisher_samples_blind, replace=False)
        X_sub = X_np[idx]
        X_aug_sub = np.hstack([X_sub, np.ones((cfg.fisher_samples_blind, 1))])
        return X_aug_sub.T @ X_aug_sub / cfg.fisher_samples_blind

    def get_data(step):
        if step < cfg.shift_step:
            return X, y1, F_rich
        else:
            return X, y2, get_blind_fisher()

    results = {}

    # 1. Fixed NG (low damping — diverges in BLIND)
    torch.manual_seed(cfg.seed)
    model = nn.Linear(cfg.d, 1)
    params = list(model.parameters())
    losses = []
    for step in range(cfg.n_steps):
        X_c, y_c, F_c = get_data(step)
        model.zero_grad()
        loss = loss_fn(model(X_c), y_c)
        loss.backward()
        grad = torch.cat([p.grad.flatten() for p in params]).numpy()
        F_d = F_c + 1e-8 * np.eye(len(grad))
        try:
            direction = np.linalg.solve(F_d, grad)
            with torch.no_grad():
                offset = 0
                for p in params:
                    n = p.numel()
                    p -= cfg.lr_ng * torch.tensor(direction[offset:offset+n],
                                                 dtype=torch.float32).view_as(p)
                    offset += n
        except np.linalg.LinAlgError:
            pass
        losses.append(loss.item())
    results["ng"] = losses

    # 2. Fixed Adam
    torch.manual_seed(cfg.seed)
    model = nn.Linear(cfg.d, 1)
    params = list(model.parameters())
    optim = torch.optim.Adam(params, lr=cfg.lr_adam)
    losses = []
    for step in range(cfg.n_steps):
        X_c, y_c, _ = get_data(step)
        if step == cfg.shift_step:
            optim = torch.optim.Adam(params, lr=cfg.lr_adam)  # reset
        optim.zero_grad()
        loss = loss_fn(model(X_c), y_c)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    results["adam"] = losses

    # 3. Hard switching (NG when rank >= d, Adam when rank < d)
    torch.manual_seed(cfg.seed)
    model = nn.Linear(cfg.d, 1)
    params = list(model.parameters())
    optim = torch.optim.Adam(params, lr=cfg.lr_adam)
    losses = []
    prev_strat = None
    for step in range(cfg.n_steps):
        X_c, y_c, F_c = get_data(step)
        rank = np.linalg.matrix_rank(F_c)
        strat = "ng" if rank >= cfg.d + 1 else "adam"  # +1 for bias term
        if strat == "adam" and prev_strat != "adam":
            optim = torch.optim.Adam(params, lr=cfg.lr_adam)
        prev_strat = strat
        if strat == "ng":
            model.zero_grad()
            loss = loss_fn(model(X_c), y_c)
            loss.backward()
            grad = torch.cat([p.grad.flatten() for p in params]).numpy()
            F_d = F_c + 1e-8 * np.eye(len(grad))
            direction = np.linalg.solve(F_d, grad)
            with torch.no_grad():
                offset = 0
                for p in params:
                    n = p.numel()
                    p -= cfg.lr_ng * torch.tensor(direction[offset:offset+n],
                                                 dtype=torch.float32).view_as(p)
                    offset += n
            losses.append(loss.item())
        else:
            optim.zero_grad()
            loss = loss_fn(model(X_c), y_c)
            loss.backward()
            optim.step()
            losses.append(loss.item())
    results["hard_adapt"] = losses

    # 4. Spectral-entropy-adaptive NG (continuous damping)
    torch.manual_seed(cfg.seed)
    model = nn.Linear(cfg.d, 1)
    params = list(model.parameters())
    losses = []
    damping_traj = []
    for step in range(cfg.n_steps):
        X_c, y_c, F_c = get_data(step)
        eigs = np.sort(np.abs(np.linalg.eigvalsh(F_c)))[::-1]
        damping, stats = adaptive_damping(eigs, cfg.eps_min, cfg.eps_max)
        damping_traj.append(damping)

        model.zero_grad()
        loss = loss_fn(model(X_c), y_c)
        loss.backward()
        grad = torch.cat([p.grad.flatten() for p in params]).numpy()
        F_d = F_c + damping * np.eye(len(grad))
        try:
            direction = np.linalg.solve(F_d, grad)
        except np.linalg.LinAlgError:
            direction = grad
        with torch.no_grad():
            offset = 0
            for p in params:
                n = p.numel()
                p -= cfg.lr_ng * torch.tensor(direction[offset:offset+n],
                                             dtype=torch.float32).view_as(p)
                offset += n
        losses.append(loss.item())
    results["adaptive"] = losses
    results["damping_traj"] = damping_traj

    # Cumulative loss
    for name in ["ng", "adam", "hard_adapt", "adaptive"]:
        arr = np.array(results[name], dtype=float)
        arr[np.isinf(arr)] = 1e10
        arr[np.isnan(arr)] = 1e10
        results[f"cum_{name}"] = float(np.cumsum(arr)[-1])

    return results
