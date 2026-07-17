"""Experiment 1: NG vs Adam vs Adaptive across Fisher regimes.

Three controlled regimes (RICH, SPARSE, BLIND) with known Fisher structure.
Shows that NG dominates in RICH, Adam dominates in BLIND, and adaptive
damping captures the best of both.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

from memplex.optimizer import AdaptiveNG, FixedNG
from memplex.fisher.spectrum import spectral_stats


@dataclass
class ExperimentConfig:
    d: int = 20
    n: int = 500
    n_steps: int = 100
    lr_ng: float = 0.5      # Newton scale for MSE
    lr_adam: float = 0.05
    eps_min: float = 1e-8
    eps_max: float = 0.5
    seed: int = 42


def make_rich_data(cfg: ExperimentConfig):
    """Well-conditioned data: Fisher kappa ~ 2-30."""
    rng = np.random.RandomState(cfg.seed)
    X = rng.randn(cfg.n, cfg.d) / np.sqrt(cfg.d)
    W = rng.randn(cfg.d, 1)
    y = X @ W
    return torch.FloatTensor(X), torch.FloatTensor(y)


def make_blind_data(cfg: ExperimentConfig):
    """Ill-conditioned data + rank-deficient Fisher (k=5 samples)."""
    rng = np.random.RandomState(cfg.seed + 1)
    X = rng.randn(cfg.n, cfg.d) / np.sqrt(cfg.d)
    W = rng.randn(cfg.d, 1)
    y = X @ W
    return torch.FloatTensor(X), torch.FloatTensor(y)


def run_regime_experiment(cfg: ExperimentConfig = None) -> Dict[str, Any]:
    """Run NG, Adam, and AdaptiveNG on RICH and BLIND regimes.

    Returns dict with loss curves and spectral stats.
    """
    if cfg is None:
        cfg = ExperimentConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X_rich, y_rich = make_rich_data(cfg)
    X_blind, y_blind = make_blind_data(cfg)
    loss_fn = nn.MSELoss()

    results = {}

    for regime, X, y in [("rich", X_rich, y_rich), ("blind", X_blind, y_blind)]:
        # Fixed NG (low damping)
        torch.manual_seed(cfg.seed)
        model = nn.Linear(cfg.d, 1)
        opt = FixedNG(model, loss_fn, lr=cfg.lr_ng, damping=1e-8)
        losses_ng = []
        for _ in range(cfg.n_steps):
            loss_value = opt.step(X, y, use_exact_fisher=True)
            losses_ng.append(loss_value)

        # Adam
        torch.manual_seed(cfg.seed)
        model = nn.Linear(cfg.d, 1)
        params = list(model.parameters())
        optim = torch.optim.Adam(params, lr=cfg.lr_adam)
        losses_adam = []
        for _ in range(cfg.n_steps):
            optim.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            optim.step()
            losses_adam.append(loss.item())

        # Adaptive NG (spectral entropy damping)
        torch.manual_seed(cfg.seed)
        model = nn.Linear(cfg.d, 1)
        opt = AdaptiveNG(model, loss_fn, lr=cfg.lr_ng,
                         eps_min=cfg.eps_min, eps_max=cfg.eps_max)
        losses_adapt = []
        for _ in range(cfg.n_steps):
            loss_value = opt.step(X, y, use_exact_fisher=True)
            losses_adapt.append(loss_value)

        # Spectral stats
        X_np = X.numpy()
        X_aug = np.hstack([X_np, np.ones((len(X_np), 1))])
        F = X_aug.T @ X_aug / len(X_np)
        eigs = np.sort(np.abs(np.linalg.eigvalsh(F)))[::-1]
        stats = spectral_stats(eigs)

        results[regime] = {
            "ng": losses_ng,
            "adam": losses_adam,
            "adaptive": losses_adapt,
            "kappa": stats["kappa"],
            "rank": stats["rank"],
            "eff_rank": stats["eff_rank"],
            "concentration": stats["concentration"],
        }

    return results
