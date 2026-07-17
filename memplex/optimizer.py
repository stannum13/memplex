"""Spectral-entropy-adaptive natural gradient optimizer.

The optimizer that uses the Fisher eigenspectrum to set Tikhonov damping
continuously, via the Renyi entropy of the eigenvalue distribution.

Usage:
    opt = AdaptiveNG(model, loss_fn, lr=0.5, eps_min=1e-8, eps_max=0.5)
    for step in range(n_steps):
        loss = opt.step(X, y)
        # opt.damping and opt.stats are updated each step

For MSE on linear models, lr=0.5 is exact Newton (because Hessian = 2*Fisher).
For CrossEntropy, lr=1.0 is the Newton scale (Fisher = Hessian).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from curvlinops import FisherMCLinearOperator

from memplex.fisher.spectrum import adaptive_damping


class AdaptiveNG:
    """Natural gradient with spectral-entropy-adaptive damping.

    Computes the Fisher information matrix via curvlinops, analyzes its
    eigenspectrum using Renyi entropy, and sets Tikhonov damping
    continuously based on spectral concentration and rank deficiency.

    The key insight: instead of switching between NG and Adam based on
    a hard threshold, we interpolate the damping from eps_min (trust the
    Fisher, aggressive Newton) to eps_max (distrust the Fisher, conservative
    regularized step). The interpolation is driven by the spectral entropy
    of the Fisher eigenspectrum.
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module,
                 lr: float = 0.5, eps_min: float = 1e-8,
                 eps_max: float = 0.5, fisher_freq: int = 1,
                 mc_samples: int = 1):
        """
        Args:
            model: torch model to optimize
            loss_fn: loss function (MSELoss, CrossEntropyLoss)
            lr: learning rate (0.5 for MSE, 1.0 for CrossEntropy)
            eps_min: minimum damping (RICH regime)
            eps_max: maximum damping (BLIND regime)
            fisher_freq: recompute Fisher every N steps
            mc_samples: MC samples for Fisher estimation (higher = more accurate)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.fisher_freq = fisher_freq
        self.mc_samples = mc_samples

        self.params = list(model.parameters())
        self.dim = sum(p.numel() for p in self.params)
        self.step_count = 0
        self.damping = eps_min
        self.stats: Dict = {}
        self.loss_history: List[float] = []
        self.damping_history: List[float] = []

    def _compute_fisher(self, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Compute Fisher matrix via curvlinops."""
        data = [(X, y)]
        fim = FisherMCLinearOperator(
            self.model, self.loss_fn, self.params, data,
            check_deterministic=False, mc_samples=self.mc_samples
        )
        mat = (fim @ torch.eye(fim.shape[1])).numpy()
        return (mat + mat.T) / 2  # symmetrize

    def _compute_fisher_exact(self, X: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Compute exact Fisher for MSE linear model (F = X_aug^T X_aug / n).

        Faster and more accurate than curvlinops for this special case.
        """
        X_np = X.detach().numpy()
        n = len(X_np)
        X_aug = np.hstack([X_np, np.ones((n, 1))])
        return X_aug.T @ X_aug / n

    def step(self, X: torch.Tensor, y: torch.Tensor,
             use_exact_fisher: bool = False) -> float:
        """Take one optimization step.

        Args:
            X, y: training data
            use_exact_fisher: if True, use analytic Fisher (MSE linear only)

        Returns:
            loss value before the step
        """
        self.model.zero_grad()
        loss = self.loss_fn(self.model(X), y)
        loss.backward()

        grad = torch.cat([p.grad.flatten() for p in self.params]).numpy()

        # Compute or reuse Fisher
        if self.step_count % self.fisher_freq == 0:
            if use_exact_fisher:
                self._F = self._compute_fisher_exact(X, y)
                eigvals = np.linalg.eigvalsh(self._F)
            else:
                self._F = self._compute_fisher(X, y)
                eigvals = np.linalg.eigvalsh(self._F)

            eigvals = np.sort(np.abs(eigvals))[::-1]
            self.damping, self.stats = adaptive_damping(
                eigvals, self.eps_min, self.eps_max
            )

        # Solve (F + eps*I) d = grad
        F_damped = self._F + self.damping * np.eye(self.dim)
        try:
            direction = np.linalg.solve(F_damped, grad)
        except np.linalg.LinAlgError:
            direction = grad  # fallback to gradient

        direction = torch.tensor(direction, dtype=torch.float32)

        with torch.no_grad():
            offset = 0
            for p in self.params:
                n = p.numel()
                p -= self.lr * direction[offset:offset + n].view_as(p)
                offset += n

        self.step_count += 1
        self.loss_history.append(loss.item())
        self.damping_history.append(self.damping)
        return loss.item()


class FixedNG(AdaptiveNG):
    """Natural gradient with fixed damping (no adaptation).

    Used as a baseline to show that adaptive damping matters.
    """

    def __init__(self, model, loss_fn, lr=0.5, damping=1e-8, **kwargs):
        super().__init__(model, loss_fn, lr=lr, eps_min=damping,
                         eps_max=damping, **kwargs)
