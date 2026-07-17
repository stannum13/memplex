"""Fisher regime classification.

Three regimes based on Fisher eigenspectrum:
  FISHER_RICH:   well-conditioned (kappa < threshold)
  FISHER_SPARSE: ill-conditioned (kappa > threshold)  
  FISHER_BLIND:  low rank OR external shift signal

Design insight (from M1 probing):
  - Eigenvalue flux is TOO SLOW for shift detection (sliding buffer 
    means Fisher takes O(buffer_size) steps to reflect a shift)
  - Performance-based detectors (CUSUM) catch shifts in 1-2 episodes
  - Therefore: shift detection should be EXTERNAL to the Fisher module
  - The Fisher module classifies RICH vs SPARSE when the data is sufficient
  - BLIND = "don't trust the Fisher yet" (insufficient data OR just shifted)

Usage:
    detector = RegimeDetector(dim=100, config=cfg)
    
    # Normal update:
    report = detector.update(eigenvalues, eigenvectors, n_samples=N)
    
    # After CUSUM fires:
    detector.flag_shift()
    # Next update will be BLIND regardless of Fisher structure
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum


class Regime(Enum):
    FISHER_RICH = "rich"
    FISHER_SPARSE = "sparse"
    FISHER_BLIND = "blind"


@dataclass
class RegimeReport:
    regime: Regime
    effective_rank: int
    condition_number: float
    reason: str


@dataclass
class RegimeConfig:
    kappa_thresh: float = 1e3
    min_rank: int = 1
    noise_floor: float = 1e-6
    blind_after_shift_steps: int = 50  # Stay BLIND for N updates after shift


class RegimeDetector:
    """Classifies Fisher eigenspectrum into operational regimes.
    
    RICH vs SPARSE is determined by the eigenspectrum (kappa).
    BLIND is triggered by: (1) low rank, or (2) external shift signal.
    """

    def __init__(self, dim: int, config: RegimeConfig = None):
        self.dim = dim
        self.config = config or RegimeConfig()
        self.n_updates = 0
        self.shift_flag = False
        self.steps_since_shift = 0

    def flag_shift(self):
        """Signal that a distribution shift was detected externally."""
        self.shift_flag = True
        self.steps_since_shift = 0

    def update(self, fisher_input, eigenvectors: np.ndarray = None,
               actual_performance: float = None,
               n_samples: int = 0) -> RegimeReport:
        """Update with Fisher information.
        
        Args:
            fisher_input: Either (dim,) eigenvalues sorted descending,
                          or (dim, dim) full Fisher matrix.
            eigenvectors: (dim, dim) -- kept for API compat, not used
            actual_performance: kept for API compat, not used
            n_samples: gradient sample count (used for rank check)
        """
        fisher_input = np.asarray(fisher_input)
        
        if fisher_input.ndim == 1:
            eigenvalues = fisher_input
        elif fisher_input.ndim == 2:
            eigenvalues = np.linalg.eigvalsh(fisher_input)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0)  # PSD correction
        else:
            raise ValueError(f"Expected 1D or 2D input, got {fisher_input.ndim}D")

        self.n_updates += 1

        if self.shift_flag:
            self.steps_since_shift += 1
            if self.steps_since_shift > self.config.blind_after_shift_steps:
                self.shift_flag = False

        report = self.classify(eigenvalues)

        return report

    def classify(self, eigenvalues: np.ndarray) -> RegimeReport:
        cfg = self.config

        r = int(np.sum(eigenvalues > cfg.noise_floor * np.sqrt(self.dim)))

        if r >= cfg.min_rank and float(eigenvalues[r - 1]) > 0:
            kappa = eigenvalues[0] / eigenvalues[r - 1]
        else:
            kappa = float("inf")

        # BLIND: insufficient data OR external shift signal
        if r < cfg.min_rank:
            return RegimeReport(
                regime=Regime.FISHER_BLIND,
                effective_rank=r,
                condition_number=kappa,
                reason=f"rank r={r} < {cfg.min_rank}",
            )

        if self.shift_flag:
            return RegimeReport(
                regime=Regime.FISHER_BLIND,
                effective_rank=r,
                condition_number=kappa,
                reason=f"external shift signal ({self.steps_since_shift}/{cfg.blind_after_shift_steps})",
            )

        # RICH vs SPARSE: based on Fisher condition number
        if kappa > cfg.kappa_thresh:
            return RegimeReport(
                regime=Regime.FISHER_SPARSE,
                effective_rank=r,
                condition_number=kappa,
                reason=f"kappa={kappa:.0f} > {cfg.kappa_thresh:.0f}, r={r}/{self.dim}",
            )
        else:
            return RegimeReport(
                regime=Regime.FISHER_RICH,
                effective_rank=r,
                condition_number=kappa,
                reason=f"kappa={kappa:.0f}, r={r}/{self.dim}",
            )

    def update_from_full_matrix(self, G_full: np.ndarray,
                                actual_performance: float = None,
                                n_samples: int = 0) -> RegimeReport:
        eigvals, eigvecs = np.linalg.eigh(G_full)
        idx = np.argsort(eigvals)[::-1]
        return self.update(eigvals[idx], eigvecs[:, idx],
                           actual_performance, n_samples)
