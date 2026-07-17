"""Spectral entropy and adaptive damping for Fisher information matrices.

The core contribution: instead of hard regime switching (RICH/SPARSE/BLIND),
we compute the Renyi entropy H(alpha) of the Fisher eigenspectrum and use it
to set the Tikhonov damping continuously.

H(alpha) = (1/(1-alpha)) * log(sum(p_i^alpha))

where p_i = lambda_i / sum(lambda_j) is the normalized eigenspectrum.

Key statistics:
  exp(H(0))  = rank (number of nonzero eigenvalues)
  exp(H(1))  = effective rank (participation ratio)
  exp(H(2))  = inverse participation ratio

The concentration ratio:
  C = 1 - exp(H(1)) / exp(H(0))
  = 1 - eff_rank / rank

C = 0 when all nonzero eigenvalues are equal (uniform spectrum).
C -> 1 when one eigenvalue dominates (peaked spectrum).

Adaptive damping:
  eps = eps_min + (eps_max - eps_min) * max(C, rank_deficiency)

where rank_deficiency = 1 - rank / dim.

This interpolates continuously between:
  - Low damping (trust the Fisher) when spectrum is uniform and full-rank
  - High damping (regularize toward SGD) when spectrum is peaked or rank-deficient
"""
import numpy as np
from scipy.special import logsumexp
from typing import Dict, Tuple


def renyi_entropy(eigvals: np.ndarray, alpha: float = 1.0) -> float:
    """Renyi entropy H(alpha) of the eigenspectrum.

    Args:
        eigvals: eigenvalues (need not be sorted, must be non-negative)
        alpha: order of the entropy

    Returns:
        H(alpha) in nats
    """
    eigvals = np.abs(eigvals)
    total = eigvals.sum()
    if total == 0:
        return 0.0

    p = eigvals / total
    p = p[p > 1e-15]  # remove zeros

    if len(p) == 0:
        return 0.0

    if abs(alpha - 1.0) < 1e-6:
        # Shannon entropy: H = -sum(p log p)
        return -np.sum(p * np.log(p))
    else:
        # Renyi: H_alpha = (1/(1-alpha)) * log(sum(p^alpha))
        # Use logsumexp for numerical stability
        log_p = np.log(p)
        return logsumexp(alpha * log_p) / (1.0 - alpha)


def spectral_stats(eigvals: np.ndarray) -> Dict[str, float]:
    """Compute spectral statistics from Fisher eigenvalues.

    Returns dict with:
        rank: numerical rank (eigenvalues above noise floor)
        dim: total dimension
        kappa: condition number (max/min nonzero eigenvalue)
        eff_rank: exp(H(1)), the participation ratio
        concentration: 1 - eff_rank/rank, spectral shape metric
        rank_deficiency: 1 - rank/dim
        H: dict of Renyi entropies at standard alpha values
    """
    eigvals = np.sort(np.abs(eigvals))[::-1]
    dim = len(eigvals)

    if eigvals[0] == 0:
        return {"rank": 0, "dim": dim, "kappa": float("inf"),
                "eff_rank": 0, "concentration": 1.0, "rank_deficiency": 1.0,
                "H": {}}

    # Numerical rank
    noise_floor = 1e-8 * eigvals[0]
    nonzero = eigvals[eigvals > noise_floor]
    rank = len(nonzero)

    if rank == 0:
        return {"rank": 0, "dim": dim, "kappa": float("inf"),
                "eff_rank": 0, "concentration": 1.0, "rank_deficiency": 1.0,
                "H": {}}

    kappa = nonzero[0] / max(nonzero[-1], 1e-15)

    # Renyi entropies
    alphas = [0.01, 0.5, 1.0, 2.0, 5.0, 10.0]
    H = {a: renyi_entropy(eigvals, alpha=a) for a in alphas}

    # Effective rank = exp(H(1))
    eff_rank = np.exp(H[1.0])

    # Concentration: 0 = uniform, 1 = peaked
    rank_from_H0 = np.exp(H[0.01])  # approximates rank
    concentration = 1.0 - eff_rank / max(rank_from_H0, 1.0)

    # Rank deficiency
    rank_deficiency = 1.0 - rank / dim

    return {
        "rank": rank, "dim": dim, "kappa": kappa,
        "eff_rank": eff_rank, "concentration": concentration,
        "rank_deficiency": rank_deficiency, "H": H,
    }


def adaptive_damping(eigvals: np.ndarray, eps_min: float = 1e-8,
                     eps_max: float = 0.5) -> Tuple[float, Dict]:
    """Compute adaptive Tikhonov damping from Fisher eigenspectrum.

    The damping interpolates continuously between:
      - eps_min (trust the Fisher) when spectrum is uniform and full-rank
      - eps_max (regularize toward identity) when spectrum is peaked or rank-deficient

    The interpolation is driven by max(concentration, rank_deficiency),
    where:
      concentration = 1 - exp(H(1)) / exp(H(0))  (spectral shape)
      rank_deficiency = 1 - rank / dim            (estimation quality)

    Args:
        eigvals: Fisher eigenvalues
        eps_min: minimum damping (RICH regime)
        eps_max: maximum damping (BLIND regime)

    Returns:
        (damping, stats_dict)
    """
    stats = spectral_stats(eigvals)
    ratio = max(stats["concentration"], stats["rank_deficiency"])
    damping = eps_min + (eps_max - eps_min) * ratio
    return damping, stats
