"""Tests for spectral entropy and adaptive damping."""
import numpy as np
from memplex.fisher.spectrum import renyi_entropy, spectral_stats, adaptive_damping


class TestRenyiEntropy:
    def test_uniform_spectrum(self):
        """Uniform eigenvalues → maximum entropy."""
        eigs = np.ones(10)
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            h = renyi_entropy(eigs, alpha=alpha)
            assert abs(h - np.log(10)) < 0.01, f"H({alpha})={h}, expected {np.log(10)}"

    def test_peaked_spectrum(self):
        """One dominant eigenvalue → low entropy at high alpha."""
        eigs = np.array([100.0] + [0.001]*9)
        h_low = renyi_entropy(eigs, alpha=0.01)
        h_high = renyi_entropy(eigs, alpha=10.0)
        assert h_low > h_high, "Peaked spectrum should have lower entropy at high alpha"

    def test_zero_eigenvalues(self):
        """Should handle zeros gracefully."""
        eigs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        h = renyi_entropy(eigs, alpha=1.0)
        assert h == 0.0  # only one nonzero eigenvalue → entropy = 0

    def test_all_zero(self):
        eigs = np.zeros(10)
        assert renyi_entropy(eigs, alpha=1.0) == 0.0

    def test_shannon_equals_special_case(self):
        """H(1) should equal Shannon entropy."""
        eigs = np.array([3.0, 1.0, 1.0, 0.5, 0.5])
        p = eigs / eigs.sum()
        shannon = -np.sum(p * np.log(p))
        h1 = renyi_entropy(eigs, alpha=1.0)
        assert abs(h1 - shannon) < 1e-10


class TestSpectralStats:
    def test_rich_spectrum(self):
        """Well-conditioned Fisher: low kappa, high eff_rank."""
        eigs = np.ones(20) * 1.0
        stats = spectral_stats(eigs)
        assert stats["kappa"] < 2
        assert stats["rank"] == 20
        assert stats["eff_rank"] > 15
        assert stats["concentration"] < 0.1

    def test_blind_spectrum(self):
        """Rank-deficient Fisher: low rank, high concentration."""
        eigs = np.array([100.0] + [1e-12]*19)
        stats = spectral_stats(eigs)
        assert stats["rank"] == 1  # only one above noise floor
        assert stats["concentration"] > 0.9
        assert stats["rank_deficiency"] > 0.9

    def test_sparse_spectrum(self):
        """Bimodal: few large, many small."""
        eigs = np.array([10.0]*5 + [0.01]*15)
        stats = spectral_stats(eigs)
        assert stats["kappa"] > 100
        assert 0.3 < stats["concentration"] < 0.9


class TestAdaptiveDamping:
    def test_rich_low_damping(self):
        """Full-rank, uniform spectrum → low damping."""
        eigs = np.ones(20)
        damping, stats = adaptive_damping(eigs, eps_min=1e-8, eps_max=0.5)
        assert damping < 0.01, f"Damping should be low in RICH, got {damping}"

    def test_blind_high_damping(self):
        """Rank-deficient → high damping."""
        eigs = np.array([100.0] + [1e-12]*19)
        damping, stats = adaptive_damping(eigs, eps_min=1e-8, eps_max=0.5)
        assert damping > 0.4, f"Damping should be high in BLIND, got {damping}"

    def test_monotonic(self):
        """Damping should increase as spectrum becomes more peaked."""
        dampings = []
        # Start from uniform, gradually concentrate mass in one direction
        for peak in [1.0, 10.0, 100.0, 10000.0]:
            # Keep small eigenvalues fixed at 1e-12 (below noise floor)
            eigs = np.array([peak] + [1e-12]*19)
            d, _ = adaptive_damping(eigs, eps_min=1e-8, eps_max=0.5)
            dampings.append(d)
        assert all(dampings[i] <= dampings[i+1] for i in range(len(dampings)-1)), \
            f"Damping should be monotonic: {dampings}"
