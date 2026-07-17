"""Tests for regime detector (kept from original)."""
import numpy as np
from memplex.fisher.regime import RegimeDetector, RegimeConfig, Regime


def test_regime_rich():
    det = RegimeDetector(dim=10)
    G = np.eye(10) * 2.0
    report = det.update(G)
    assert report.regime == Regime.FISHER_RICH


def test_regime_sparse():
    det = RegimeDetector(dim=10)
    G = np.diag([1000.0] + [0.001] * 9)
    report = det.update(G)
    assert report.regime == Regime.FISHER_SPARSE


def test_regime_blind():
    det = RegimeDetector(dim=10)
    G = np.eye(10) * 1e-10
    report = det.update(G)
    assert report.regime == Regime.FISHER_BLIND


def test_regime_shift_flag():
    """External shift signal forces BLIND regime."""
    det = RegimeDetector(dim=10, config=RegimeConfig(blind_after_shift_steps=5))
    G = np.eye(10) * 2.0
    assert det.update(G).regime == Regime.FISHER_RICH
    det.flag_shift()
    assert det.update(G).regime == Regime.FISHER_BLIND
    for _ in range(5):
        det.update(G)
    assert det.update(G).regime == Regime.FISHER_RICH
