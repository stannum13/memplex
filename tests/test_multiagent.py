"""Tests for multi-agent knowledge sharing experiment."""
import pytest
import numpy as np


class TestMultiAgent:
    def test_sharing_beats_no_sharing(self):
        """Any sharing protocol should beat no sharing."""
        from memplex.experiments.multiagent import run_multi_agent_experiment, MultiAgentConfig
        r = run_multi_agent_experiment(MultiAgentConfig(n_steps=200, n_expert=200, n_noisy=30))
        # FedAvg should beat individual agents
        assert r["fedavg"] > r["A_alone"]
        assert r["fedavg"] > r["B_alone"]

    def test_fisher_methods_beat_fedavg(self):
        """Bayesian and gated should beat naive FedAvg."""
        from memplex.experiments.multiagent import run_multi_agent_experiment, MultiAgentConfig
        r = run_multi_agent_experiment(MultiAgentConfig(n_steps=200, n_expert=200, n_noisy=30))
        assert r["bayesian"] > r["fedavg"]
        assert r["hard_gate"] > r["fedavg"]

    def test_fisher_gated_share_function(self):
        """The sharing function should return valid weights."""
        from memplex.experiments.multiagent import fisher_gated_share
        import numpy as np
        dim = 20
        W_A = np.random.randn(dim)
        W_B = np.random.randn(dim)
        W_avg = (W_A + W_B) / 2
        F_A = np.eye(dim)
        F_B = np.eye(dim) * 0.1

        W_hard = fisher_gated_share(W_A, F_A, W_B, F_B, W_avg, method="hard")
        W_soft = fisher_gated_share(W_A, F_A, W_B, F_B, W_avg, method="soft")

        assert W_hard.shape == (dim,)
        assert W_soft.shape == (dim,)
        assert not np.any(np.isnan(W_hard))
        assert not np.any(np.isnan(W_soft))
