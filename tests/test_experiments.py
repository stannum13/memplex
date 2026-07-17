"""Tests for experiments: regime comparison and transition experiment."""

class TestRegimeExperiment:
    def test_ng_beats_adam_in_rich(self):
        """NG should converge faster than Adam in Fisher-RICH regime."""
        from memplex.experiments.regimes import run_regime_experiment, ExperimentConfig
        r = run_regime_experiment(ExperimentConfig(n_steps=50))
        ng = r["rich"]["ng"][49]
        adam = r["rich"]["adam"][49]
        assert ng < adam * 0.1, f"NG ({ng:.6f}) should be <10% of Adam ({adam:.6f})"

    def test_adaptive_beats_adam(self):
        """Adaptive damping should beat Adam in RICH."""
        from memplex.experiments.regimes import run_regime_experiment, ExperimentConfig
        r = run_regime_experiment(ExperimentConfig(n_steps=50))
        adapt = r["rich"]["adaptive"][49]
        adam = r["rich"]["adam"][49]
        assert adapt < adam, f"Adaptive ({adapt:.6f}) should beat Adam ({adam:.6f})"


class TestTransitionExperiment:
    def test_ng_diverges_in_blind(self):
        """Fixed NG should diverge in the BLIND phase."""
        from memplex.experiments.transition import run_transition_experiment, TransitionConfig
        r = run_transition_experiment(TransitionConfig(n_steps=150, shift_step=75))
        ng_losses = r["ng"]
        # NG should have very high loss in BLIND phase
        blind_loss = max(ng_losses[75:])
        assert blind_loss > 100, f"NG should diverge in BLIND, max loss={blind_loss}"

    def test_adaptive_does_not_diverge(self):
        """Adaptive NG should not diverge."""
        from memplex.experiments.transition import run_transition_experiment, TransitionConfig
        r = run_transition_experiment(TransitionConfig(n_steps=150, shift_step=75))
        adapt_losses = r["adaptive"]
        max_loss = max(adapt_losses)
        assert max_loss < 100, f"Adaptive should not diverge, max loss={max_loss}"

    def test_adaptive_beats_adam(self):
        """Adaptive cumulative loss should be lower than Adam."""
        from memplex.experiments.transition import run_transition_experiment, TransitionConfig
        r = run_transition_experiment(TransitionConfig(n_steps=150, shift_step=75))
        assert r["cum_adaptive"] < r["cum_adam"]
