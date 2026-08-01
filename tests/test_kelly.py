"""Tests for sizing.py and risk.py.

Two tests here are worth reading even if you skip the rest:

  test_kelly_reduces_to_the_classic_formula  -- the cheapest possible check
  that the general multi-outcome Kelly formula is right, by confirming it
  reproduces the familiar two-outcome one.

  test_var_sign_convention -- the sign of a loss is the most commonly
  inverted thing in risk code, and inverting it produces numbers that still
  look entirely reasonable.
"""

import numpy as np
import pytest

from blackjack import risk
from blackjack.counting import PreDealTracker
from blackjack.sizing import FlatSizer, KellySizer, kelly_fraction


# --------------------------------------------------------------- Kelly
def test_kelly_reduces_to_the_classic_formula():
    """For an even-money two-outcome bet, mu/E[X^2] must equal p - q.

    X = +1 with probability p, -1 with probability q = 1-p. Then
    mu = p - q and E[X^2] = 1, so f* = p - q, the textbook Kelly fraction.
    If the general implementation did not reproduce this, it would be wrong.
    """
    for p in (0.5, 0.55, 0.6, 0.75):
        q = 1 - p
        mu, ex2 = p - q, 1.0
        assert kelly_fraction(mu, ex2) == pytest.approx(p - q)


def test_kelly_is_zero_for_a_non_positive_edge():
    assert kelly_fraction(0.0, 1.0) == 0.0
    assert kelly_fraction(-0.05, 1.0) == 0.0


def test_kelly_grows_with_edge_and_shrinks_with_second_moment():
    assert kelly_fraction(0.02, 1.0) > kelly_fraction(0.01, 1.0)
    assert kelly_fraction(0.01, 2.0) < kelly_fraction(0.01, 1.0)


def test_wider_payoffs_call_for_a_smaller_fraction():
    """Doubling widens the payoff range, raising E[X^2], so f* falls.

    This is the result that sounds wrong at first: a better strategy calls
    for a smaller bet fraction, because Kelly prices variance as well as
    expectation.
    """
    edge = 0.01
    ex2_no_double = 1.0
    ex2_with_double = 1.3
    assert kelly_fraction(edge, ex2_with_double) < kelly_fraction(edge, ex2_no_double)


def test_kelly_handles_degenerate_inputs():
    assert kelly_fraction(np.nan, 1.0) == 0.0
    assert kelly_fraction(0.01, 0.0) == 0.0
    assert kelly_fraction(0.01, np.nan) == 0.0


# ---------------------------------------------------------------- sizer
def _tracker_with(bin_tc: float, payoffs: list[float], repeat: int) -> PreDealTracker:
    t = PreDealTracker()
    for _ in range(repeat):
        for x in payoffs:
            t.record(bin_tc, x)
    return t


def test_sizer_bets_minimum_where_there_is_no_edge():
    t = _tracker_with(0.5, [1.0, -1.0], 5000)      # exactly break-even
    s = KellySizer(t)
    assert s.bet(0.5, 1000.0) == pytest.approx(1.0)


def test_sizer_bets_more_than_minimum_on_a_clear_edge():
    """A large, well-measured positive edge should produce a real bet."""
    t = _tracker_with(3.5, [1.0] * 6 + [-1.0] * 4, 20_000)   # +20% edge
    s = KellySizer(t)
    assert s.bet(3.5, 1000.0) > 1.0


def test_significance_gate_blocks_a_noisy_positive_edge():
    """A positive point estimate from few samples must NOT be bet on.

    Twelve hands showing a 33% edge is noise, not a signal. The gate
    requires the lower confidence bound to clear zero, so this bin is
    refused; with the gate switched off, the same data would be bet on --
    which is exactly the systematic overbetting the gate exists to stop.
    """
    t = _tracker_with(3.5, [1.0, 1.0, -1.0], 4)      # 12 hands, +33% "edge"
    gated = KellySizer(t, use_gate=True)
    ungated = KellySizer(t, use_gate=False)
    assert gated.bet(3.5, 1000.0) == pytest.approx(1.0)
    assert ungated.bet(3.5, 1000.0) > 1.0


def test_bet_never_exceeds_the_cap_or_the_bankroll():
    t = _tracker_with(3.5, [1.0] * 9 + [-1.0], 20_000)    # huge edge
    s = KellySizer(t, f_max=0.02)
    assert s.bet(3.5, 1000.0) <= 0.02 * 1000.0 + 1e-9
    assert s.bet(3.5, 0.5) <= 0.5


def test_half_kelly_bets_half_of_full_kelly():
    t = _tracker_with(3.5, [1.0] * 6 + [-1.0] * 4, 20_000)
    full = KellySizer(t, lam=1.0, f_max=1.0)
    half = KellySizer(t, lam=0.5, f_max=1.0)
    assert half.bet(3.5, 1000.0) == pytest.approx(full.bet(3.5, 1000.0) / 2)


def test_flat_sizer_ignores_the_count():
    f = FlatSizer(size=5.0)
    assert f.bet(-5.0, 1000.0) == f.bet(5.0, 1000.0) == 5.0


# ----------------------------------------------------------------- risk
def test_var_sign_convention():
    """Losses are positive. VaR of a losing distribution must be positive.

    This is the test that catches a flipped sign -- the most common bug in
    risk code, and one that produces perfectly plausible-looking output.
    """
    losses = np.array([-10.0, -5.0, 0.0, 5.0, 50.0])   # last one = lost 50
    assert risk.var(losses, 0.99) > 0


def test_var_is_the_quantile():
    losses = np.arange(101, dtype=float)      # 0..100
    assert risk.var(losses, 0.95) == pytest.approx(95.0)


def test_cvar_is_at_least_var_always():
    """A mathematical identity, so it is checked on many random samples."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        losses = rng.standard_t(df=3, size=500) * 10
        for alpha in (0.9, 0.95, 0.99):
            assert risk.cvar(losses, alpha) >= risk.var(losses, alpha) - 1e-9


def test_cvar_handles_a_probability_atom_at_var():
    """The case that breaks the naive implementation.

    99 paths break even, one is wiped out. VaR at 90% is 0, so averaging
    everything at-or-above VaR averages all 100 paths and gives 1.0. The
    worst 10% is nine zeros and one 100, whose mean is 10.0. Discrete payoffs
    make this situation ordinary rather than exotic, which is why the naive
    version is a real bug and not a technicality.
    """
    losses = np.array([0.0] * 99 + [100.0])
    assert risk.var(losses, 0.90) == pytest.approx(0.0)
    assert risk.cvar(losses, 0.90) == pytest.approx(10.0)


def test_cvar_reduces_to_the_simple_average_without_atoms():
    """With no mass sitting at VaR, the correction term must vanish."""
    losses = np.arange(1000, dtype=float)
    v = risk.var(losses, 0.99)
    simple = losses[losses >= v].mean()
    assert risk.cvar(losses, 0.99) == pytest.approx(simple, rel=2e-3)


def test_cvar_sees_tail_shape_that_var_misses():
    """Two samples with the same VaR but very different CVaR.

    This is the concrete reason regulators moved from VaR to expected
    shortfall: VaR cannot distinguish a mild tail from a catastrophic one.
    """
    mild = np.array([0.0] * 99 + [100.0])
    severe = np.array([0.0] * 99 + [10_000.0])
    assert risk.var(mild, 0.98) == risk.var(severe, 0.98)
    assert risk.cvar(severe, 0.98) > risk.cvar(mild, 0.98)


def test_max_drawdown_is_a_fraction():
    path = np.array([100.0, 120.0, 60.0, 90.0])
    # peak 120 then trough 60 -> drawdown 0.5
    assert risk.max_drawdown(path) == pytest.approx(0.5)
    assert 0.0 <= risk.max_drawdown(path) <= 1.0


def test_max_drawdown_of_a_rising_path_is_zero():
    assert risk.max_drawdown(np.array([1.0, 2.0, 3.0, 4.0])) == 0.0


def test_drawdown_uses_the_running_peak_not_the_start():
    """A path that rises then falls back to its start has a real drawdown."""
    assert risk.max_drawdown(np.array([100.0, 200.0, 100.0])) == pytest.approx(0.5)


def test_growth_rate_of_a_ruined_path_is_minus_infinity():
    assert risk.growth_rate(1000.0, 0.0, 100) == -np.inf


def test_growth_rate_matches_the_definition():
    g = risk.growth_rate(100.0, 200.0, 10)
    assert g == pytest.approx(np.log(2) / 10)


def test_risk_of_ruin_counts_paths_that_ever_touched():
    """A path that dips below the threshold and recovers still counts."""
    paths = np.array([[100.0, 40.0, 100.0],     # dipped -> ruined
                      [100.0, 90.0, 95.0]])     # never dipped
    assert risk.risk_of_ruin(paths, 100.0, 0.5) == pytest.approx(0.5)


def test_theoretical_ruin_matches_the_published_figures():
    """Full Kelly halves the bankroll with probability 1/2; half Kelly 1/8."""
    assert risk.theoretical_ruin(1.0, 0.5) == pytest.approx(0.5)
    assert risk.theoretical_ruin(0.5, 0.5) == pytest.approx(0.125)
    assert risk.theoretical_ruin(0.25, 0.5) == pytest.approx(0.5 ** 7)


def test_theoretical_ruin_is_monotone_in_lambda():
    lams = [0.25, 0.5, 0.75, 1.0]
    ruins = [risk.theoretical_ruin(l) for l in lams]
    assert ruins == sorted(ruins)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    losses = rng.normal(0, 10, 2000)
    point = risk.cvar(losses, 0.95)
    lo, hi = risk.bootstrap_ci(losses, lambda x: risk.cvar(x, 0.95), rng=rng)
    assert lo <= point <= hi


def test_summarise_asserts_cvar_above_var_and_reports_everything():
    rng = np.random.default_rng(2)
    paths = 1000.0 + np.cumsum(rng.normal(0, 5, (50, 200)), axis=1)
    paths = np.hstack([np.full((50, 1), 1000.0), paths])
    out = risk.summarise(paths, 1000.0, rng=rng)
    for key in ("var", "cvar", "mdd_mean", "ruin", "growth_median", "cvar_ci"):
        assert key in out
    assert out["cvar"] >= out["var"] - 1e-9
    assert 0.0 <= out["ruin"] <= 1.0
    assert 0.0 <= out["mdd_mean"] <= 1.0
