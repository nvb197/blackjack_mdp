"""Tests for simulate.py.

This module had no tests at all, which matters more here than elsewhere: it
contains the one line where a look-ahead bug has already occurred once in
this project's history. `bankroll_paths` sizes each bet from
`env.pre_deal_true_count()`, and an earlier version called `env.true_count()`
instead -- reading the count of a shoe that had just reached penetration and
was about to be shuffled away. Nothing crashed; the bet was simply sized on
information that had expired.

test_stake_is_sized_on_the_count_that_preceded_the_deal exists to make that
class of mistake impossible to reintroduce silently.
"""

import numpy as np
import pytest

from blackjack import dp
from blackjack.finite_env import FiniteBlackjackEnv
from blackjack.simulate import (bankroll_paths, basic_strategy, measure_edge,
                                play_hand)
from blackjack.sizing import FlatSizer


class RecordingSizer:
    """A sizer that bets a flat unit and remembers every count it was given.

    Duck-types FlatSizer: `bankroll_paths` only ever calls `.bet(tc, bank)`,
    so anything with that method can stand in. That is what makes the
    sizing decision observable from outside without touching the code under
    test.
    """

    def __init__(self):
        self.seen: list[float] = []

    def bet(self, pre_deal_tc: float, bankroll: float) -> float:
        self.seen.append(float(pre_deal_tc))
        return 1.0


# ------------------------------------------------------------------ #
# The invariant this file exists for
# ------------------------------------------------------------------ #
def test_stake_is_sized_on_the_count_that_preceded_the_deal():
    """Every count handed to the sizer must be the pre-deal count.

    The check replays the identical seed by hand, recording the count
    *before* each hand is dealt, and demands the sizer saw exactly that
    sequence. If the stake were sized after the deal -- or on the stale
    count of a shoe due to be shuffled -- the two lists would diverge.

    500 hands is chosen deliberately: at 75% penetration a shoe lasts
    roughly 40 hands, so this exercises about a dozen reshuffles, which is
    where the stale-count version of the bug shows up.
    """
    n_hands = 500
    sizer = RecordingSizer()
    bankroll_paths(sizer, n_paths=1, n_hands=n_hands, initial=1e9, seed=100)

    pi, should_double = basic_strategy(allow_double=True)
    env = FiniteBlackjackEnv(np.random.default_rng(100), allow_double=True)
    expected = []
    for _ in range(n_hands):
        expected.append(env.pre_deal_true_count())
        play_hand(env, pi, should_double)

    assert len(sizer.seen) == n_hands
    assert sizer.seen == pytest.approx(expected)


def test_sizer_is_given_zero_when_the_shoe_is_about_to_be_shuffled():
    """At penetration the bettor must size on zero, not the dying shoe's count.

    A player watches the shuffle, so once the shoe is spent they know the
    count is about to reset. Sizing on the expiring count would concentrate
    a fake signal in exactly the extreme bins Kelly bets hardest on.
    """
    sizer = RecordingSizer()
    bankroll_paths(sizer, n_paths=1, n_hands=500, initial=1e9, seed=100)

    env = FiniteBlackjackEnv(np.random.default_rng(100), allow_double=True)
    pi, should_double = basic_strategy(allow_double=True)
    saw_a_shuffle = False
    for i in range(500):
        if env.shoe.needs_reshuffle():
            saw_a_shuffle = True
            assert sizer.seen[i] == 0.0, (
                "sized on the stale count of a shoe about to be shuffled")
        play_hand(env, pi, should_double)
    assert saw_a_shuffle, "500 hands never reached penetration -- check the shoe"


# ------------------------------------------------------------------ #
# Bankroll mechanics
# ------------------------------------------------------------------ #
def test_bankroll_never_goes_negative_and_ruin_is_absorbing():
    """A bankroll cannot fall below zero, and once at zero it stays there.

    Started at 2 units against a 1-unit minimum bet so that ruin actually
    happens -- the `if bank <= 0` branch in bankroll_paths is otherwise
    never executed by any other test.
    """
    paths = bankroll_paths(FlatSizer(1.0), n_paths=30, n_hands=300,
                           initial=2.0, seed=5)
    assert np.all(paths >= 0.0)

    ruined_any = False
    for p in paths:
        zeros = np.flatnonzero(p == 0.0)
        if zeros.size:
            ruined_any = True
            assert np.all(p[zeros[0]:] == 0.0), "a ruined path recovered"
    assert ruined_any, "no path was ruined -- the absorbing branch went untested"


def test_paths_start_at_the_initial_bankroll():
    paths = bankroll_paths(FlatSizer(1.0), n_paths=4, n_hands=20,
                           initial=500.0, seed=1)
    assert paths.shape == (4, 21)
    assert np.all(paths[:, 0] == 500.0)


def test_same_seed_gives_identical_bankroll_paths():
    """Reproducibility is the project's central guarantee; pin it here too."""
    a = bankroll_paths(FlatSizer(1.0), n_paths=3, n_hands=200, seed=7)
    b = bankroll_paths(FlatSizer(1.0), n_paths=3, n_hands=200, seed=7)
    assert np.array_equal(a, b)


def test_different_seeds_give_different_paths():
    """The counterpart to the test above: the seed must actually matter."""
    a = bankroll_paths(FlatSizer(1.0), n_paths=2, n_hands=200, seed=7)
    b = bankroll_paths(FlatSizer(1.0), n_paths=2, n_hands=200, seed=999)
    assert not np.array_equal(a, b)


def test_each_path_uses_its_own_shoe():
    """Paths must be independent, not continuations of one another.

    bankroll_paths builds a fresh environment per path with seed + p. If it
    reused one environment, path 2 would start from a shoe that path 1 had
    already partly dealt, and the paths would not be independent samples.
    """
    paths = bankroll_paths(FlatSizer(1.0), n_paths=2, n_hands=150, seed=100)
    assert not np.array_equal(paths[0], paths[1])


# ------------------------------------------------------------------ #
# The policy tables and the edge measurement
# ------------------------------------------------------------------ #
def test_basic_strategy_matches_the_exact_solution():
    """The tables played against the shoe must be Phase 1's optimal policy.

    If these drifted apart, Phase 3 would be measuring the edge of some
    other strategy while the write-up claimed it was the optimal one.
    """
    pi_sim, should_double = basic_strategy(allow_double=True)
    V, pi_dp, _ = dp.value_iteration()
    assert np.array_equal(pi_sim, pi_dp)
    assert np.array_equal(should_double, dp.double_values() > V)


def test_basic_strategy_without_doubling_never_doubles():
    _, should_double = basic_strategy(allow_double=False)
    assert not should_double.any()


def test_measure_edge_records_every_hand_exactly_once():
    """No hand may be dropped or double-counted.

    Naturals end before any decision and are easy to skip by accident; they
    still carry a payoff and must appear in the tracker.
    """
    n = 20_000
    tracker = measure_edge(n, seed=3, allow_double=True)
    assert tracker.total() == n


def test_measure_edge_is_reproducible():
    a = measure_edge(5_000, seed=11, allow_double=True)
    b = measure_edge(5_000, seed=11, allow_double=True)
    assert np.array_equal(a.hist, b.hist)


def test_play_hand_returns_a_legal_payoff():
    """Payoffs must come from the six values the game can produce."""
    legal = {-2.0, -1.0, 0.0, 1.0, 1.5, 2.0}
    pi, should_double = basic_strategy(allow_double=True)
    env = FiniteBlackjackEnv(np.random.default_rng(21), allow_double=True)
    for _ in range(3_000):
        _, payoff = play_hand(env, pi, should_double)
        assert payoff in legal


def test_doubling_is_refused_when_the_bankroll_cannot_cover_it():
    """A hand cannot place a second stake the bankroll cannot pay.

    Starting at 2 units against a 1-unit minimum, roughly 7% of doubles were
    previously placed without the funds to cover them. The loss was clamped
    at zero, so the casino silently absorbed the shortfall -- uncollateralised
    credit, which flatters both drawdown and risk of ruin.
    """
    pi, should_double = basic_strategy(allow_double=True)
    env = FiniteBlackjackEnv(np.random.default_rng(500), allow_double=True)
    for _ in range(2000):
        # bankroll below twice the stake: doubling must never happen
        _, payoff = play_hand(env, pi, should_double, can_double=False)
        assert abs(payoff) != 2.0, "doubled without the funds to cover it"


def test_bankroll_paths_refuse_doubles_they_cannot_fund():
    """A path below twice the stake must never lose more than one stake.

    This is the check that actually detects unfunded doubling, and my first
    attempt at it did not. Testing play_hand(can_double=False) directly only
    proves the parameter works; it says nothing about whether bankroll_paths
    passes the right value. The clamp at zero also hides the symptom -- a
    bankroll of 1.5 losing a 2-unit double simply lands on 0, which looks
    like any other ruin.

    The signature that survives the clamp: with a flat 1-unit stake, a
    bankroll below 2.0 cannot fund a double, so a single hand may cost at
    most 1.0. A larger drop in that regime means a double was placed on
    credit.
    """
    paths = bankroll_paths(FlatSizer(1.0), n_paths=40, n_hands=300,
                           initial=2.0, seed=500)
    assert np.all(paths >= 0.0)

    unfunded = 0
    for p in paths:
        for before, after in zip(p[:-1], p[1:]):
            if 0.0 < before < 2.0 and (before - after) > 1.0 + 1e-9:
                unfunded += 1
    assert unfunded == 0, (
        f"{unfunded} hands lost more than one stake while unable to fund a "
        "double -- the casino absorbed the shortfall")