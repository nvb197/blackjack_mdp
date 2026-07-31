"""Tests for counting.py and finite_env.py.

The most important test in this file is
test_pre_deal_count_precedes_the_deal. Read its docstring even if you skip
the rest: it is the one guarding against the bug that would invalidate all
of Phase 3 and 4 without producing any error message.
"""

import numpy as np
import pytest

from blackjack import counting as ct
from blackjack.counting import PreDealTracker, running_count, tc_bin, true_count
from blackjack.finite_env import FiniteBlackjackEnv
from blackjack.rules import make_rng
from blackjack.shoe import SHOE_SIZE


# ---------------------------------------------------------------- Hi-Lo
def test_hilo_values_are_the_standard_ones():
    assert [ct.HILO[c] for c in range(1, 11)] == [-1, 1, 1, 1, 1, 1, 0, 0, 0, -1]


def test_hilo_is_balanced_over_a_full_shoe():
    """A full shoe must count to exactly zero.

    Hi-Lo is a "balanced" system: 20 low cards at +1 and 20 high cards at -1
    per deck cancel exactly (4 each of ranks 2-6 = 20; 16 tens + 4 aces =
    20). That is what makes the running count meaningful as a measure of
    *deviation* from a neutral shoe -- if it did not balance, the count
    would drift even through an untouched shoe and mean nothing.
    """
    from blackjack.shoe import CARDS_PER_RANK
    assert int((ct.HILO * CARDS_PER_RANK).sum()) == 0


def test_running_count_of_low_cards_is_positive():
    assert running_count([2, 3, 4, 5, 6]) == 5
    assert running_count([10, 10, 1]) == -3
    assert running_count([7, 8, 9]) == 0


# ------------------------------------------------------------ true count
def test_true_count_normalises_by_decks_remaining():
    """The same running count means more when fewer decks are left."""
    assert true_count(6, 6 * 52) == pytest.approx(1.0)
    assert true_count(6, 3 * 52) == pytest.approx(2.0)
    assert true_count(6, 1 * 52) == pytest.approx(6.0)


def test_true_count_of_zero_is_zero_regardless_of_depth():
    for remaining in (312, 200, 100, 26):
        assert true_count(0, remaining) == 0.0


def test_true_count_does_not_divide_by_zero():
    assert np.isfinite(true_count(5, 0))


@pytest.mark.parametrize("tc,expected_label", [
    (-9.0, "<-3"), (-3.0, "-3..-2"), (-1.5, "-2..-1"), (-0.2, "-1..0"),
    (0.0, "0..1"), (0.5, "0..1"), (1.0, "1..2"), (2.9, "2..3"),
    (3.0, ">=3"), (12.0, ">=3"),
])
def test_tc_bin_boundaries(tc, expected_label):
    """Bins are half-open [lo, hi): a true count of exactly 1.0 is in 1..2."""
    assert ct.BIN_LABELS[tc_bin(tc)] == expected_label


def test_every_bin_index_is_in_range():
    for tc in np.linspace(-20, 20, 501):
        assert 0 <= tc_bin(tc) < ct.N_BINS


# --------------------------------------------------------------- tracker
def test_tracker_records_into_the_right_bin():
    t = PreDealTracker()
    t.record(2.5, 1.0)
    t.record(2.5, -1.0)
    t.record(-4.0, 1.5)
    assert t.n(tc_bin(2.5)) == 2
    assert t.n(tc_bin(-4.0)) == 1
    assert t.total() == 3


def test_tracker_mean_matches_hand_computation():
    t = PreDealTracker()
    for payoff in (1.0, 1.0, -1.0, 0.0, 1.5):
        t.record(0.5, payoff)
    s = t.stats(tc_bin(0.5))
    assert s["n"] == 5
    assert s["mean"] == pytest.approx((1 + 1 - 1 + 0 + 1.5) / 5)


def test_tracker_second_moment_is_not_the_variance():
    """E[X^2] must include the mean^2 term -- Kelly needs E[X^2], not Var."""
    t = PreDealTracker()
    for payoff in (2.0, 2.0, 2.0, 2.0):
        t.record(0.0, payoff)
    s = t.stats(tc_bin(0.0))
    assert s["ex2"] == pytest.approx(4.0)
    assert s["std"] == pytest.approx(0.0)


def test_tracker_empty_bin_is_reported_not_crashed():
    t = PreDealTracker()
    s = t.stats(0)
    assert s["n"] == 0 and np.isnan(s["mean"])


def test_confidence_interval_narrows_as_the_root_of_n():
    """Doubling the sample should shrink the interval by about sqrt(2)."""
    t1, t2 = PreDealTracker(), PreDealTracker()
    rng = make_rng(0)
    draws = rng.choice([-1.0, 1.0], size=8000)
    for x in draws[:2000]:
        t1.record(0.5, float(x))
    for x in draws:
        t2.record(0.5, float(x))
    w1 = t1.stats(tc_bin(0.5))["se"]
    w2 = t2.stats(tc_bin(0.5))["se"]
    assert w1 / w2 == pytest.approx(2.0, rel=0.1)


def test_sample_size_needed_for_a_one_percent_edge():
    """About 38,000 hands per bin -- see the method's docstring."""
    t = PreDealTracker()
    n = t.hands_needed_for_significance(edge=0.01, sigma=1.0)
    assert 30_000 < n < 45_000


# ------------------------------------------------------------ finite env
def test_running_count_matches_recomputation():
    """Maintaining the count incrementally must equal counting from scratch.

    This guards the one place cards are dealt (_draw) against counting a
    card twice or not at all -- a corruption that no other test could see,
    because a slightly-wrong count still looks like a perfectly plausible
    count.
    """
    env = FiniteBlackjackEnv(make_rng(1))
    for _ in range(300):
        _, info = env.reset()
        if not info["done"]:
            done = False
            while not done:
                _, _, done, _ = env.step(0)
        dealt = env.shoe.cards[:env.shoe.pos]
        assert env.running == running_count(dealt)


def test_pre_deal_count_precedes_the_deal():
    """THE critical test: info["pre_deal_tc"] must be the count BEFORE dealing.

    Read this if you read nothing else.

    A bet is placed before any card of the hand is visible. So the only
    count a bettor may use is the count as it stood before the deal. If the
    code instead reported the count *after* the hand's cards came out, and
    the outcome of that hand were then attributed to it, the analysis would
    be using information that did not exist at decision time -- look-ahead
    bias, the same error as a backtest reading tomorrow's prices.

    That bug raises no exception. It produces a smooth, convincing,
    completely fictitious edge curve. This test is the only thing standing
    between the project and that outcome, so it checks the invariant
    directly: read the count from the outside before reset(), and demand
    that reset() reports exactly that value.
    """
    env = FiniteBlackjackEnv(make_rng(2))
    for _ in range(400):
        expected = env.true_count()
        cards_before = env.shoe.pos
        _, info = env.reset()
        if env.shoe.pos < cards_before:      # a reshuffle happened
            assert info["pre_deal_tc"] == 0.0
        else:
            assert info["pre_deal_tc"] == pytest.approx(expected)
            assert env.shoe.pos > cards_before, "reset dealt no cards"
        if not info["done"]:
            done = False
            while not done:
                _, _, done, _ = env.step(0)


def test_count_resets_to_zero_after_reshuffle():
    env = FiniteBlackjackEnv(make_rng(3))
    saw_reshuffle = False
    for _ in range(2000):
        before = env.shoe.pos
        _, info = env.reset()
        if env.shoe.pos < before:
            saw_reshuffle = True
            # after a reshuffle only this hand's cards have been counted
            assert abs(env.running) <= 6
        if not info["done"]:
            done = False
            while not done:
                _, _, done, _ = env.step(0)
    assert saw_reshuffle, "no reshuffle occurred -- penetration logic broken?"


def test_card_conservation_over_many_hands():
    env = FiniteBlackjackEnv(make_rng(4))
    for _ in range(3000):
        _, info = env.reset()
        assert env.shoe.cards_dealt() + env.shoe.cards_remaining() == SHOE_SIZE
        if not info["done"]:
            done = False
            while not done:
                _, _, done, _ = env.step(1 if env.shoe.pos % 3 else 0)


def test_true_count_is_roughly_symmetric_around_zero():
    """Over many hands the pre-deal true count should centre on zero.

    A balanced count over a shoe dealt from a fair shuffle has no reason to
    drift either way. A clear bias here would mean the count is being
    updated on the wrong cards, or reset at the wrong moment.
    """
    env = FiniteBlackjackEnv(make_rng(5))
    tcs = []
    for _ in range(20_000):
        _, info = env.reset()
        tcs.append(info["pre_deal_tc"])
        if not info["done"]:
            done = False
            while not done:
                _, _, done, _ = env.step(0)
    tcs = np.array(tcs)
    assert abs(tcs.mean()) < 0.1
    assert tcs.std() > 0.5          # and it genuinely varies


def test_count_state_adds_the_bin():
    plain = FiniteBlackjackEnv(make_rng(6), with_count_state=False)
    counted = FiniteBlackjackEnv(make_rng(6), with_count_state=True)
    s1, _ = plain.reset()
    s2, _ = counted.reset()
    assert len(s1) == 3 and len(s2) == 4
    assert 0 <= s2[3] < ct.N_BINS


def test_pre_deal_count_is_zero_when_a_reshuffle_is_due():
    """A bettor must size on 0, not on the stale count, once the shoe is spent.

    The player watches the shuffle. Sizing the next bet on the dying shoe's
    count would use information that has just expired -- and because the
    count at the end of a shoe is often extreme, that mistake would show up
    as a fake edge concentrated in exactly the tail bins Kelly bets hardest
    on.
    """
    env = FiniteBlackjackEnv(make_rng(31))
    hit_it = False
    for _ in range(4000):
        if env.shoe.needs_reshuffle():
            hit_it = True
            assert env.pre_deal_true_count() == 0.0
            _, info = env.reset()
            assert info["pre_deal_tc"] == 0.0
        else:
            assert env.pre_deal_true_count() == pytest.approx(env.true_count())
            _, info = env.reset()
        if not info["done"]:
            done = False
            while not done:
                _, _, done, _ = env.step(0)
    assert hit_it, "never reached penetration -- test did not exercise the path"


def test_the_two_environments_agree_on_the_rules():
    """Both environments must score identical hands identically.

    They now share HandPlay, so this checks the extraction did not change
    behaviour -- and it guards the far more important property that the
    exact Phase 1 solution and the Phase 3 measurements are describing the
    same game. If the two environments ever diverged on a rule, the
    cross-check between them would compare two different games and mean
    nothing.

    The shoes differ, so the cards differ; what is compared is the SET of
    outcomes each environment can produce and the aggregate over many hands,
    not hand-by-hand equality.
    """
    from blackjack.env import BlackjackEnv

    def outcomes(env, n):
        seen, total = set(), 0.0
        for _ in range(n):
            _, info = env.reset()
            if info["done"]:
                seen.add(info["reward"]); total += info["reward"]; continue
            done, r = False, 0.0
            while not done:
                out = env.step(1 if env.player_total < 17 else 0)
                r, done = out[1], out[2]
            seen.add(r); total += r
        return seen, total / n

    inf_set, inf_mean = outcomes(BlackjackEnv(make_rng(77)), 40_000)
    fin_set, fin_mean = outcomes(FiniteBlackjackEnv(make_rng(77)), 40_000)

    assert inf_set == fin_set, "the two environments produce different payoffs"
    # Mimic-the-dealer play loses about 5.7% (Phase 1); both must be close.
    assert abs(inf_mean - fin_mean) < 0.01