import numpy as np
import pytest

from blackjack import dp
from blackjack.rules import STAND, HIT, make_rng
from blackjack.env import BlackjackEnv


@pytest.fixture(scope="module")
def solution():
    return dp.value_iteration()


def test_dealer_distribution_is_a_distribution():
    for total in range(4, 22):
        for soft in (False, True):
            d = dp.dealer_distribution(total, soft)
            assert sum(d) == pytest.approx(1.0)
            assert all(p >= 0 for p in d)


def test_dealer_standing_hands_are_deterministic():
    d = dp.dealer_distribution(20, False)
    assert d[3] == pytest.approx(1.0)


def test_dealer_bust_probabilities_match_published_tables():
    """Bust probability given the upcard, after conditioning on no natural."""
    six = dp.dealer_distribution_from_upcard(6)[dp.BUST_IDX]
    ten = dp.dealer_distribution_from_upcard(10)[dp.BUST_IDX]
    assert six == pytest.approx(0.4232, abs=5e-4)
    assert ten == pytest.approx(0.2298, abs=5e-4)
    assert six > ten  # the classic reason six is the dealer's worst card


def test_peek_conditioning_changes_the_answer():
    """Renormalising for the peek must matter, or it is not being applied."""
    raw = np.zeros(6)
    for hole in range(1, 11):
        t, s = dp.hand_value([10, hole])
        raw += dp.CARD_PROBS[hole] * np.asarray(dp.dealer_distribution(t, s))
    conditioned = dp.dealer_distribution_from_upcard(10)
    assert not np.allclose(raw, conditioned, atol=1e-3)


def test_stand_values_are_bounded():
    sv = dp.stand_values()[4:22]
    assert sv.min() >= -1.0 and sv.max() <= 1.0


def test_standing_on_21_beats_standing_on_16():
    sv = dp.stand_values()
    for up in range(10):
        assert sv[21, up] > sv[16, up]


def test_value_iteration_converges_quickly(solution):
    _, _, sweeps = solution
    assert sweeps < 20


def test_optimal_values_at_reference_states(solution):
    V, _, _ = solution
    assert V[20, 5, 0] == pytest.approx(0.7040, abs=1e-3)
    assert V[16, 9, 0] == pytest.approx(-0.5398, abs=1e-3)


def test_values_are_bounded(solution):
    V, _, _ = solution
    assert np.all(V[4:22] >= -1.0) and np.all(V[4:22] <= 1.0)


def test_hitting_is_optimal_below_twelve(solution):
    _, pi, _ = solution
    assert np.all(pi[4:12, :, 0] == HIT)


def test_standing_is_optimal_on_hard_seventeen_plus(solution):
    _, pi, _ = solution
    assert np.all(pi[17:22, :, 0] == STAND)


def test_expected_values_match_published_figures(solution):
    V, _, _ = solution
    assert dp.expected_value(V, allow_double=False) == pytest.approx(
        -0.02421, abs=5e-4)
    assert dp.expected_value(V, allow_double=True) == pytest.approx(
        -0.01087, abs=5e-4)


def test_doubling_can_only_help(solution):
    V, _, _ = solution
    assert (dp.expected_value(V, allow_double=True)
            > dp.expected_value(V, allow_double=False))


def test_policy_evaluation_of_the_optimal_policy_recovers_v_star(solution):
    V, pi, _ = solution
    assert np.allclose(dp.policy_evaluation(pi)[4:22], V[4:22], atol=1e-7)


def test_a_worse_policy_has_a_worse_value(solution):
    V, pi, _ = solution
    always_stand = np.zeros_like(pi)
    V_bad = dp.policy_evaluation(always_stand)
    assert dp.expected_value(V_bad) < dp.expected_value(V)


def test_simulation_agrees_with_the_exact_solution(solution):
    """Play the optimal policy for 300k hands and compare with V*.

    This is the cross-check that matters: the solver computes transition
    probabilities by hand and never simulates, while the environment
    simulates and never uses those probabilities. Agreement within a few
    standard errors means an error would have to appear in both, the same
    way, independently.
    """
    V, pi, _ = solution
    env = BlackjackEnv(make_rng(123))
    n = 300_000
    total = 0.0
    for _ in range(n):
        (t, up, soft), info = env.reset()
        if info["done"]:
            total += info["reward"]
            continue
        done = False
        while not done:
            a = int(pi[t, up - 1, int(soft)])
            (t, up, soft), r, done, _ = env.step(a)
        total += r

    mean = total / n
    exact = dp.expected_value(V, allow_double=False)
    se = 1.14 / np.sqrt(n)  # per-hand standard deviation is close to 1.14
    assert abs(mean - exact) < 3 * se
