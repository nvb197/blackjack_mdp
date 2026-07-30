import numpy as np
import pytest

from blackjack.rules import (CARD_PROBS, STAND, HIT, DOUBLE,
                             add_card, hand_value, make_rng)
from blackjack.env import BlackjackEnv


def test_card_probabilities_sum_to_one():
    assert CARD_PROBS.sum() == pytest.approx(1.0)


def test_ten_class_has_four_times_the_mass():
    assert CARD_PROBS[10] == pytest.approx(4 / 13)
    assert CARD_PROBS[1] == pytest.approx(1 / 13)


@pytest.mark.parametrize("total,soft,card,expected", [
    (17, True, 5, (12, False)),    # soft 17 + 5 demotes the ace
    (11, False, 1, (12, False)),   # ace taken as 1, since 11 would bust
    (21, True, 10, (21, False)),   # soft 21 + 10 stays at 21
    (20, False, 5, (25, False)),   # plain bust
    (0, False, 1, (11, True)),
])
def test_add_card(total, soft, card, expected):
    assert add_card(total, soft, card) == expected


def test_two_aces_is_soft_twelve():
    assert hand_value([1, 1]) == (12, True)


def test_three_aces_is_soft_thirteen():
    assert hand_value([1, 1, 1]) == (13, True)


def test_draw_matches_the_card_distribution():
    """Chi-square goodness of fit for the batched inverse-transform sampler."""
    env = BlackjackEnv(make_rng(0))
    n = 200_000
    counts = np.bincount([env.draw() for _ in range(n)], minlength=11)[1:]
    expected = CARD_PROBS[1:] * n
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert chi2 < 27.9  # 0.999 quantile, 9 degrees of freedom


def test_hand_terminates_and_reward_is_bounded():
    env = BlackjackEnv(make_rng(1), allow_double=True)
    for _ in range(2000):
        _, info = env.reset()
        if info["done"]:
            assert -1.0 <= info["reward"] <= 1.5
            continue
        steps, done, reward = 0, False, 0.0
        while not done:
            action = HIT if steps < 1 else STAND
            _, reward, done, _ = env.step(action)
            steps += 1
            assert steps < 25
        assert -2.0 <= reward <= 2.0


def test_double_is_rejected_after_the_first_decision():
    env = BlackjackEnv(make_rng(2), allow_double=True)
    while True:
        _, info = env.reset()
        if not info["done"]:
            break
    _, _, done, _ = env.step(HIT)
    if not done:
        with pytest.raises(AssertionError):
            env.step(DOUBLE)


def test_same_seed_gives_identical_hands():
    a = [BlackjackEnv(make_rng(7)).draw() for _ in range(100)]
    b = [BlackjackEnv(make_rng(7)).draw() for _ in range(100)]
    assert a == b
