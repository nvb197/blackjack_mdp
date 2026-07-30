"""Card distribution and hand arithmetic.

House rules are fixed here and nowhere else:

* Infinite deck -- cards are drawn with replacement, so P(10-value) = 4/13
  (ten, jack, queen and king are a single card class).
* Dealer stands on all 17 (S17).
* Dealer peeks for blackjack when showing an ace or a ten.
* Natural blackjack pays 3:2.
* Double is allowed on the first decision only. No splitting.
"""

import numpy as np

DEALER_HITS_SOFT_17 = False
DEALER_PEEKS = True
BLACKJACK_PAYOUT = 1.5

# CARD_PROBS[c] = P(drawing a card of value c). Index 0 is unused.
CARD_PROBS = np.array([0.0] + [1 / 13] * 9 + [4 / 13])
CARD_CDF = np.cumsum(CARD_PROBS)

STAND, HIT, DOUBLE = 0, 1, 2


def add_card(total: int, usable_ace: bool, card: int) -> tuple[int, bool]:
    """Add one card to a hand and return the new (total, usable_ace).

    An ace counts as 11 whenever that does not bust the hand. If the total
    then exceeds 21 and the hand still holds an ace valued at 11, that ace
    is demoted to 1 and the total drops by 10.

    Three cases worth testing, because they are where this function usually
    goes wrong:  soft 17 + 5 -> hard 12 (not a bust);  A + A -> soft 12;
    soft 21 + 10 -> hard 21.
    """
    if card == 1 and total + 11 <= 21:
        total += 11
        usable_ace = True
    else:
        total += card
    if total > 21 and usable_ace:
        total -= 10
        usable_ace = False
    return total, usable_ace


def hand_value(cards: list[int]) -> tuple[int, bool]:
    """Return (total, usable_ace) for a list of card values."""
    total, soft = 0, False
    for c in cards:
        total, soft = add_card(total, soft, c)
    return total, soft


def make_rng(seed: int = 42) -> np.random.Generator:
    """Return an independent PCG64 generator.

    Every random source in this project is created here and passed
    explicitly, so a run is reproducible from its seed alone. Nothing
    reads or writes the global numpy random state.
    """
    return np.random.default_rng(seed)
