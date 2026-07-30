"""Blackjack simulator for a single player against the dealer.

Because cards are drawn with replacement, the card distribution never
changes and the game is a stationary Markov decision process. That is what
makes the exact dynamic-programming solution in :mod:`blackjack.dp` valid,
and it is why this simulator is only needed to check that solution.

State is (player_total, dealer_upcard, usable_ace); rewards are +1 for a
win, -1 for a loss, 0 for a push, +1.5 for a natural, and doubled when the
player doubles. The dynamic-programming solver must use exactly the same
convention or the two will silently disagree.
"""

import numpy as np

from .hand import HandPlay
from .rules import CARD_CDF

State = tuple[int, int, bool]


class BlackjackEnv(HandPlay):
    """Infinite-deck blackjack. Actions: 0 stand, 1 hit, 2 double."""

    _BUF_SIZE = 1 << 16

    def __init__(self, rng: np.random.Generator, allow_double: bool = False):
        self.rng = rng
        self.allow_double = allow_double
        self._done = True
        self._buf = np.empty(0, dtype=np.int64)
        self._buf_i = 0

    def draw(self) -> int:
        """Draw one card by inverse-transform sampling from a prefilled batch.

        Drawing 65k uniforms at a time and inverting the CDF gives the same
        distribution as ``rng.choice`` but avoids revalidating the
        probability vector on every call, which matters over millions of
        hands (it cut the test suite from 24s to 3s).
        """
        if self._buf_i >= self._buf.size:
            u = self.rng.random(self._BUF_SIZE)
            self._buf = np.searchsorted(CARD_CDF, u, side="right")
            self._buf_i = 0
        c = self._buf[self._buf_i]
        self._buf_i += 1
        return int(c)

    def _state(self) -> State:
        return (self.player_total, self.dealer_upcard, self.player_soft)

    def reset(self) -> tuple[State, dict]:
        """Deal a new hand. See HandPlay._deal_opening_hand for naturals."""
        info = self._deal_opening_hand()
        return self._state(), info


