"""Blackjack against a finite six-deck shoe, with the count exposed safely.

This is the Phase 3 counterpart of env.py. The rules of play are identical --
same dealer logic, same peek, same payoffs -- but cards come from shoe.py
instead of being drawn with replacement, so the composition drifts and the
count carries information.

WHAT CHANGES RELATIVE TO env.py
--------------------------------
1. Cards come from a Shoe. They run out and get reshuffled.
2. A Hi-Lo running count is maintained incrementally as cards appear.
3. reset() captures the true count BEFORE dealing and returns it in
   info["pre_deal_tc"]. This is the only count a bettor is allowed to use.
4. The state returned to a learning agent optionally includes the true
   count bin, so a policy can be count-dependent ("index plays").

THE ONE THING TO GET RIGHT
---------------------------
Line ordering in reset(). The count snapshot happens FIRST, before a single
card of the new hand is dealt:

    pre_deal_tc = self.pre_deal_true_count()   # <- before anything
    self.shoe.maybe_reshuffle()         # <- between hands only
    ... deal ...

If those lines were reordered, every downstream number in Phase 3 and 4
would be contaminated by look-ahead bias, and nothing would visibly break.
See counting.py's module docstring for why that is the most dangerous bug
in this project.

Note the reshuffle also happens here, between hands -- never mid-hand. A
mid-hand reshuffle would change the composition a player is reasoning about
partway through their decision.
"""

import numpy as np

from .counting import HILO, tc_bin, true_count
from .hand import HandPlay
from .shoe import Shoe

State = tuple[int, int, bool]
CountState = tuple[int, int, bool, int]


class FiniteBlackjackEnv(HandPlay):
    """Six-deck blackjack. Actions: 0 stand, 1 hit, 2 double."""

    def __init__(self, rng: np.random.Generator, allow_double: bool = False,
                 with_count_state: bool = False):
        self.shoe = Shoe(rng)
        self.allow_double = allow_double
        self.with_count_state = with_count_state
        self.running = 0
        self._pending_hole: int | None = None
        self._done = True

    # ------------------------------------------------------------------ #
    # counting
    # ------------------------------------------------------------------ #
    def true_count(self) -> float:
        return true_count(self.running, self.shoe.cards_remaining())

    def pre_deal_true_count(self) -> float:
        """The count a bettor sees before this hand, accounting for a due shuffle.

        A player at the table watches the shuffle happen, so once the shoe
        has reached penetration they know the count is about to be zero and
        must size the next bet on zero -- not on the stale count from the
        exhausted shoe. Calling true_count() directly here would use
        information that has just expired.

        This is the value reset() reports as info["pre_deal_tc"], exposed
        separately so a bettor can obtain it BEFORE the hand starts, which
        is when a bet must be placed.
        """
        if self.shoe.needs_reshuffle():
            return 0.0
        return self.true_count()

    def draw(self) -> int:
        """Deal one card and fold it into the running count.

        Doing both in one place is deliberate: a card that is dealt but not
        counted (or counted twice) silently corrupts the true count, and
        there is no test that can distinguish a slightly-wrong count from a
        correct one without recomputing from scratch -- which
        test_running_count_matches_recomputation does exactly to guard this.
        """
        c = self.shoe.draw()
        self.running += int(HILO[c])
        return c

    def draw_hole(self) -> int:
        """Take the dealer's face-down card out of the shoe without counting it.

        The card has physically left the shoe, so `cards_remaining` drops and
        the composition changes. But it is face down, so it must not enter the
        running count until the hand ends -- otherwise the count an agent
        observes encodes the one card a real player cannot see, and Phase 3b
        would be learning to read the hole card rather than to count.
        """
        c = self.shoe.draw()
        self._pending_hole = c
        return c

    def _reveal_hole(self) -> None:
        """Fold the hole card into the running count once the hand is over."""
        if self._pending_hole is not None:
            self.running += int(HILO[self._pending_hole])
            self._pending_hole = None

    def _state(self):
        if self.with_count_state:
            return (self.player_total, self.dealer_upcard, self.player_soft,
                    tc_bin(self.true_count()))
        return (self.player_total, self.dealer_upcard, self.player_soft)

    # ------------------------------------------------------------------ #
    # hand lifecycle
    # ------------------------------------------------------------------ #
    def reset(self):
        """Start a new hand. See the module docstring on line ordering.

        The first act is to publish any hole card still pending. Normally the
        previous hand ended and published its own, but a caller can abandon a
        hand -- call reset() twice without playing the first out. Without this
        line that card would leave the shoe and never enter the count, and the
        running count would drift silently away from the cards actually dealt.
        By the time a new hand begins, every card of the previous one is
        public, so this is also the correct moment semantically.
        """
        self._reveal_hole()
        pre_deal_tc = self.pre_deal_true_count()  # BEFORE anything is dealt
        if self.shoe.maybe_reshuffle():           # between hands only
            self.running = 0
        info = self._deal_opening_hand()
        info["pre_deal_tc"] = pre_deal_tc
        return self._state(), info