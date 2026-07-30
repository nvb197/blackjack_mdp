"""Hand-playing logic shared by both environments.

WHY THIS FILE EXISTS
--------------------
env.py (infinite deck) and finite_env.py (six-deck shoe) differ in exactly
one thing: where a card comes from. Everything else -- how the dealer plays,
how a showdown is scored, what a hit or a double does, how naturals are
resolved -- is identical, because those are the rules of blackjack and the
rules do not care how the deck is modelled.

Before this refactor, step() was 19 of 20 lines identical between the two
files. That is the kind of duplication that goes wrong slowly: a rule gets
fixed in one file and not the other, and the two environments quietly stop
agreeing, which would silently break the cross-check between Phase 1's exact
solution and Phase 3's measurements.

A NOTE ON WHEN A BASE CLASS IS WORTH IT
----------------------------------------
Phase 1 of this project deliberately DELETED an abstract base class, because
there was only one environment and the abstraction was pure indirection --
a layer that cost a file and bought nothing.

Now there are genuinely two implementations that share substantial behaviour,
so the base class earns its place. The rule of thumb worth taking away:
abstract when you have two real cases, not when you imagine you might.

Subclasses must provide:
    draw()    -> int          deal one card
    _state()  -> tuple        the observation handed to an agent
"""

from .rules import (
    BLACKJACK_PAYOUT,
    DEALER_HITS_SOFT_17,
    DEALER_PEEKS,
    DOUBLE,
    STAND,
    add_card,
    hand_value,
)


class HandPlay:
    """The rules of a blackjack hand, independent of how cards are supplied."""

    allow_double: bool

    # -------------------------------------------------------------- #
    # to be supplied by the subclass
    # -------------------------------------------------------------- #
    def draw(self) -> int:
        raise NotImplementedError

    def _state(self):
        raise NotImplementedError

    # -------------------------------------------------------------- #
    # shared rules
    # -------------------------------------------------------------- #
    def _deal_opening_hand(self) -> dict:
        """Deal two cards each and resolve naturals. Returns the info dict.

        A natural on either side ends the hand before any decision is made,
        which happens on roughly 8% of hands. reset() cannot return a reward
        (its signature only returns a state), so the outcome is reported
        through info -- and a learning agent must skip these hands entirely,
        because the reward belongs to no action.
        """
        p1, p2 = self.draw(), self.draw()
        self.dealer_upcard = self.draw()
        self.dealer_hole = self.draw()
        self.player_total, self.player_soft = hand_value([p1, p2])
        self.first_decision = True
        self._done = False

        dealer_total, _ = hand_value([self.dealer_upcard, self.dealer_hole])
        player_bj = self.player_total == 21 and self.player_soft
        dealer_bj = dealer_total == 21

        info = {"done": False, "reward": 0.0}
        if player_bj or (DEALER_PEEKS and dealer_bj):
            self._done = True
            info["done"] = True
            if player_bj and dealer_bj:
                info["reward"] = 0.0
            elif player_bj:
                info["reward"] = BLACKJACK_PAYOUT
            else:
                info["reward"] = -1.0
        return info

    def _dealer_play(self) -> int:
        """Draw for the dealer until standing. The dealer has no choices."""
        total, soft = hand_value([self.dealer_upcard, self.dealer_hole])
        while total < 17 or (soft and total == 17 and DEALER_HITS_SOFT_17):
            total, soft = add_card(total, soft, self.draw())
        return total

    def _showdown(self, bet: float = 1.0) -> float:
        d = self._dealer_play()
        if d > 21 or self.player_total > d:
            return bet
        if self.player_total < d:
            return -bet
        return 0.0

    def step(self, action: int):
        """Apply one action. Actions: 0 stand, 1 hit, 2 double."""
        assert not self._done, "hand is over -- call reset() first"
        if action == DOUBLE:
            assert self.allow_double and self.first_decision, \
                "double is only legal on the first decision"
        self.first_decision = False

        if action == STAND:
            self._done = True
            return self._state(), self._showdown(), True, {}

        self.player_total, self.player_soft = add_card(
            self.player_total, self.player_soft, self.draw())
        bust = self.player_total > 21

        if action == DOUBLE:
            self._done = True
            reward = -2.0 if bust else self._showdown(bet=2.0)
            return self._state(), reward, True, {}

        if bust:
            self._done = True
            return self._state(), -1.0, True, {}
        return self._state(), 0.0, False, {}
