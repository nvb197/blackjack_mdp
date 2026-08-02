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

Subclasses may override:
    draw_hole()    -> int     deal the dealer's face-down card
    _reveal_hole() -> None    called when the hand ends and it becomes public

THE HOLE CARD IS NOT PUBLIC INFORMATION
----------------------------------------
The dealer's second card is dealt face down. A player -- and therefore any
agent standing in for one -- cannot see it while deciding. In the
infinite-deck environment this is irrelevant, because nothing observable
depends on which cards have left the deck. On a finite shoe it matters a
great deal: if the hole card were folded into the running count at the deal,
the count an agent observes would encode the one card it is not allowed to
know, and any policy learned on that count would be reading through the back
of a card rather than doing arithmetic.

The two hooks below keep that impossible. `draw_hole` takes the card out of
the shoe (it has physically left, so the composition changes) without
publishing it; `_reveal_hole` publishes it when the hand ends. The default
implementations are plain pass-throughs, so the infinite-deck environment
is unaffected.

Modelling note: the hole card is treated as revealed at the end of every
hand, including hands where the player busts. Some casinos return it face
down in that case, which would leave a real counter's count slightly stale.
Modelling that would be more faithful and is not done here.
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

    def draw_hole(self) -> int:
        """Deal the dealer's face-down card. Public information by default."""
        return self.draw()

    def _reveal_hole(self) -> None:
        """The hand has ended; the hole card is now public. No-op by default."""

    # -------------------------------------------------------------- #
    # shared rules
    # -------------------------------------------------------------- #
    def _deal_opening_hand(self) -> dict:
        """Deal two cards each and resolve naturals. Returns the info dict.

        A natural on either side ends the hand before any decision is made,
        which happens on roughly 9% of hands (9.1% measured over 300k deals).
        reset() cannot return a reward -- its signature only returns a state
        -- so the outcome is reported through info, and a learning agent must
        skip these hands entirely, because the reward belongs to no action.
        """
        p1, p2 = self.draw(), self.draw()
        self.dealer_upcard = self.draw()
        self.dealer_hole = self.draw_hole()   # out of the shoe, not yet public
        self.player_total, self.player_soft = hand_value([p1, p2])
        self.first_decision = True
        self._done = False

        dealer_total, _ = hand_value([self.dealer_upcard, self.dealer_hole])
        player_bj = self.player_total == 21 and self.player_soft
        dealer_bj = dealer_total == 21

        info = {"done": False, "reward": 0.0}
        if player_bj or (DEALER_PEEKS and dealer_bj):
            self._done = True
            self._reveal_hole()
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
            reward = self._showdown()
            self._reveal_hole()
            return self._state(), reward, True, {}

        self.player_total, self.player_soft = add_card(
            self.player_total, self.player_soft, self.draw())
        bust = self.player_total > 21

        if action == DOUBLE:
            self._done = True
            reward = -2.0 if bust else self._showdown(bet=2.0)
            self._reveal_hole()
            return self._state(), reward, True, {}

        if bust:
            self._done = True
            self._reveal_hole()
            return self._state(), -1.0, True, {}
        return self._state(), 0.0, False, {}