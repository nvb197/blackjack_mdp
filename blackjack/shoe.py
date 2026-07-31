"""A finite six-deck shoe: cards run out, and that is the entire point.

READ THIS FIRST if the word "shoe" is new to you.

In real blackjack the casino does not draw one card and put it back. Several
decks are shuffled together into a "shoe" (6 decks = 312 cards here), and
cards are dealt from it one at a time without replacement until the shoe is
mostly used up, at which point it is reshuffled.

Why this matters mathematically: in blackjack/env.py (the infinite-deck
version from Phase 1-2), P(next card is a ten) is ALWAYS 4/13, no matter what
happened before. That is what made the exact DP solution possible -- the
probabilities never change, so the problem is *stationary*.

With a real shoe, that stops being true. If the last 20 cards dealt were
mostly low cards (2-6), then the cards *left in the shoe* are disproportion-
ately high cards (10s and aces), so your next card is MORE likely to be a
ten than 4/13. The composition of the shoe drifts as it empties, and that
drift is the entire reason card counting can work: a player who tracks which
cards have come out can tell, at any moment, whether the remaining shoe is
better or worse for them than average.

That is also why this cannot be solved exactly like Phase 1-2 was. The state
would need to include the exact remaining composition of the shoe -- a
10-number vector -- and the number of distinct shoe compositions is around
3.7 * 10^14, which multiplied by the 200 player states gives about
7.4 * 10^16. Value iteration over that is not merely slow: storing one
number per state at eight bytes would need roughly 590 petabytes.

So Phase 3 does not try to solve the game exactly. It measures, empirically,
how a summary statistic of the shoe (the "true count", built in counting.py)
relates to your edge.

WHAT THIS FILE PROVIDES
-----------------------
A `Shoe` class that:
  1. Holds 6 decks (312 cards) in a shuffled array.
  2. Deals cards from it one at a time, without replacement.
  3. Reshuffles once "penetration" (75%) of the shoe has been dealt.
  4. Tracks the composition of the remaining shoe as a byproduct, so that
     counting.py and dp.py (Phase 4 cross-check) can use it.

DESIGN CHOICE, AND WHY
-----------------------
The shoe is represented as an actual shuffled array of 312 integers (card
ranks), read with a moving pointer -- NOT as an abstract "10 numbers
remaining" vector that you draw from directly.

Why: with an explicit array, running out of a rank is *structurally
impossible* -- you simply cannot deal a card that was never in the array.
With a probability-vector approach, you would have to remember, everywhere,
to check "is this rank still available?" before drawing from it, and
forgetting that check even once is a silent bug: the code keeps running, it
just occasionally deals a card that should not exist. Prefer designs where
a bug is impossible over designs where a bug is merely detectable.

A `counts` array is ALSO kept (10 numbers, ranks 1-10) as a derived
quantity, updated every time a card is dealt. It is not the source of truth
-- the array is -- but counting.py needs an O(1) way to ask "how many of
each rank are left", and recomputing that from the array on every draw would
be wasteful. Because `counts` is derived, you can (and the tests will) check
that it always agrees with what's actually left in the array.

CARD ENCODING
-------------
Cards are encoded 1-10 exactly as in rules.py: 1 = ace, 2..9 literal, and
10 stands for the whole ten-value class (10, J, Q, K). A single 6-deck shoe
therefore contains:
    24 cards of each rank 1..9   (6 decks x 4 suits)
    96 cards of rank 10          (6 decks x 4 suits x 4 ten-cards: 10,J,Q,K)
    total = 24*9 + 96 = 312
"""

import numpy as np

N_DECKS = 6
PENETRATION = 0.75          # reshuffle once this fraction of the shoe is dealt
CARDS_PER_RANK = np.array([0, 24, 24, 24, 24, 24, 24, 24, 24, 24, 96])
# index:                    0   1   2   3   4   5   6   7   8   9   10
SHOE_SIZE = int(CARDS_PER_RANK.sum())   # 312


class Shoe:
    """A 6-deck shoe dealt without replacement, reshuffled at 75% penetration."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.cards: np.ndarray = np.empty(0, dtype=np.int64)
        self.pos: int = 0
        self.counts: np.ndarray = np.zeros(11, dtype=np.int64)
        self._shuffle()

    def _shuffle(self) -> None:
        """Build a freshly shuffled 312-card shoe and reset the pointer.

        `np.repeat(np.arange(11), CARDS_PER_RANK)` produces the multiset:
        0 appears CARDS_PER_RANK[0] = 0 times, rank 1 appears 24 times, ...,
        rank 10 appears 96 times. Shuffling is done through self.rng and
        never through np.random.*, so a run is reproducible from its seed.

        Test that will check this: test_fresh_shoe_has_correct_composition
        """
        cards = np.repeat(np.arange(11), CARDS_PER_RANK)
        self.cards = self.rng.permutation(cards)
        self.pos = 0
        self.counts = CARDS_PER_RANK.copy()

    def cards_dealt(self) -> int:
        """How many cards have been dealt since the last shuffle."""
        return self.pos

    def cards_remaining(self) -> int:
        return SHOE_SIZE - self.pos

    def needs_reshuffle(self) -> bool:
        """True once penetration has been reached.

        The comparison is cross-multiplied (cards_dealt() >= PENETRATION *
        SHOE_SIZE) rather than divided, to keep the boundary exact.

        IMPORTANT DESIGN NOTE: this method only
        REPORTS whether a reshuffle is due. It does not reshuffle. The
        actual reshuffle happens in maybe_reshuffle(), called at the START
        of a new hand (see env.py's reset()), never in the middle of one.
        Real casinos do not reshuffle mid-hand, and if you did, you'd
        silently change the composition the player is relying on partway
        through a decision -- a subtle form of look-ahead bias.
        """
        return self.pos >= PENETRATION * SHOE_SIZE

    def maybe_reshuffle(self) -> bool:
        """Reshuffle if due. Call this ONLY between hands. Returns whether it fired."""
        if self.needs_reshuffle():
            self._shuffle()
            return True
        return False

    def draw(self) -> int:
        """Deal the next card from the shoe. Returns a rank 1-10.

        The assertion should be unreachable: penetration stops dealing with
        78 cards to spare. It costs nothing and pins the invariant anyway.
        """
        assert self.pos < SHOE_SIZE, "shoe is empty -- reshuffle was missed"
        card = self.cards[self.pos]
        self.pos += 1
        self.counts[card] -= 1
        return int(card)

    def probs(self) -> np.ndarray:
        """Current P(next card = rank) given what's left in the shoe, shape (11,).

        This is the shoe's analogue of rules.CARD_PROBS -- except it CHANGES
        as the shoe empties, which is the whole point of Phase 3.

            probs()[r] = counts[r] / cards_remaining()

        Index 0 is zero and unused, matching the convention in rules.py.
        """
        p = np.zeros(11)
        p[1:11] = self.counts[1:11] / self.cards_remaining()
        return p