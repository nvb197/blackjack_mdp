"""Card counting: turning a 10-number shoe composition into one usable number.

READ THIS FIRST if card counting is new to you.

Phase 1-2 showed that with an infinite deck, perfect play still loses 1.087%
per hand. That number is fixed -- every hand is identical, so there is
nothing to react to.

With a finite shoe (shoe.py), that stops being true. As cards leave the shoe,
what remains gets richer or poorer in the cards that favour you. Concretely:

  * HIGH cards (tens and aces) favour the PLAYER. More tens and aces left
    means more natural blackjacks -- and a natural pays you 3:2 but only
    pays the dealer 1:1, so that asymmetry is worth money to you. It also
    means the dealer, who is forced to draw to 17, busts more often.

  * LOW cards (2-6) favour the DEALER. With small cards left, the dealer can
    creep up to 17-21 safely instead of busting.

So if lots of low cards have already been dealt, the remaining shoe is
high-rich, and your expected value goes UP. Sometimes above zero.

THE PROBLEM: the shoe's exact state is a 10-number vector, and there are
about 3.7 * 10^14 possible values of it. No human can track that at a table,
and no computer can solve a decision problem over it (see shoe.py's note).

THE SOLUTION (Thorp 1962, refined into Hi-Lo): compress those 10 numbers
into ONE number that captures most of what matters.

    Hi-Lo card values:   2,3,4,5,6  ->  +1     (low cards, gone = good)
                         7,8,9      ->   0     (neutral)
                         10, A      ->  -1     (high cards, gone = bad)

Keep a running total ("running count", RC) of these as cards appear. RC > 0
means more low cards than high cards have been dealt, so the shoe is now
high-rich: good for you.

WHY DIVIDE BY DECKS REMAINING
------------------------------
A running count of +6 means very different things at different points in the
shoe. If 5 decks are left, a surplus of 6 low-cards-gone is spread thin --
barely noticeable. If half a deck is left, that same surplus is concentrated,
and the next card is dramatically more likely to be high.

    true count (TC) = running count / decks remaining

This is the single most important idea in counting, and it is just
normalisation: you care about the CONCENTRATION of the surplus, not its
absolute size. Same reason you quote a solution's concentration rather than
"how much salt I added".

THIS IS A LOSSY COMPRESSION -- SAY SO
--------------------------------------
Hi-Lo throws information away. It cannot distinguish a shoe missing five 2s
from one missing five 6s, though those are not equally good for you (this is
called "effect of removal", and different counting systems weight ranks
differently to capture it better). So the true count is NOT a sufficient
statistic for the shoe -- it is an *approximately* sufficient one. Being
precise about this is the difference between understanding the method and
reciting it.

THE DANGEROUS BUG THIS FILE IS DESIGNED TO PREVENT
---------------------------------------------------
You place your bet BEFORE any cards of the hand are dealt. So the only count
you are allowed to use for sizing is the count as it stood BEFORE the deal.

If you instead read the count after the hand's cards are out -- and then
attribute the hand's outcome to that count -- you are using information that
did not exist when the bet was placed. That is look-ahead bias, exactly the
same error as a backtest that peeks at tomorrow's prices.

It will not crash. It will not raise an exception. It will produce a
beautiful, smoothly increasing edge-versus-count curve that is completely
fictitious. This is the single most dangerous thing in Phase 3.

The defence here is structural, not disciplinary: the environment captures
the count as the very first statement of reset(), before dealing anything,
and hands it back in info["pre_deal_tc"]. Callers cannot get it wrong,
because they never compute it themselves.
"""

import numpy as np

# Hi-Lo value of each rank. Index 0 unused, matching rules.py.
#              0   A   2   3   4   5   6   7   8   9  10
HILO = np.array([0, -1, +1, +1, +1, +1, +1, 0, 0, 0, -1])

CARDS_PER_DECK = 52

# Bin edges for the true count. Bin i covers [BIN_EDGES[i-1], BIN_EDGES[i]),
# with the two tails open. 8 bins total.
BIN_EDGES = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
N_BINS = len(BIN_EDGES) + 1          # 8
BIN_LABELS = ["<-3", "-3..-2", "-2..-1", "-1..0",
              "0..1", "1..2", "2..3", ">=3"]

# Payoff values a single hand can take, in units of the base bet. Used as the
# support of the histogram in PreDealTracker.
#   -2, +2  : doubled hands
#   -1, +1  : ordinary loss / win
#    0      : push
#   +1.5    : natural blackjack, paid 3:2
PAYOFFS = np.array([-2.0, -1.0, 0.0, 1.0, 1.5, 2.0])
PAYOFF_INDEX = {v: i for i, v in enumerate(PAYOFFS)}


def running_count(cards_seen: np.ndarray) -> int:
    """Hi-Lo running count of a sequence of ranks. Mostly for testing.

    In the simulation the count is maintained incrementally (add HILO[card]
    on every draw) rather than recomputed, but having this available makes
    the incremental version testable against a straightforward definition.
    """
    return int(HILO[np.asarray(cards_seen, dtype=np.int64)].sum())


def decks_remaining(cards_remaining: int) -> float:
    """Cards left, expressed in decks. Never returns 0 (guards the division)."""
    return max(cards_remaining / CARDS_PER_DECK, 1e-9)


def true_count(running: int, cards_remaining: int) -> float:
    """running count normalised by decks remaining -- see the module docstring."""
    return running / decks_remaining(cards_remaining)


def tc_bin(tc: float) -> int:
    """Index 0..7 of the bin containing this true count.

    np.searchsorted with side="right" gives exactly the half-open
    convention described at BIN_EDGES: a true count of exactly 1.0 lands in
    bin "1..2", not "0..1".

    On the bias-variance tradeoff in the bin width: narrower bins resolve
    the edge-versus-count relationship more finely but put fewer hands in
    each bin, so each estimate is noisier (higher variance). Wider bins pool
    counts that genuinely differ in value, so each estimate is biased toward
    the bin average. Width 1.0 is chosen here because the published
    literature quotes edge in units of "per true count", making the results
    directly comparable; the tails are pooled because |TC| >= 3 is rare
    enough that finer tail bins would have unusable sample sizes.
    """
    return int(np.searchsorted(BIN_EDGES, tc, side="right"))


class PreDealTracker:
    """Histogram of hand payoffs, keyed on the true count BEFORE the deal.

    Why a full histogram instead of just (wins, losses, pushes): Kelly
    sizing needs E[X^2], not just the mean, and E[X^2] depends on the whole
    payoff distribution. A hand that wins 1.5 (natural) and one that wins
    2.0 (doubled) are both "a win" but contribute very differently to
    variance, and therefore to the correct bet size. Collapsing them loses
    exactly the information sizing depends on.

    Storage is a (N_BINS, len(PAYOFFS)) integer array of counts -- compact,
    exact, and enough to reconstruct mean, variance and any quantile.
    """

    def __init__(self) -> None:
        self.hist = np.zeros((N_BINS, len(PAYOFFS)), dtype=np.int64)

    def record(self, pre_deal_tc: float, payoff: float) -> None:
        """Record one completed hand under the count that preceded it."""
        b = tc_bin(pre_deal_tc)
        self.hist[b, PAYOFF_INDEX[float(payoff)]] += 1

    def n(self, b: int) -> int:
        return int(self.hist[b].sum())

    def total(self) -> int:
        return int(self.hist.sum())

    def stats(self, b: int, z: float = 1.96) -> dict:
        """Mean, standard deviation, E[X^2] and a confidence interval for a bin.

        The interval is the ordinary CLT one, mu_hat +- z * se. Note it is NOT
        a Wilson interval: Wilson is for a binomial proportion, and the payoff
        here takes six distinct values, not two. Reaching for Wilson because
        "it's about win rate" is a common slip.

        THE STANDARD ERROR USES n-1, NOT n
        -----------------------------------
        `ex2 - mean**2` is the POPULATION variance -- it divides by n. The
        unbiased estimate of the variance from a sample divides by n-1, so

            se = sqrt(population_variance / (n - 1))

        which is the same as sample_std / sqrt(n). Using sqrt(n) on the
        population variance understates the standard error by sqrt((n-1)/n):
        0.1% at n = 1000, but 41% at n = 2. Since the significance gate in
        sizing.py compares mu_hat against z*se, understating se at small n is
        exactly the direction that lets noise through.

        Returns n = 0 with everything else nan when the bin is empty, and
        se = inf when n = 1, since one observation carries no information
        about its own spread.
        """
        counts = self.hist[b]
        n = int(counts.sum())
        if n == 0:
            return {"n": 0, "mean": np.nan, "std": np.nan, "ex2": np.nan,
                    "se": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        p = counts / n
        mean = float((p * PAYOFFS).sum())
        ex2 = float((p * PAYOFFS ** 2).sum())
        var = max(ex2 - mean ** 2, 0.0)          # population variance
        std = float(np.sqrt(var))
        se = float(np.sqrt(var / (n - 1))) if n > 1 else float("inf")
        return {"n": n, "mean": mean, "std": std, "ex2": ex2, "se": se,
                "ci_low": mean - z * se, "ci_high": mean + z * se}

    def all_stats(self, z: float = 1.96) -> list[dict]:
        out = []
        for b in range(N_BINS):
            s = self.stats(b, z)
            s["bin"] = b
            s["label"] = BIN_LABELS[b]
            out.append(s)
        return out

    def hands_needed_for_significance(self, edge: float = 0.01,
                                      sigma: float = 1.0,
                                      z: float = 1.96) -> int:
        """How many hands per bin are needed to tell an edge of `edge` from zero.

        Setting z * sigma / sqrt(n) < edge and solving for n gives
        n > (z * sigma / edge)^2. For a 1% edge with sigma ~ 1 that is about
        38,000 hands *per bin*.

        This number is worth internalising: it is why real card counters need
        hundreds of hours at the table before their results mean anything,
        and why a few thousand simulated hands per bin tells you essentially
        nothing. Most of the apparent structure in a small sample is noise.
        """
        return int(np.ceil((z * sigma / edge) ** 2))