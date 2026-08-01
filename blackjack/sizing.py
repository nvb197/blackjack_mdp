"""How much to bet: the Kelly criterion, and why you should bet less than it says.

READ THIS FIRST if Kelly is new to you.

Phase 3 established that at a high true count your edge can be positive. So:
how much of your bankroll should you stake?

Two obvious answers are both wrong.

"Bet everything when the edge is positive" is wrong because you will
eventually lose a hand and be at zero, from which no future edge can help
you. Ruin is absorbing.

"Bet a fixed small amount" is wrong in the other direction: it leaves growth
on the table when the edge is large.

Kelly (1956) asks a sharper question: what fraction f of your bankroll
should you stake on each bet to maximise the long-run GROWTH RATE of the
bankroll? Not the expected profit -- the expected *logarithm*, which is what
compounds.

WHY LOGARITHM
-------------
Because gains and losses compound multiplicatively. Lose 50% then gain 50%
and you are at 0.75, not 1.0. Maximising expected profit each round ignores
this and leads to overbetting; maximising expected log-wealth accounts for
it exactly. A useful way to see it: log turns products into sums, and the
long-run behaviour of a product of random factors is governed by the sum of
their logs (law of large numbers).

THE DERIVATION
--------------
Let X be the payoff per unit staked (for blackjack, X takes values -2, -1,
0, +1, +1.5, +2). Stake fraction f of bankroll B. After one hand:

    B_new = B * (1 + f * X)

Growth rate to maximise:

    g(f) = E[log(1 + f X)]

Taylor-expand log(1 + u) ~ u - u^2/2 for small u:

    g(f) ~ f * E[X] - (1/2) f^2 * E[X^2]
         = f * mu   - (1/2) f^2 * E[X^2]

Differentiate and set to zero:

    g'(f) = mu - f * E[X^2] = 0   =>   f* = mu / E[X^2]

THIS IS A SECOND-ORDER APPROXIMATION, NOT THE EXACT OPTIMUM
-----------------------------------------------------------
Be precise about what f* = mu / E[X^2] is. It comes from truncating the
Taylor expansion of log at second order, so it is the growth-optimal fraction
of a quadratic approximation to g(f) -- not the exact maximiser of
E[log(1 + fX)]. The exact answer has no closed form for a six-outcome payoff
and must be found numerically.

Measured on the >=3 count bin of this project (mu = 0.00890, E[X^2] =
1.19474): the Taylor value is f* = 0.00745 against an exact numerical
optimum of 0.00746 -- the approximation UNDER-bets by 0.1%.

The direction is worth understanding rather than memorising. The next Taylor
term is +f^3 E[X^3]/3, and blackjack's payoff distribution is positively
skewed here (E[X^3] = +0.213, driven by the +1.5 natural and the +2 double),
so including it raises the optimum slightly. A negatively skewed payoff would
push the other way, and the second-order formula would then over-bet. Saying
"the approximation over-bets on skewed distributions" without checking the
sign of the third moment gets it backwards half the time.

At 0.1% the error is irrelevant next to the estimation error in mu itself,
which is why the approximation is used -- but "irrelevant here" and "exact"
are different claims.

DO NOT USE (bp - q)/b
---------------------
The formula f* = (bp - q)/b that appears in most popular write-ups is the
special case of a TWO-outcome bet: win b per unit with probability p, lose
the stake with probability q. Blackjack is not two-outcome -- a natural pays
1.5, a push pays 0, a doubled hand pays +/-2. Forcing b = 1 silently drops
the contribution of the 1.5 and 2.0 payoffs and gets both mu and the
denominator wrong.

The general formula reduces to the classic one, which is the cheapest way to
check an implementation: if X = +1 with probability p and -1 with
probability q = 1-p, then mu = p - q and E[X^2] = 1, so f* = p - q. That is
exactly the classic Kelly fraction for an even-money bet. It generalises the familiar
one across payoff structures rather than contradicting it -- while remaining,
as above, second-order in f.

A CONSEQUENCE THAT SOUNDS WRONG
--------------------------------
Allowing doubling makes your strategy BETTER (Phase 1: -2.42% -> -1.09%) but
it makes E[X^2] LARGER, because the payoff range widens to +/-2. Since
E[X^2] is the denominator, f* goes DOWN for the same edge. A stronger
strategy calls for a smaller fraction of bankroll. The reason is that Kelly
prices risk, not just return, and doubling adds variance along with the
expectation.

WHY FRACTIONAL KELLY (lambda < 1)
----------------------------------
g(f) is concave, peaks at f*, and hits ZERO again at f = 2f*: staking twice
the Kelly fraction gives zero long-run growth despite a positive edge. So
the penalty for overbetting is severe and asymmetric -- underbetting by 20%
costs a little growth, overbetting by 100% costs all of it.

Now add estimation error. You never know mu; you estimate it from a finite
sample. Because f* is proportional to mu-hat, noise in mu-hat translates
directly into noise in your bet fraction -- and since the growth penalty is
asymmetric, symmetric noise around the right answer produces an
*expected loss* of growth. Betting a fraction lambda ~ 0.5 of the Kelly
amount buys a large reduction in that exposure for a modest reduction in
growth. The standard figures: half Kelly gives up about 25% of the growth
rate and reduces the probability of ever halving your bankroll from about
50% to about 12.5%.

THE SIGNIFICANCE GATE
---------------------
There is a second, sharper problem. At rare true counts the sample is small,
so mu-hat is mostly noise. Kelly applied to a noisy estimate does not merely
add variance -- it systematically overbets, because the sizing rule takes
whatever mu-hat it is given at face value.

The defence implemented here: bet the minimum unless the LOWER end of the
confidence interval on the edge is above zero. In plain terms -- only bet
big when the edge is not just positive, but positive by more than its own
measurement error. This is exactly the discipline a trading desk applies to
a signal: size on it only once it clears its estimation noise.
"""

import numpy as np

from .counting import PreDealTracker, tc_bin

MIN_BET = 1.0           # in betting units
KELLY_LAMBDA = 0.5      # fractional Kelly multiplier
KELLY_F_MAX = 0.02      # never stake more than 2% of bankroll
CI_Z = 1.96             # 95% two-sided


def kelly_fraction(mu: float, ex2: float) -> float:
    """f* = mu / E[X^2], clipped at zero -- second order in f, not exact.

    Returns 0 for a non-positive edge: Kelly says do not bet at all, and
    negative f (betting on the dealer) is not an available action.
    """
    if not np.isfinite(mu) or not np.isfinite(ex2) or ex2 <= 0:
        return 0.0
    return max(mu / ex2, 0.0)


class KellySizer:
    """Turns a pre-deal true count into a bet size.

    Built from a PreDealTracker: the tracker holds the measured payoff
    distribution per count bin, and this class reads mu-hat, E[X^2] and the
    confidence interval out of it.

    Note the separation of concerns. The playing policy decides HOW to play
    a hand and never looks at the bankroll. This class decides HOW MUCH to
    stake and never looks at the cards. Keeping alpha generation apart from
    position sizing is the structural point of the whole project.
    """

    def __init__(self, tracker: PreDealTracker,
                 lam: float = KELLY_LAMBDA, f_max: float = KELLY_F_MAX,
                 min_bet: float = MIN_BET, z: float = CI_Z,
                 use_gate: bool = True):
        self.lam = lam
        self.f_max = f_max
        self.min_bet = min_bet
        self.use_gate = use_gate
        # Precompute per-bin sizing so the hot loop is a single lookup.
        self.fractions = np.zeros(len(tracker.hist))
        self.gated = np.zeros(len(tracker.hist), dtype=bool)
        for b in range(len(tracker.hist)):
            s = tracker.stats(b, z=z)
            if s["n"] == 0:
                continue
            if use_gate and not (s["ci_low"] > 0):
                self.gated[b] = True
                continue
            f = kelly_fraction(s["mean"], s["ex2"])
            self.fractions[b] = min(self.lam * f, self.f_max)

    def bet(self, pre_deal_tc: float, bankroll: float) -> float:
        """Stake for this hand, in betting units.

        Never below min_bet: at a table you must bet something to be dealt
        in. Never above f_max * bankroll, and never above the bankroll
        itself -- you cannot stake money you do not have.
        """
        f = self.fractions[tc_bin(pre_deal_tc)]
        return float(min(max(self.min_bet, f * bankroll), bankroll))


class FlatSizer:
    """Bet the same amount every hand. The baseline to beat."""

    def __init__(self, size: float = MIN_BET):
        self.size = size

    def bet(self, pre_deal_tc: float, bankroll: float) -> float:
        return float(min(self.size, bankroll))
