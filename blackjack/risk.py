"""Risk measurement: VaR, CVaR, drawdown, risk of ruin, growth rate.

READ THIS FIRST if these terms are new.

A positive edge is not enough. Two strategies with the same expected profit
can differ enormously in how likely they are to wipe you out along the way.
This module measures that.

SIGN CONVENTION -- FIX IT ONCE AND TEST IT
-------------------------------------------
Everything here works with LOSS, defined as

    L = -PnL

so a loss of 30 units is L = +30 and a profit of 30 units is L = -30. With
this convention VaR and CVaR come out POSITIVE when there is money at risk,
which is how risk reports read.

Getting this backwards is the single most common bug in risk code, and it is
invisible: the numbers still look like numbers. Hence
test_var_sign_convention.

VALUE AT RISK (VaR)
-------------------
    VaR_alpha(L) = inf { x : P(L <= x) >= alpha }

In words: the loss level that is not exceeded with probability alpha. At
alpha = 0.99, "VaR is 40 units" means: on 99% of paths you lose at most 40.

The thing VaR does not tell you: what happens on the other 1%. Two
strategies can share a VaR of 40 while one loses 41 in the tail and the
other loses your entire bankroll. VaR is blind to tail SHAPE.

CONDITIONAL VALUE AT RISK (CVaR, also "expected shortfall")
-----------------------------------------------------------
    CVaR_alpha(L) = E[ L | L >= VaR_alpha(L) ]

The average loss GIVEN that you are in the bad tail. This sees tail shape,
which is why regulators moved from VaR to expected shortfall in FRTB.

CVaR >= VaR always, by construction: an average over a region where every
value is at least VaR cannot be below VaR. Because that is a mathematical
identity rather than an empirical finding, it belongs in the code as an
assertion -- if it ever fails, there is a bug, full stop.

WHY CVaR IS "COHERENT" AND VaR IS NOT
--------------------------------------
Artzner et al. (1999) set out four properties a sensible risk measure should
have: monotonicity, translation invariance, positive homogeneity, and
SUBADDITIVITY -- rho(X + Y) <= rho(X) + rho(Y), i.e. combining two positions
cannot be riskier than the sum of their separate risks.

VaR violates subadditivity. The standard counterexample: two independent
bonds, each defaulting with probability 4%, each losing 100 on default. At
the 95% level, each alone has VaR = 0, because default sits in the 4% tail
which 95% VaR does not reach. Together, the chance that at least one
defaults is about 7.8%, so the 95% VaR of the pair is 100 -- greater than
0 + 0. VaR has just told you diversification increased risk, which is
nonsense, and it means VaR cannot be aggregated across desks. CVaR
satisfies all four axioms.

MAXIMUM DRAWDOWN
----------------
    MDD = max over t of  (running_peak_t - B_t) / running_peak_t

The worst peak-to-trough fall along the path, as a fraction. Measured on the
bankroll after EVERY hand, not just at the end -- the whole point is the
path, and a strategy that dips to 10% of the bankroll before recovering is
not equivalent to one that never dips, even if they finish at the same
place.

RISK OF RUIN
------------
Fraction of paths that ever touch a threshold (here 50% of the starting
bankroll). "Ever touch" matters: a path that halves and then recovers still
counts, because in reality that is the point at which most people stop, get
stopped, or lose their nerve.

Kelly theory gives a closed form for the probability of ever touching a
fraction x of the bankroll under fractional Kelly with multiplier lambda:

    P(touch x) = x^((2 - lambda) / lambda)

Full Kelly (lambda = 1): P(halving) = 0.5^1 = 50%.
Half Kelly (lambda = 0.5): P(halving) = 0.5^3 = 12.5%.

So halving the bet fraction cuts catastrophic risk fourfold at the cost of
about a quarter of the growth rate. That single line is the whole philosophy
of position sizing.

Note this formula comes from a diffusion approximation -- continuous time,
infinitely divisible bets, known edge. Blackjack violates all three
(discrete hands, minimum bet, estimated edge), so agreement should be
expected only in order of magnitude. Reporting the gap honestly is more
informative than hiding it.

AN HONEST WARNING ABOUT TAIL ESTIMATES
---------------------------------------
A 99% quantile estimated from 1,000 paths rests on about 10 observations.
The standard error on CVaR is then large. This module therefore provides a
bootstrap interval, and callers should quote it rather than presenting a
point estimate as though it were precise.
"""

import numpy as np


def var(losses: np.ndarray, alpha: float = 0.99) -> float:
    """Value at Risk at level alpha, on the LOSS convention (L = -PnL)."""
    return float(np.quantile(np.asarray(losses, dtype=float), alpha))


def cvar(losses: np.ndarray, alpha: float = 0.99) -> float:
    """Conditional Value at Risk: mean loss in the worst (1-alpha) tail.

    THE ATOM PROBLEM, AND WHY THE OBVIOUS VERSION IS WRONG
    ------------------------------------------------------
    The natural implementation -- average everything at or above VaR -- is
    incorrect whenever a chunk of probability sits exactly at VaR, which is
    routine for discrete payoffs.

    Concretely: 99 paths break even (loss 0) and one is wiped out (loss 100).
    VaR at 90% is 0. Averaging `losses >= 0` averages ALL 100 paths and gives
    1.0. But the worst 10% consists of nine zeros and one 100, whose mean is
    10.0 -- ten times larger. The naive version silently understates the tail
    by pulling in mass that does not belong to it.

    The correct definition for a distribution with atoms is

        ES = 1/(1-a) * ( E[L * 1{L > VaR}] + VaR * (P(L <= VaR) - a) )

    The first term takes everything strictly beyond VaR; the second adds back
    exactly the sliver of probability sitting AT VaR that is needed to make
    the tail weigh (1-a) and no more. Where there is no atom the second term
    vanishes and this reduces to the naive formula, which is why the bug can
    hide for a long time on continuous-looking data.

    Note this is more accurate than the common shortcut of averaging the
    ceil((1-a)*n) worst observations: that rounds the tail size up to a whole
    number of samples, which biases the result when (1-a)*n is not an integer.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.size == 0:
        return float("nan")
    v = var(losses, alpha)
    beyond = losses[losses > v]
    tail_weight = 1.0 - alpha
    if tail_weight <= 0:
        return float(losses.max())
    at_or_below = float(np.mean(losses <= v))
    es = (beyond.sum() / losses.size + v * (at_or_below - alpha)) / tail_weight
    # CVaR >= VaR is an identity; clamp against floating-point drift only.
    return float(max(es, v))


def max_drawdown(bankroll_path: np.ndarray) -> float:
    """Worst peak-to-trough decline along the path, as a fraction in [0, 1]."""
    b = np.asarray(bankroll_path, dtype=float)
    peak = np.maximum.accumulate(b)
    dd = (peak - b) / np.where(peak > 0, peak, 1.0)
    return float(np.clip(dd.max(), 0.0, 1.0))


def growth_rate(initial: float, final: float, n_hands: int) -> float:
    """Log growth rate per hand: (1/T) log(B_T / B_0).

    Returns -inf for a ruined path, which is the mathematically honest
    answer -- log(0) is -inf, and a wiped-out bankroll cannot compound.
    Callers must decide how to aggregate that rather than having it silently
    turned into a finite number.
    """
    if final <= 0:
        return -np.inf
    return float(np.log(final / initial) / n_hands)


def risk_of_ruin(final_or_paths: np.ndarray, initial: float,
                 threshold: float = 0.5) -> float:
    """Fraction of paths that EVER touched `threshold` * initial.

    Accepts a 2-D array of shape (n_paths, n_steps): the minimum along each
    path is what matters, not the endpoint, because touching the threshold
    is what counts as ruin.
    """
    paths = np.atleast_2d(np.asarray(final_or_paths, dtype=float))
    return float(np.mean(paths.min(axis=1) <= threshold * initial))


def theoretical_ruin(lam: float, x: float = 0.5) -> float:
    """Closed-form P(ever touch x * bankroll) under fractional Kelly.

    P = x^((2 - lambda) / lambda). Diffusion approximation -- see the module
    docstring on why blackjack only matches this in order of magnitude.
    """
    return float(x ** ((2.0 - lam) / lam))


def bootstrap_ci(values: np.ndarray, statistic, n_boot: int = 2000,
                 alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Percentile bootstrap interval for any statistic of `values`.

    Used for CVaR, where the point estimate rests on a handful of tail
    observations and quoting it alone would overstate the precision.
    """
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    n = values.size
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = statistic(values[rng.integers(0, n, n)])
    return (float(np.quantile(stats, alpha / 2)),
            float(np.quantile(stats, 1 - alpha / 2)))


def summarise(paths: np.ndarray, initial: float, alpha: float = 0.99,
              ruin_threshold: float = 0.5, rng=None) -> dict:
    """Full risk report for a bundle of bankroll paths.

    `paths` has shape (n_paths, n_hands + 1), including the starting
    bankroll as column 0.
    """
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    n_hands = paths.shape[1] - 1
    final = paths[:, -1]
    losses = initial - final                      # L = -PnL

    v = var(losses, alpha)
    c = cvar(losses, alpha)
    assert c >= v - 1e-9, "CVaR below VaR is mathematically impossible"

    mdd = np.array([max_drawdown(p) for p in paths])
    survivors = final > 0
    growth = np.array([growth_rate(initial, f, n_hands) for f in final])

    lo, hi = bootstrap_ci(losses, lambda x: cvar(x, alpha), rng=rng)
    return {
        "n_paths": int(paths.shape[0]),
        "n_hands": n_hands,
        "mean_final": float(final.mean()),
        "median_final": float(np.median(final)),
        "var": v,
        "cvar": c,
        "cvar_ci": (lo, hi),
        "mdd_mean": float(mdd.mean()),
        "mdd_worst": float(mdd.max()),
        "ruin": risk_of_ruin(paths, initial, ruin_threshold),
        "growth_median": float(np.median(growth[survivors]))
        if survivors.any() else -np.inf,
        "survived": float(survivors.mean()),
    }
