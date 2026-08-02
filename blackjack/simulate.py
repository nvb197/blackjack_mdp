"""Playing hands against the finite shoe and recording the results.

This module is the bridge between Phase 1-2 and Phase 3. It takes the
policy that Phase 1 proved optimal for the infinite deck, plays it against
the finite shoe from shoe.py, and records every outcome under the true count
that preceded it.

WHY PLAY THE INFINITE-DECK POLICY AGAINST A FINITE SHOE
-------------------------------------------------------
It sounds like a mismatch, and it is -- deliberately. Real card counters use
one fixed strategy chart (derived exactly the way Phase 1 derived it) and
vary their BET, not their playing decisions. Separating the two is what lets
you attribute a change in results to one cause at a time: if the playing
strategy is held fixed, any relationship between the count and the outcome
must come from the shoe composition, not from the policy reacting to it.

Count-dependent playing decisions ("index plays") are a separate, later
question, handled by training an agent on the count-augmented state.
"""

import numpy as np

from . import dp
from .counting import PreDealTracker
from .finite_env import FiniteBlackjackEnv
from .rules import DOUBLE, HIT, STAND


def basic_strategy(allow_double: bool = True):
    """The Phase 1 optimal policy, as two lookup tables.

    Returns (pi, should_double) where pi[t, up-1, soft] is 0 for stand and 1
    for hit, and should_double[t, up-1, soft] is True where doubling beats
    both. Doubling is only ever legal on the first decision, so the caller
    must check that separately -- these tables say what is *best*, not what
    is *legal*.
    """
    V, pi, _ = dp.value_iteration()
    if not allow_double:
        return pi, np.zeros_like(pi, dtype=bool)
    dv = dp.double_values()
    should_double = dv > V
    return pi, should_double


def play_hand(env: FiniteBlackjackEnv, pi: np.ndarray,
              should_double: np.ndarray,
              can_double: bool = True) -> tuple[float, float]:
    """Play one hand to completion. Returns (pre_deal_tc, payoff).

    The payoff is in units of the base bet: sizing is applied later, by
    multiplying this number. Keeping the two apart is the separation of
    concerns described in sizing.py -- the hand does not know how much money
    is on it.

    `can_double` is the one place that separation has to be broken. Doubling
    puts a second stake on the table, and a player who cannot cover it is not
    allowed to double. Leaving that out lets a bankroll of 1.2 units place a
    2-unit double and lose 2.4, with the shortfall silently absorbed by the
    max(..., 0) clamp -- the casino extending uncollateralised credit. The
    caller passes False when the bankroll cannot cover twice the stake.
    """
    (t, up, soft), info = env.reset()
    tc = info["pre_deal_tc"]
    if info["done"]:
        return tc, info["reward"]

    first = True
    done = False
    reward = 0.0
    while not done:
        s = int(soft)
        if first and can_double and env.allow_double and should_double[t, up - 1, s]:
            action = DOUBLE
        else:
            action = HIT if pi[t, up - 1, s] == 1 else STAND
        first = False
        (t, up, soft), reward, done, _ = env.step(action)
    return tc, reward


def play_hand_count(env, pi_count, should_double,
                    can_double: bool = True) -> tuple[float, float]:
    """Play one hand with a COUNT-DEPENDENT policy. Returns (pre_deal_tc, payoff).

    `pi_count` is indexed [total, upcard-1, soft, tc_bin]. The environment
    must have been constructed with with_count_state=True so that the bin
    arrives as part of the state.
    """
    (t, up, soft, b), info = env.reset()
    tc = info["pre_deal_tc"]
    if info["done"]:
        return tc, info["reward"]
    first, done, reward = True, False, 0.0
    while not done:
        sft = int(soft)
        if (first and can_double and env.allow_double
                and should_double[t, up - 1, sft]):
            action = DOUBLE
        else:
            action = HIT if pi_count[t, up - 1, sft, b] == 1 else STAND
        first = False
        (t, up, soft, b), reward, done, _ = env.step(action)
    return tc, reward


def measure_edge(n_hands: int, seed: int = 42, allow_double: bool = True,
                 progress_every: int = 0) -> PreDealTracker:
    """Play `n_hands` hands of basic strategy and record edge by pre-deal count."""
    pi, should_double = basic_strategy(allow_double)
    env = FiniteBlackjackEnv(np.random.default_rng(seed),
                             allow_double=allow_double)
    tracker = PreDealTracker()
    for i in range(n_hands):
        tc, payoff = play_hand(env, pi, should_double)
        tracker.record(tc, payoff)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i+1:,}/{n_hands:,}")
    return tracker


def bankroll_paths(sizer, n_paths: int, n_hands: int, initial: float = 1000.0,
                   seed: int = 100, allow_double: bool = True,
                   pi_count: np.ndarray | None = None) -> np.ndarray:
    """Simulate `n_paths` independent bankroll trajectories under `sizer`.

    Returns an array of shape (n_paths, n_hands + 1) including the starting
    bankroll in column 0.

    The bet is sized from the pre-deal true count only. The hand itself is
    played by the fixed basic-strategy tables, so any difference between
    sizing rules is attributable purely to sizing -- the cards and the
    playing decisions are identical given the same seed.

    A path that reaches zero stops: the remaining columns are filled with
    zero rather than allowed to go negative, because a bankroll cannot be
    less than nothing and ruin is absorbing.

    Doubling is refused when the bankroll cannot cover twice the stake. That
    check has to live here rather than in play_hand, because it is the only
    place that knows the bankroll -- and without it a nearly-broke path can
    place a double it cannot pay for, with the shortfall clamped away at
    zero. At a starting bankroll of 1000 against a 1-unit minimum this never
    fires; start at 2 units and it fires on about 7% of doubles.
    """
    pi, should_double = basic_strategy(allow_double)
    use_count = pi_count is not None
    out = np.zeros((n_paths, n_hands + 1))
    out[:, 0] = initial
    for p in range(n_paths):
        env = FiniteBlackjackEnv(np.random.default_rng(seed + p),
                                 allow_double=allow_double,
                                 with_count_state=use_count)
        bank = initial
        for h in range(n_hands):
            if bank <= 0:
                out[p, h + 1:] = 0.0
                break
            stake = sizer.bet(env.pre_deal_true_count(), bank)
            can_double = bank >= 2.0 * stake
            if use_count:
                tc, payoff = play_hand_count(env, pi_count, should_double,
                                             can_double)
            else:
                tc, payoff = play_hand(env, pi, should_double, can_double)
            bank = max(bank + stake * payoff, 0.0)
            out[p, h + 1] = bank
    return out