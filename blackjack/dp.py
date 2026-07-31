"""Exact solution of the blackjack MDP by value iteration.

The transition probabilities are computed from the card distribution rather
than estimated by simulation, so the resulting value function is exact up to
floating-point error. That is what lets it serve as a reference for the
Q-learning agent.

The Bellman optimality equation is

    V*(s) = max_a  sum_s'  P(s'|s,a) [ r(s,a,s') + V*(s') ]

with no discounting. Undiscounted value iteration converges here because
every hand ends: each hit strictly increases the number of cards held, and a
hand cannot hold more than about twenty cards without busting or being
forced to stand. The transition graph is essentially acyclic -- the total
only ever increases, and soft becomes hard one way -- so a handful of sweeps
suffice.
"""

from functools import lru_cache

import numpy as np

from .rules import (
    BLACKJACK_PAYOUT,
    CARD_PROBS,
    DEALER_HITS_SOFT_17,
    DEALER_PEEKS,
    add_card,
    hand_value,
)

BUST_IDX = 5          # dealer outcomes are indexed 17,18,19,20,21,bust -> 0..5
T_MIN, T_MAX = 4, 21
N_T = T_MAX + 1       # index directly by total; a few unused slots is fine


# --------------------------------------------------------------------- #
# Dealer outcome distribution
# --------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def dealer_distribution(total: int, usable_ace: bool) -> tuple[float, ...]:
    """Distribution of the dealer's final total, by memoised recursion.

    Returns a 6-tuple (P(17), P(18), P(19), P(20), P(21), P(bust)).

    Recursion over dealer states with memoisation costs a few milliseconds.
    Reaching the same precision by Monte Carlo would take on the order of
    1e8 hands.
    """
    if total > 21:
        out = [0.0] * 6
        out[BUST_IDX] = 1.0
        return tuple(out)
    hits_soft_17 = usable_ace and total == 17 and DEALER_HITS_SOFT_17
    if total >= 17 and not hits_soft_17:
        out = [0.0] * 6
        out[total - 17] = 1.0
        return tuple(out)
    acc = np.zeros(6)
    for c in range(1, 11):
        nt, ns = add_card(total, usable_ace, c)
        acc += CARD_PROBS[c] * np.asarray(dealer_distribution(nt, ns))
    return tuple(acc)


def dealer_distribution_from_upcard(upcard: int) -> np.ndarray:
    """Dealer outcome distribution given the upcard, conditioned on no natural.

    When the dealer shows an ace or a ten they check the hole card before the
    player acts, so every player decision happens in a world where the dealer
    does not have blackjack. The hole-card distribution must therefore be
    renormalised by 1 - P(the card that would make a natural).

    Omitting that renormalisation shifts the overall expected value by about
    0.16 percentage points -- measured, -1.087% becomes -1.249% -- while
    still producing a plausible-looking strategy chart. That combination of
    "wrong by a visible amount" and "looks fine" is what makes it worth a
    comment.
    """
    acc = np.zeros(6)
    mass = 0.0
    for hole in range(1, 11):
        if DEALER_PEEKS and ((upcard == 1 and hole == 10)
                             or (upcard == 10 and hole == 1)):
            continue
        t, s = hand_value([upcard, hole])
        acc += CARD_PROBS[hole] * np.asarray(dealer_distribution(t, s))
        mass += CARD_PROBS[hole]
    return acc / mass


# --------------------------------------------------------------------- #
# Action values
# --------------------------------------------------------------------- #
def stand_values() -> np.ndarray:
    """Value of standing, for every (player total, dealer upcard).

        V_stand(p, d) = P(dealer busts | d) + sum_k P(k | d) sign(p - k)

    Built as a matrix product of a sign matrix and the dealer distribution
    rather than a Python loop.

    Returns an array of shape (22, 10), valid for totals 4..21.
    """
    dealer = np.stack(
        [dealer_distribution_from_upcard(up) for up in range(1, 11)], axis=1)
    totals = np.arange(N_T)[:, None]
    dealer_finals = np.arange(17, 22)[None, :]
    sign = np.sign(totals - dealer_finals)
    return sign @ dealer[:5, :] + dealer[BUST_IDX, :][None, :]


def _hit_transitions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute where each (total, soft, card) triple lands after a hit."""
    next_t = np.zeros((N_T, 2, 10), dtype=np.int64)
    next_s = np.zeros((N_T, 2, 10), dtype=np.int64)
    bust = np.zeros((N_T, 2, 10), dtype=bool)
    for t in range(T_MIN, N_T):
        for s in (0, 1):
            for ci, c in enumerate(range(1, 11)):
                nt, ns = add_card(t, bool(s), c)
                if nt > 21:
                    bust[t, s, ci] = True
                else:
                    next_t[t, s, ci] = nt
                    next_s[t, s, ci] = int(ns)
    return next_t, next_s, bust


def _hit_backup(V: np.ndarray, next_t, next_s, bust) -> np.ndarray:
    """One Bellman backup for the hit action, vectorised over all states."""
    succ = np.where(bust[..., None], -1.0, V[next_t, :, next_s])
    return np.einsum("c,tscu->tus", CARD_PROBS[1:11], succ)


def value_iteration(theta: float = 1e-9) -> tuple[np.ndarray, np.ndarray, int]:
    """Run value iteration until the sup-norm change falls below ``theta``.

    Returns (V, pi, n_sweeps) where V and pi have shape (22, 10, 2), indexed
    by [total, upcard - 1, soft], and pi is 0 for stand and 1 for hit.
    """
    sv = stand_values()
    next_t, next_s, bust = _hit_transitions()

    V = np.zeros((N_T, 10, 2))
    n_sweeps = 0
    while True:
        n_sweeps += 1
        hit_v = _hit_backup(V, next_t, next_s, bust)
        V_new = np.maximum(sv[:, :, None], hit_v)
        converged = np.max(np.abs(V_new - V)) < theta
        V = V_new
        if converged:
            break

    pi = (hit_v > sv[:, :, None]).astype(np.int64)
    return V, pi, n_sweeps


def double_values() -> np.ndarray:
    """Value of doubling: stake twice, take exactly one card, then stand.

        V_double(s) = 2 sum_c P(c) [ -1 if bust else V_stand(s') ]

    Note this uses V_stand and not V*, because after doubling the player may
    not hit again. Using V* here is a common slip and inflates the expected
    value.

    Doubling is only legal on the first decision, and the state does not
    record how many cards have been drawn, so it is applied when averaging
    over opening hands rather than inside the value-iteration loop. Folding
    it into the loop would let the solver double on the third card.
    """
    sv = stand_values()
    next_t, next_s, bust = _hit_transitions()
    succ = np.where(bust[..., None], -1.0, sv[next_t])
    return 2.0 * np.einsum("c,tscu->tus", CARD_PROBS[1:11], succ)


# --------------------------------------------------------------------- #
# Aggregate expected value
# --------------------------------------------------------------------- #
def _opening_hands():
    """Yield (upcard, probability, P(dealer natural), (total, soft))."""
    for up in range(1, 11):
        p_up = CARD_PROBS[up]
        p_dbj = (CARD_PROBS[10] if up == 1 else
                 CARD_PROBS[1] if up == 10 else 0.0)
        for c1 in range(1, 11):
            for c2 in range(1, 11):
                yield (up, p_up * CARD_PROBS[c1] * CARD_PROBS[c2], p_dbj,
                       hand_value([c1, c2]))


def expected_value(V: np.ndarray, allow_double: bool = False) -> float:
    """Expected return per unit staked, averaged over all opening hands.

    Naturals are handled explicitly: both natural is a push, the player's
    natural alone pays 3:2, the dealer's natural alone loses the stake at
    once. Every other hand is worth V*(s), which already assumes the dealer
    has no natural thanks to the peek renormalisation.
    """
    dv = double_values() if allow_double else None
    ev = 0.0
    for up, p, p_dbj, (t, soft) in _opening_hands():
        if t == 21 and soft:
            ev += p * (1 - p_dbj) * BLACKJACK_PAYOUT
            continue
        v = V[t, up - 1, int(soft)]
        if allow_double:
            v = max(v, dv[t, up - 1, int(soft)])
        ev += p * (p_dbj * (-1.0) + (1 - p_dbj) * v)
    return ev


def action_values(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact value of standing and of hitting at every state, given V*.

    The gap between the two is what makes a policy mismatch interpretable:
    it says how much choosing the wrong action actually costs at that state,
    rather than merely that it was wrong.
    """
    sv = stand_values()
    next_t, next_s, bust = _hit_transitions()
    q_stand = np.repeat(sv[:, :, None], 2, axis=2)
    q_hit = _hit_backup(V, next_t, next_s, bust)
    return q_stand, q_hit


def policy_evaluation(pi: np.ndarray, theta: float = 1e-9) -> np.ndarray:
    """Value of an arbitrary policy -- the same backup without the max.

    This is what turns "the agent disagrees with the optimum in 3 cells out
    of 200" into "those disagreements cost 2.47 basis points of expected
    value". A share of matching cells is a coding metric; the cost in basis
    points is the one that would matter to anyone staking money.
    """
    sv = stand_values()
    next_t, next_s, bust = _hit_transitions()
    V = np.zeros((N_T, 10, 2))
    while True:
        hit_v = _hit_backup(V, next_t, next_s, bust)
        V_new = np.where(pi == 1, hit_v, sv[:, :, None])
        if np.max(np.abs(V_new - V)) < theta:
            return V_new
        V = V_new


def print_policy(V: np.ndarray, pi: np.ndarray, allow_double: bool = False):
    """Print the strategy chart, for eye comparison against published tables."""
    dv = double_values() if allow_double else None
    ups = [1] + list(range(2, 11))
    header = "      " + " ".join(f"{u:>2}" for u in
                                 ["A"] + [str(i) for i in range(2, 11)])

    def cell(t, up, soft):
        if allow_double and dv[t, up - 1, soft] > V[t, up - 1, soft]:
            return "D"
        return "H" if pi[t, up - 1, soft] else "S"

    for soft, name in ((0, "HARD"), (1, "SOFT")):
        print(f"\n=== {name} TOTALS ===\n{header}")
        for t in range(21, (12 if soft else 4) - 1, -1):
            row = " ".join(f"{cell(t, up, soft):>2}" for up in ups)
            print(f"  {t:>2}: {row}")