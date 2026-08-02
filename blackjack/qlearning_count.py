"""Q-learning over the count-augmented state: learning "index plays".

Phase 1-2 learned a policy over (total, upcard, soft) -- 200 decision states.
That policy is optimal for an infinite deck, where the count carries no
information because the composition never changes.

With a finite shoe the composition drifts, so the best action can depend on
the count as well. The classic example: hard 16 against a dealer ten. Phase 1
found standing and hitting almost exactly equal there (a gap of 0.0006). In a
ten-rich shoe -- a high count -- hitting is more likely to bust you, so
standing becomes the better choice. Card-counting literature calls these
"index plays".

This module adds the true count bin to the state:

    (total, upcard, soft)  ->  (total, upcard, soft, tc_bin)

    200 decision states    ->  1600 decision states

THE COST, AND WHY IT IS WORSE THAN "8 TIMES"
--------------------------------------------
The state space grows 8-fold, so naively you need 8 times the data. It is
worse than that, because the true count is roughly bell-shaped around zero:
the bins near 0 collect most hands and the tails collect very few. Measured
frequencies from a 3-million-hand run:

    <-3   8.7%    -1..0  18.9%    1..2  11.8%
   -3..-2 6.8%     0..1  26.6%    2..3   6.4%
   -2..-1 12.4%                   >=3    8.4%

So the tail bins get a fraction of the data the central bins get. Since the
value gaps this is trying to resolve are around 0.002-0.02 (Phase 2 showed
the agent's noise floor at 5 million hands was already about 0.007), the tail
bins will be undersampled at any realistic budget, and their learned actions
should be treated as noisy rather than authoritative.

Stating that limitation is more useful than presenting a full 1600-cell chart
as though every cell were equally trustworthy. `index_deviations` therefore
reports the visit count for every deviation it lists, so an undersampled cell
is visible rather than hidden.

THE CONFOUNDING TRAP THIS MODULE EXISTS TO AVOID
-------------------------------------------------
If you change the playing strategy AND the bet sizing at the same time and
the results improve, you cannot tell which change caused it. That is
textbook confounding, and it is the first thing a quant interviewer will
probe.

The fix is a 2x2 experiment on identical seeds:

                         flat bet     Kelly bet
    fixed strategy       baseline        (a)
    count strategy          (b)          (c)

Then (a) - baseline isolates the value of sizing, (b) - baseline isolates the
value of count-dependent play, and (c) shows whether they add up or interact.

Published work (Griffin, Wong) puts bet variation at roughly 90% of the total
benefit of counting and index plays at roughly 10%. That split is NOT
reproduced here, and the experiment as run cannot reproduce it: measured on
400 paired paths, bet sizing is worth +61.45 with t = 12.80 while
count-dependent play is worth +4.41 with t = 1.74 -- a term indistinguishable
from zero. A percentage split cannot be taken of a quantity whose sign is
uncertain. The direction agrees with the literature; the ratio is not
something this sample size can supply.

WHAT THIS MODULE PRODUCED, AND WHAT HAPPENED TO IT
---------------------------------------------------
The first run of this module reported that the agent had recovered the
best-known index play -- stand on hard 16 against a ten once the true count
reaches zero -- at exactly the published threshold, with 68,681 hands behind
it. That was an artefact of a data leak: the dealer's face-down card was being
folded into the count before the agent decided (see finite_env.draw_hole).
Rerun without the leak, that cell goes back to hitting and the threshold moves
up one bin, while a deviation pointing the wrong way appears in a cell holding
36,825 hands. The direction of the index plays reproduces; the thresholds do
not resolve at this sample size and should not be quoted. See the README.
"""

import numpy as np

from .counting import N_BINS
from .finite_env import FiniteBlackjackEnv
from .qlearning import T_HI, T_LO

N_TOTAL, N_UP, N_SOFT, N_ACTIONS = 22, 10, 2, 2


class CountQLearningAgent:
    """Tabular Q-learning over (total, upcard, soft, tc_bin) x (stand, hit).

    Identical update rule and step-size schedule to QLearningAgent -- the
    only change is the extra state dimension. Keeping the algorithm the same
    is deliberate: any difference in results is then attributable to the
    state representation, not to a change in the learner.
    """

    def __init__(self, rng: np.random.Generator, omega: float = 1.0,
                 eps_start: float = 1.0, eps_min: float = 0.05,
                 eps_decay_episodes: int = 2_000_000):
        self.rng = rng
        self.omega = omega
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay_episodes = eps_decay_episodes
        shape = (N_TOTAL, N_UP, N_SOFT, N_BINS, N_ACTIONS)
        self.Q = np.zeros(shape)
        self.N = np.zeros(shape, dtype=np.int64)

    def epsilon(self, episode: int) -> float:
        if episode >= self.eps_decay_episodes:
            return self.eps_min
        frac = episode / self.eps_decay_episodes
        return self.eps_start + frac * (self.eps_min - self.eps_start)

    def act(self, t, up, soft, b, eps) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(0, N_ACTIONS))
        q = self.Q[t, up, soft, b]
        if q[0] == q[1]:
            return int(self.rng.integers(0, N_ACTIONS))
        return int(np.argmax(q))

    def update(self, t, up, soft, b, a, r, nt, nup, nsoft, nb, done) -> None:
        self.N[t, up, soft, b, a] += 1
        alpha = self.N[t, up, soft, b, a] ** (-self.omega)
        target = r if done else r + np.max(self.Q[nt, nup, nsoft, nb])
        self.Q[t, up, soft, b, a] += alpha * (target - self.Q[t, up, soft, b, a])

    def greedy_policy(self) -> np.ndarray:
        """Greedy action per state, ties broken deterministically. See QLearningAgent.

        The tie-break uses a locally seeded generator, not self.rng, so that
        this query does not advance the agent's stream and perturb training.
        """
        q_stand, q_hit = self.Q[..., 0], self.Q[..., 1]
        coin = np.random.default_rng(0).integers(0, 2, size=q_stand.shape)
        return np.where(q_stand == q_hit, coin,
                        (q_hit > q_stand).astype(np.int64)).astype(np.int64)


def train_count(agent: CountQLearningAgent, env: FiniteBlackjackEnv,
                n_episodes: int) -> None:
    """Train on the finite shoe with the count in the state.

    Hands that end at the deal produce no state-action pair and are skipped,
    exactly as in the infinite-deck trainer: the reward belongs to no action.
    """
    for ep in range(n_episodes):
        eps = agent.epsilon(ep)
        (t, up, soft, b), info = env.reset()
        if info["done"]:
            continue
        done = False
        while not done:
            a = agent.act(t, up - 1, int(soft), b, eps)
            (nt, nup, nsoft, nb), r, done, _ = env.step(a)
            agent.update(t, up - 1, int(soft), b, a, r,
                         nt, nup - 1, int(nsoft), nb, done)
            t, up, soft, b = nt, nup, nsoft, nb


def index_deviations(agent: CountQLearningAgent,
                     pi_star: np.ndarray) -> list[dict]:
    """Cells where the count-aware policy departs from the infinite-deck one.

    Only cells with enough visits to be worth reporting are returned; the
    rest are noise, for the reasons in the module docstring.
    """
    pi = agent.greedy_policy()
    out = []
    for t in range(T_LO, T_HI + 1):
        for up in range(10):
            for soft in range(2):
                base = pi_star[t, up, soft]
                for b in range(N_BINS):
                    if pi[t, up, soft, b] != base:
                        visits = int(agent.N[t, up, soft, b].sum())
                        out.append({
                            "total": t,
                            "upcard": "A" if up == 0 else up + 1,
                            "soft": bool(soft),
                            "bin": b,
                            "visits": visits,
                            "from": "H" if base else "S",
                            "to": "S" if base else "H",
                        })
    return sorted(out, key=lambda d: -d["visits"])