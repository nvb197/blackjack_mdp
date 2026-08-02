"""Tabular Q-learning, and the machinery for comparing it to the exact solution.

The update is the usual off-policy temporal-difference control rule

    Q(s,a) <- Q(s,a) + alpha_t(s,a) [ r + max_a' Q(s',a') - Q(s,a) ]

which converges to Q* provided every state-action pair is visited infinitely
often, rewards are bounded, and the step sizes satisfy the Robbins-Monro
conditions  sum alpha = infinity,  sum alpha^2 < infinity.

Two design choices follow from those conditions.

The step size decays per state-action pair, alpha = N(s,a)^-omega with N
counting visits from 1, not on a global clock. Visit frequencies are very
uneven -- hard 12 against a six comes up orders of magnitude more often than
soft 21 against an ace (measured 135x between the most and least visited cell
of the decision region). A global clock would drive the step size to zero at
rare states that have barely been updated, and Robbins-Monro would fail
exactly where the estimates are worst.

Exploration decays to a floor rather than to zero, which keeps every pair
reachable. That is safe here only because Q-learning is off-policy: the max
in the target means it learns the value of the greedy policy whatever the
behaviour policy does. SARSA under the same schedule would converge to the
value of the epsilon-greedy policy instead, not to Q*.
"""

import numpy as np

from .env import BlackjackEnv


N_TOTAL, N_UP, N_SOFT, N_ACTIONS = 22, 10, 2, 2

# Totals below 12 are excluded from every comparison: hitting cannot bust, so
# it is trivially optimal and there is no decision to get right.
T_LO, T_HI = 12, 21


class QLearningAgent:
    """Q-learning over the (total, upcard, soft) x (stand, hit) table."""

    def __init__(self, rng: np.random.Generator, omega: float = 1.0,
                 eps_start: float = 1.0, eps_min: float = 0.05,
                 eps_decay_episodes: int = 500_000):
        self.rng = rng
        self.omega = omega
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.eps_decay_episodes = eps_decay_episodes
        self.Q = np.zeros((N_TOTAL, N_UP, N_SOFT, N_ACTIONS))
        self.N = np.zeros((N_TOTAL, N_UP, N_SOFT, N_ACTIONS), dtype=np.int64)

    def epsilon(self, episode: int) -> float:
        """Linear decay to a floor, then constant."""
        if episode >= self.eps_decay_episodes:
            return self.eps_min
        frac = episode / self.eps_decay_episodes
        return self.eps_start + frac * (self.eps_min - self.eps_start)

    def act(self, t: int, up: int, soft: int, eps: float) -> int:
        if self.rng.random() < eps:
            return int(self.rng.integers(0, N_ACTIONS))
        q = self.Q[t, up, soft]
        if q[0] == q[1]:
            # Break ties at random. The table starts at zero, so always
            # taking argmax would systematically favour standing early on.
            return int(self.rng.integers(0, N_ACTIONS))
        return int(np.argmax(q))

    def update(self, t, up, soft, a, r, nt, nup, nsoft, done) -> float:
        """One temporal-difference update. Returns the TD error.

        At a terminal state there is no successor, so max_a' Q(s',a') is
        taken to be zero and the target collapses to the reward. Getting
        this wrong propagates backwards through the bootstrap into almost
        every state, since most blackjack states sit one to three steps
        from the end of the hand.
        """
        self.N[t, up, soft, a] += 1          # visit count starts at 1
        alpha = self.N[t, up, soft, a] ** (-self.omega)
        target = r if done else r + np.max(self.Q[nt, nup, nsoft])
        td_error = target - self.Q[t, up, soft, a]
        self.Q[t, up, soft, a] += alpha * td_error
        return float(td_error)

    def greedy_policy(self) -> np.ndarray:
        """Greedy action per state, with ties broken deterministically-at-random.

        Two requirements pull against each other here.

        np.argmax alone returns the FIRST maximal index, so a state with
        Q(stand) == Q(hit) -- which is what a never-visited cell looks like,
        both entries still at zero -- would always be reported as "stand".
        That is a bias built into the reporting rather than into the learning.

        But this is a read-only QUERY, so it must not mutate the agent. An
        earlier version drew the tie-break from self.rng, which advanced the
        agent's random stream: calling compare() at training checkpoints then
        perturbed the training itself, and the final Q table depended on how
        many times it had been inspected (measured max difference 0.48). An
        observer effect in a reproducibility-critical path is worse than the
        bias it was fixing.

        The resolution is a local generator with a fixed seed. It is unbiased
        between the two actions, identical on every call, and touches neither
        self.rng nor the global numpy state.

        For scale: on a 2-million-hand run the decision region (totals 12-21)
        contains zero exact ties, and the 160 cells that do tie are unreachable
        states such as a total of 0. Flipping all of them moves the reported
        cost by 0.000000 bps. This is defensive, not load-bearing.
        """
        q_stand, q_hit = self.Q[..., 0], self.Q[..., 1]
        coin = np.random.default_rng(0).integers(0, 2, size=q_stand.shape)
        return np.where(q_stand == q_hit, coin,
                        (q_hit > q_stand).astype(np.int64)).astype(np.int64)

    def value_estimate(self) -> np.ndarray:
        return np.max(self.Q, axis=-1)


def train(agent: QLearningAgent, env: BlackjackEnv, n_episodes: int,
          eval_every: int = 0, eval_fn=None) -> list[dict]:
    """Train for ``n_episodes`` hands, optionally recording diagnostics.

    Hands that end at the deal -- a natural on either side -- produce no
    state-action pair and are skipped. Their reward belongs to no action and
    must not be credited to one, or every state with a total of 21 picks up
    a bias. The periodic evaluation deliberately sits outside that skip:
    using ``continue`` would silently drop any checkpoint that happened to
    land on a natural, which is about 9% of hands (see hand.py).
    """
    history: list[dict] = []
    for ep in range(n_episodes):
        eps = agent.epsilon(ep)
        (t, up, soft), info = env.reset()

        if not info["done"]:
            done = False
            while not done:
                a = agent.act(t, up - 1, int(soft), eps)
                (nt, nup, nsoft), r, done, _ = env.step(a)
                agent.update(t, up - 1, int(soft), a, r,
                             nt, nup - 1, int(nsoft), done)
                t, up, soft = nt, nup, nsoft

        if eval_every and eval_fn is not None and (ep + 1) % eval_every == 0:
            history.append(eval_fn(agent, ep + 1))
    return history


# --------------------------------------------------------------------- #
# Comparison against the exact solution
# --------------------------------------------------------------------- #
def _decision_region(arr: np.ndarray) -> np.ndarray:
    return arr[T_LO:T_HI + 1]


def compare(agent: QLearningAgent, V_star: np.ndarray,
            pi_star: np.ndarray) -> dict:
    """Mean squared value error and share of matching decisions.

    Both are needed. A small value error does not imply the right decision:
    where the two actions are nearly equal in value, an error of 0.001 still
    flips the argmax. And matching decisions do not imply small value error,
    since a constant offset in Q leaves the argmax alone.
    """
    d = _decision_region(agent.value_estimate()) - _decision_region(V_star)
    agree = _decision_region(agent.greedy_policy()) == _decision_region(pi_star)
    return {"mse": float(np.mean(d ** 2)),
            "max_err": float(np.max(np.abs(d))),
            "agreement": float(np.mean(agree)),
            "min_visits": int(_decision_region(agent.N).min())}


def disagreements(agent: QLearningAgent, pi_star: np.ndarray) -> list[dict]:
    """List every mismatched cell together with the value gap between actions.

    Cells that still disagree after convergence are almost always ones where
    |Q(s,stand) - Q(s,hit)| is tiny, so the two actions are nearly
    equivalent and choosing wrongly costs almost nothing. Sorting by that gap
    makes the point directly.
    """
    pi_hat = agent.greedy_policy()
    out = []
    for t in range(T_LO, T_HI + 1):
        for up in range(10):
            for soft in range(2):
                if pi_hat[t, up, soft] != pi_star[t, up, soft]:
                    out.append({
                        "total": t,
                        "upcard": "A" if up == 0 else up + 1,
                        "soft": bool(soft),
                        "gap": abs(agent.Q[t, up, soft, 0]
                                   - agent.Q[t, up, soft, 1]),
                    })
    return sorted(out, key=lambda d: d["gap"])