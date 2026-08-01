"""Tests for qlearning_count.py.

This module also had no tests. It repeats several pieces of QLearningAgent
with an extra state dimension, and repetition is where a fix applied to one
copy quietly fails to reach the other. In particular `greedy_policy` here
carries the same side-effect-free requirement that cost this project a real
reproducibility bug in the infinite-deck agent -- so it is pinned here too,
rather than assumed to have been copied correctly.
"""

import numpy as np
import pytest

from blackjack.counting import N_BINS
from blackjack.qlearning_count import (CountQLearningAgent, index_deviations,
                                       train_count)
from blackjack.rules import make_rng


# ------------------------------------------------------------------ #
# Shape and construction
# ------------------------------------------------------------------ #
def test_q_table_carries_the_count_dimension():
    """State is (total, upcard, soft, tc_bin), so Q must be five-dimensional.

    Catches the case where N_BINS changes in counting.py but the agent is
    not rebuilt around it.
    """
    agent = CountQLearningAgent(make_rng(0))
    assert agent.Q.shape == (22, 10, 2, N_BINS, 2)
    assert agent.N.shape == agent.Q.shape
    assert np.all(agent.Q == 0.0)
    assert np.all(agent.N == 0)


def test_decision_region_is_eight_times_the_infinite_deck_one():
    """200 decision states become 1600 once the count enters the state."""
    agent = CountQLearningAgent(make_rng(0))
    decision_cells = agent.Q[12:22].shape[:-1]
    assert int(np.prod(decision_cells)) == 200 * N_BINS


# ------------------------------------------------------------------ #
# The update rule -- same schedule as the infinite-deck agent
# ------------------------------------------------------------------ #
def test_first_visit_estimate_is_the_observed_reward():
    """alpha = N^-1 with N starting at 1 means the first update has alpha = 1.

    The same check as in test_qlearning.py, repeated because this agent has
    its own copy of the update rule. A fix applied to one and not the other
    is exactly the failure this file guards against.
    """
    agent = CountQLearningAgent(make_rng(0))
    agent.update(15, 3, 0, 4, 1, 1.0, 16, 3, 0, 4, done=True)
    assert agent.Q[15, 3, 0, 4, 1] == pytest.approx(1.0)


def test_running_estimate_is_the_exact_sample_mean():
    agent = CountQLearningAgent(make_rng(0))
    rewards = [1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.5, -1.0]
    for r in rewards:
        agent.update(15, 3, 0, 4, 0, r, 16, 3, 0, 4, done=True)
    assert agent.Q[15, 3, 0, 4, 0] == pytest.approx(np.mean(rewards))


def test_terminal_update_targets_the_reward_alone():
    """No bootstrap at a terminal state, even across count bins."""
    agent = CountQLearningAgent(make_rng(0))
    agent.Q[20, 3, 0, 6, 1] = 99.0      # would leak in if the successor were used
    agent.update(15, 3, 0, 4, 0, -1.0, 20, 3, 0, 6, done=True)
    assert agent.Q[15, 3, 0, 4, 0] == pytest.approx(-1.0)


def test_bins_are_updated_independently():
    """A visit in one count bin must not touch the same hand in another.

    This is the whole point of adding the dimension: hard 16 against a ten
    is allowed to have a different answer at a high count than at a low one.
    """
    agent = CountQLearningAgent(make_rng(0))
    agent.update(16, 9, 0, 7, 0, 1.0, 17, 9, 0, 7, done=True)
    assert agent.Q[16, 9, 0, 7, 0] == pytest.approx(1.0)
    assert agent.Q[16, 9, 0, 3, 0] == 0.0
    assert agent.N[16, 9, 0, 3, 0] == 0


def test_step_size_decays_per_cell_not_globally():
    agent = CountQLearningAgent(make_rng(0))
    for _ in range(50):
        agent.update(15, 5, 0, 4, 1, 0.0, 16, 5, 0, 4, False)
    assert agent.N[15, 5, 0, 4, 1] == 50
    assert agent.N[15, 5, 0, 5, 1] == 0     # neighbouring bin untouched


# ------------------------------------------------------------------ #
# Exploration schedule
# ------------------------------------------------------------------ #
def test_epsilon_decays_to_the_floor_and_stays_there():
    agent = CountQLearningAgent(make_rng(0), eps_decay_episodes=1000,
                                eps_min=0.05)
    assert agent.epsilon(0) == pytest.approx(1.0)
    assert agent.epsilon(500) == pytest.approx(0.525)
    assert agent.epsilon(1000) == pytest.approx(0.05)
    assert agent.epsilon(10 ** 9) == pytest.approx(0.05)


def test_exploration_reaches_both_actions():
    agent = CountQLearningAgent(make_rng(0))
    agent.Q[15, 3, 0, 4] = [0.5, -0.5]
    chosen = {agent.act(15, 3, 0, 4, eps=1.0) for _ in range(200)}
    assert chosen == {0, 1}


def test_greedy_action_is_taken_when_epsilon_is_zero():
    agent = CountQLearningAgent(make_rng(0))
    agent.Q[15, 3, 0, 4] = [0.5, -0.5]
    assert all(agent.act(15, 3, 0, 4, eps=0.0) == 0 for _ in range(20))


# ------------------------------------------------------------------ #
# greedy_policy must stay side-effect free -- the bug that already happened
# ------------------------------------------------------------------ #
def test_greedy_policy_does_not_touch_the_agents_rng():
    """A read-only query must not advance the agent's random stream.

    In the infinite-deck agent an earlier version drew the tie-break from
    self.rng, so inspecting the agent at training checkpoints perturbed the
    training itself. This agent carries a copy of that code; the guarantee
    has to be pinned separately or the fix could regress here alone.
    """
    agent = CountQLearningAgent(make_rng(1))
    before = agent.rng.bit_generator.state["state"]["state"]
    agent.greedy_policy()
    assert agent.rng.bit_generator.state["state"]["state"] == before


def test_greedy_policy_is_idempotent():
    agent = CountQLearningAgent(make_rng(1))
    assert np.array_equal(agent.greedy_policy(), agent.greedy_policy())


def test_greedy_policy_does_not_systematically_prefer_stand_on_ties():
    agent = CountQLearningAgent(make_rng(1))
    assert set(np.unique(agent.greedy_policy())) == {0, 1}


def test_greedy_policy_follows_the_larger_value():
    agent = CountQLearningAgent(make_rng(1))
    agent.Q[16, 9, 0, 7] = [1.0, -1.0]      # stand better
    agent.Q[16, 9, 0, 0] = [-1.0, 1.0]      # hit better
    pol = agent.greedy_policy()
    assert pol[16, 9, 0, 7] == 0
    assert pol[16, 9, 0, 0] == 1


# ------------------------------------------------------------------ #
# Training loop
# ------------------------------------------------------------------ #
def test_hands_that_end_at_the_deal_produce_no_update():
    """A natural belongs to no action, so it must credit no state-action pair."""
    class AlwaysNatural:
        def reset(self):
            return (21, 5, True, 4), {"done": True, "reward": 1.5}

        def step(self, action):
            raise AssertionError("step must not be called after a natural")

    agent = CountQLearningAgent(make_rng(0))
    train_count(agent, AlwaysNatural(), 100)
    assert agent.N.sum() == 0
    assert np.all(agent.Q == 0.0)


# ------------------------------------------------------------------ #
# Reporting
# ------------------------------------------------------------------ #
def test_index_deviations_reports_nothing_when_the_policies_agree():
    """An untrained agent ties everywhere, so build a policy that matches it."""
    agent = CountQLearningAgent(make_rng(1))
    pi = agent.greedy_policy()
    # Collapse the count dimension: pi_star has no bins, so take bin 0 and
    # make every bin agree with it.
    pi_star = pi[..., 0]
    for b in range(N_BINS):
        agent.Q[..., b, :] = 0.0
    agent.Q[..., 1] = np.where(pi_star[..., None] == 1, 1.0, -1.0)
    agent.Q[..., 0] = np.where(pi_star[..., None] == 1, -1.0, 1.0)
    assert index_deviations(agent, pi_star) == []


def test_index_deviations_finds_a_planted_disagreement():
    """Plant one deviation and check it is reported with its visit count."""
    agent = CountQLearningAgent(make_rng(1))
    pi_star = np.zeros((22, 10, 2), dtype=np.int64)      # stand everywhere
    agent.Q[..., 0] = 1.0                                 # agent stands too
    agent.Q[..., 1] = -1.0
    agent.Q[16, 9, 0, 7] = [-1.0, 1.0]                    # except here: hit
    agent.N[16, 9, 0, 7] = [3, 4]

    out = index_deviations(agent, pi_star)
    assert len(out) == 1
    d = out[0]
    assert (d["total"], d["upcard"], d["soft"], d["bin"]) == (16, 10, False, 7)
    assert d["from"] == "S" and d["to"] == "H"
    assert d["visits"] == 7


def test_index_deviations_are_sorted_by_evidence():
    """Most-visited deviations first, so undersampled ones cannot lead."""
    agent = CountQLearningAgent(make_rng(1))
    pi_star = np.zeros((22, 10, 2), dtype=np.int64)
    agent.Q[..., 0] = 1.0
    agent.Q[..., 1] = -1.0
    agent.Q[16, 9, 0, 7] = [-1.0, 1.0]
    agent.N[16, 9, 0, 7] = [1, 1]
    agent.Q[15, 4, 0, 2] = [-1.0, 1.0]
    agent.N[15, 4, 0, 2] = [50, 50]

    visits = [d["visits"] for d in index_deviations(agent, pi_star)]
    assert visits == sorted(visits, reverse=True)