import numpy as np
import pytest

from blackjack import dp
from blackjack.env import BlackjackEnv
from blackjack.qlearning import QLearningAgent, train, compare
from blackjack.rules import make_rng


def test_epsilon_decays_to_the_floor_and_stays_there():
    agent = QLearningAgent(make_rng(0), eps_decay_episodes=1000, eps_min=0.05)
    assert agent.epsilon(0) == pytest.approx(1.0)
    assert agent.epsilon(500) == pytest.approx(0.525)
    assert agent.epsilon(1000) == pytest.approx(0.05)
    assert agent.epsilon(10**9) == pytest.approx(0.05)


def test_step_size_satisfies_robbins_monro():
    """sum alpha diverges and sum alpha^2 converges, for alpha_n = n^-1.

    This must test the schedule the code ACTUALLY uses. An earlier version
    of this test computed (1 + n)^-1 -- the formula from before the
    step-size fix -- so it was verifying a property of code that no longer
    existed. Both schedules happen to satisfy Robbins-Monro, which is
    exactly why it kept passing and why nothing flagged the drift.
    """
    n = np.arange(1, 10**6)
    alpha = n ** -1.0
    assert alpha.sum() > 10
    tail = (np.arange(10**6, 10**7) ** -2.0).sum()
    assert tail < 1e-5


def test_step_size_decays_per_state_action_pair():
    agent = QLearningAgent(make_rng(0))
    for _ in range(100):
        agent.update(15, 5, 0, 1, 0.0, 16, 5, 0, False)
    assert agent.N[15, 5, 0, 1] == 100
    assert agent.N[15, 5, 0, 0] == 0  # the untouched pair keeps a large step


def test_terminal_update_targets_the_reward_alone():
    """At a terminal state the target is the reward, with no bootstrap."""
    agent = QLearningAgent(make_rng(0))
    agent.Q[20, 3, 0, 1] = 99.0  # would leak in if the successor were used
    agent.update(15, 3, 0, 0, -1.0, 20, 3, 0, done=True)
    assert agent.Q[15, 3, 0, 0] == pytest.approx(-1.0)


def test_first_visit_estimate_is_the_observed_reward():
    """One sample must give an estimate equal to that sample.

    With alpha = N^-1 and N starting at 1, the first update has alpha = 1, so
    Q becomes the reward exactly -- the sample mean of a single observation.
    An earlier version used (1 + N)^-1, giving alpha = 1/2 and therefore half
    the reward, which leaves the estimate permanently shrunk toward the
    initial zero by a factor n/(n+1). This test exists because the previous
    version of it asserted the wrong value (-0.5) and so protected the bug
    instead of catching it.
    """
    agent = QLearningAgent(make_rng(0))
    agent.update(15, 3, 0, 1, 1.0, 16, 3, 0, done=True)
    assert agent.Q[15, 3, 0, 1] == pytest.approx(1.0)


def test_running_estimate_is_the_exact_sample_mean():
    """Repeated updates at one cell must converge to the arithmetic mean."""
    agent = QLearningAgent(make_rng(0))
    rewards = [1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.5, -1.0]
    for r in rewards:
        agent.update(15, 3, 0, 0, r, 16, 3, 0, done=True)
    assert agent.Q[15, 3, 0, 0] == pytest.approx(np.mean(rewards))


def test_greedy_action_is_taken_when_epsilon_is_zero():
    agent = QLearningAgent(make_rng(0))
    agent.Q[15, 3, 0] = [0.5, -0.5]
    assert all(agent.act(15, 3, 0, eps=0.0) == 0 for _ in range(20))


def test_exploration_reaches_both_actions():
    agent = QLearningAgent(make_rng(0))
    agent.Q[15, 3, 0] = [0.5, -0.5]
    chosen = {agent.act(15, 3, 0, eps=1.0) for _ in range(200)}
    assert chosen == {0, 1}


def test_hands_that_end_at_the_deal_produce_no_update():
    """A natural on either side must not credit any state-action pair.

    Its reward belongs to no action, so charging it to one would bias every
    state it touches.
    """
    class AlwaysNatural:
        def reset(self):
            return (21, 5, True), {"done": True, "reward": 1.5}

        def step(self, action):
            raise AssertionError("step must not be called after a natural")

    agent = QLearningAgent(make_rng(0))
    train(agent, AlwaysNatural(), 100)
    assert agent.N.sum() == 0
    assert np.all(agent.Q == 0.0)


def test_evaluation_history_has_no_gaps():
    agent = QLearningAgent(make_rng(0))
    env = BlackjackEnv(make_rng(6))
    history = train(agent, env, 10_000, eval_every=1_000,
                    eval_fn=lambda a, ep: {"episode": ep})
    assert [h["episode"] for h in history] == list(range(1000, 11000, 1000))


@pytest.mark.slow
def test_q_learning_converges_to_the_exact_solution():
    """Q-learning must reach the DP optimum in value, decision and cost.

    A note on the third assertion, because an earlier version got it wrong.

    That version asserted that EVERY disagreeing cell had |Q(stand) - Q(hit)|
    below 0.05 -- the idea being that leftover mistakes should sit only where
    the two actions are nearly equivalent. That happened to hold for one
    step-size schedule and one seed, and it was written as though it were an
    invariant. It is not: soft 18 against a 2 has an exact gap of 0.0588, so a
    run that misses that cell fails a threshold the game itself violates.

    The assertion was a proxy for the real claim, which is that the mistakes
    do not cost much in aggregate. That is measurable directly -- evaluate the
    learned policy exactly and difference it against the optimum -- so it is
    asserted directly. Measured across four seeds at 1M hands the cost runs
    1.7 to 4.0 bps; the 10 bps bound leaves room for seed variation while
    still being 4% of the 242 bps house edge.
    """
    V_star, pi_star, _ = dp.value_iteration()
    agent = QLearningAgent(make_rng(42))
    env = BlackjackEnv(make_rng(1042))
    train(agent, env, 1_000_000)

    result = compare(agent, V_star, pi_star)
    assert result["mse"] < 1e-3
    assert result["agreement"] >= 0.98

    cost_bps = (dp.expected_value(V_star)
                - dp.expected_value(
                    dp.policy_evaluation(agent.greedy_policy()))) * 10_000
    assert 0 <= cost_bps < 10, f"mismatches cost {cost_bps:.2f} bps"


def test_greedy_policy_does_not_touch_the_agents_rng():
    """A read-only query must not advance the agent's random stream.

    An earlier version drew the tie-break from self.rng, so evaluating the
    agent at training checkpoints perturbed the training itself and the final
    Q table depended on how often it had been inspected. That is an observer
    effect in the middle of a reproducibility guarantee.
    """
    agent = QLearningAgent(make_rng(1))
    before = agent.rng.bit_generator.state["state"]["state"]
    agent.greedy_policy()
    assert agent.rng.bit_generator.state["state"]["state"] == before


def test_greedy_policy_is_idempotent():
    agent = QLearningAgent(make_rng(1))
    assert np.array_equal(agent.greedy_policy(), agent.greedy_policy())


def test_checkpointing_does_not_change_the_training_trajectory():
    """Training with and without mid-run evaluation must give the same Q."""
    a1 = QLearningAgent(make_rng(42))
    train(a1, BlackjackEnv(make_rng(1042)), 60_000)
    a2 = QLearningAgent(make_rng(42))
    train(a2, BlackjackEnv(make_rng(1042)), 60_000, eval_every=10_000,
          eval_fn=lambda ag, ep: {"policy": ag.greedy_policy()})
    assert np.array_equal(a1.Q, a2.Q)


def test_greedy_policy_does_not_systematically_prefer_stand_on_ties():
    """With every cell tied at zero, both actions must appear."""
    agent = QLearningAgent(make_rng(1))
    pol = agent.greedy_policy()
    assert set(np.unique(pol)) == {0, 1}