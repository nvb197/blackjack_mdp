"""Command-line entry point.

    python main.py dp --chart --double
    python main.py ql --episodes 5000000
    python main.py omega
    python main.py count --hands 3000000
    python main.py risk --paths 400
    python main.py figures
"""

import argparse
import time

import numpy as np

from blackjack import dp
from blackjack.env import BlackjackEnv
from blackjack.qlearning import QLearningAgent, train, compare, disagreements
from blackjack.rules import make_rng

# Published figures for these rules, used as a regression check.
REFERENCE = {"hit_stand": -0.02421, "with_double": -0.01087}


def run_dp(chart: bool, double: bool) -> None:
    V, pi, sweeps = dp.value_iteration()
    ev0 = dp.expected_value(V, allow_double=False)
    ev1 = dp.expected_value(V, allow_double=True)

    print(f"value iteration converged in {sweeps} sweeps")
    print(f"{'':<24}{'computed':>12}{'published':>12}")
    print(f"{'EV, hit/stand only':<24}{ev0:>12.5f}"
          f"{REFERENCE['hit_stand']:>12.5f}")
    print(f"{'EV, doubling allowed':<24}{ev1:>12.5f}"
          f"{REFERENCE['with_double']:>12.5f}")
    print(f"{'value of the double':<24}{(ev1 - ev0) * 100:>11.3f}%")

    if chart:
        dp.print_policy(V, pi, allow_double=double)


def run_ql(n_episodes: int) -> None:
    V_star, pi_star, _ = dp.value_iteration()
    agent = QLearningAgent(make_rng(42))
    env = BlackjackEnv(make_rng(1042))

    print(f"training on {n_episodes:,} hands")
    t0 = time.perf_counter()
    train(agent, env, n_episodes)
    print(f"done in {time.perf_counter() - t0:.0f}s\n")

    result = compare(agent, V_star, pi_star)
    cost = dp.expected_value(V_star) - dp.expected_value(
        dp.policy_evaluation(agent.greedy_policy()))

    print(f"{'mean squared value error':<28}{result['mse']:>12.2e}")
    print(f"{'largest value error':<28}{result['max_err']:>12.2e}")
    print(f"{'matching decisions':<28}{result['agreement']:>11.1%}")
    print(f"{'cost of the mismatches':<28}{cost * 10000:>9.2f} bps")

    rows = disagreements(agent, pi_star)
    if rows:
        print(f"\n{len(rows)} cells disagree, value gap between actions:")
        for r in rows:
            kind = "soft" if r["soft"] else "hard"
            print(f"  {kind} {r['total']} vs {r['upcard']}: {r['gap']:.4f}")


def run_omega() -> None:
    """Sweep the step-size exponent. See the note in the README."""
    V_star, pi_star, _ = dp.value_iteration()
    print(f"{'omega':>7}{'mse':>12}{'agreement':>12}")
    for omega in (0.6, 0.7, 0.8, 0.9, 1.0):
        agent = QLearningAgent(make_rng(42), omega=omega)
        train(agent, BlackjackEnv(make_rng(1042)), 1_000_000)
        r = compare(agent, V_star, pi_star)
        print(f"{omega:>7.1f}{r['mse']:>12.2e}{r['agreement']:>11.1%}")


def run_count(n_hands: int) -> None:
    """Measure the player edge against the Hi-Lo true count."""
    from blackjack.simulate import measure_edge
    V, _, _ = dp.value_iteration()
    inf_ev = dp.expected_value(V, allow_double=True)
    print(f"playing basic strategy for {n_hands:,} hands against a 6-deck shoe")
    tr = measure_edge(n_hands, seed=42, allow_double=True)
    print(f"\n{'bin':>8} {'hands':>10} {'edge %':>9} {'95% CI':>20}")
    for s in tr.all_stats():
        if s["n"] == 0:
            continue
        print(f"{s['label']:>8} {s['n']:>10,} {s['mean']*100:>8.3f}% "
              f"[{s['ci_low']*100:>7.3f}, {s['ci_high']*100:>7.3f}]")
    overall = sum(x["mean"] * x["n"] for x in tr.all_stats() if x["n"]) / tr.total()
    print(f"\n{'overall finite-shoe edge':<28}{overall*100:>9.4f} %")
    print(f"{'infinite-deck EV (Phase 1)':<28}{inf_ev*100:>9.4f} %")
    print(f"{'difference':<28}{(overall-inf_ev)*10000:>7.2f} bps")


def run_risk(n_paths: int, n_hands: int) -> None:
    """Compare bet-sizing rules on identical seeds and report risk metrics."""
    from blackjack.simulate import measure_edge, bankroll_paths
    from blackjack.sizing import FlatSizer, KellySizer
    from blackjack import risk

    print("calibrating the edge table (out-of-sample seeds used for evaluation)")
    tr = measure_edge(3_000_000, seed=42, allow_double=True)
    initial = 1000.0
    rng = np.random.default_rng(7)
    print(f"\n{n_paths} paths x {n_hands:,} hands, bankroll {initial:.0f}\n")
    print(f"{'strategy':>22} {'median':>9} {'VaR99':>8} {'CVaR99':>8} "
          f"{'MDD':>7} {'ruin':>7} {'theory':>7}")
    rows = [
        ("flat, must bet 1", FlatSizer(1.0), None),
        ("half Kelly, must bet", KellySizer(tr, lam=0.5), 0.5),
        ("half Kelly, sit out", KellySizer(tr, lam=0.5, min_bet=0.0), 0.5),
        ("full Kelly, sit out", KellySizer(tr, lam=1.0, min_bet=0.0), 1.0),
    ]
    for name, sizer, lam in rows:
        p = bankroll_paths(sizer, n_paths, n_hands, initial, seed=100)
        r = risk.summarise(p, initial, rng=rng)
        th = f"{risk.theoretical_ruin(lam)*100:5.1f}%" if lam else "    -"
        print(f"{name:>22} {r['median_final']:>9.1f} {r['var']:>8.1f} "
              f"{r['cvar']:>8.1f} {r['mdd_mean']:>6.1%} {r['ruin']:>6.1%} {th:>7}")


def run_figures() -> None:
    """Regenerate every figure in the README. Takes a few minutes."""
    import os
    from blackjack import plots

    os.makedirs("figures", exist_ok=True)
    V_star, pi_star, _ = dp.value_iteration()
    q_stand, q_hit = dp.action_values(V_star)
    gaps = np.abs(q_stand - q_hit)

    print("training with checkpoints ...")
    agent = QLearningAgent(make_rng(42))
    history = train(agent, BlackjackEnv(make_rng(1042)), 2_000_000,
                    eval_every=25_000,
                    eval_fn=lambda a, ep: {"episode": ep,
                                           **compare(a, V_star, pi_star)})
    plots.convergence(history, "figures/convergence.png")
    plots.policy_heatmaps(pi_star, agent.greedy_policy(), gaps,
                          "figures/policy.png")

    print("sweeping the step-size exponent ...")
    omegas = [0.6, 0.7, 0.8, 0.9, 1.0]
    mses = []
    for omega in omegas:
        a = QLearningAgent(make_rng(42), omega=omega)
        train(a, BlackjackEnv(make_rng(1042)), 1_000_000)
        mses.append(compare(a, V_star, pi_star)["mse"])
        print(f"  omega={omega}  mse={mses[-1]:.2e}")
    plots.omega_sweep(omegas, mses, "figures/omega.png")

    print("measuring edge by true count (a few minutes) ...")
    from blackjack.simulate import measure_edge, bankroll_paths
    from blackjack.sizing import KellySizer
    from blackjack import risk
    tr = measure_edge(3_000_000, seed=42, allow_double=True)
    plots.edge_by_count(tr.all_stats(),
                        dp.expected_value(V_star, allow_double=True),
                        "figures/edge_by_count.png")

    print("simulating bankroll paths ...")
    sizer = KellySizer(tr, lam=0.5, min_bet=0.0)
    paths = bankroll_paths(sizer, 400, 5000, 1000.0, seed=100)
    r = risk.summarise(paths, 1000.0, rng=make_rng(7))
    plots.bankroll_fan(paths, 1000.0, r["var"], r["cvar"],
                       "figures/bankroll.png")
    print("written to figures/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase",
                    choices=["dp", "ql", "omega", "count", "risk", "figures"])
    ap.add_argument("--hands", type=int, default=3_000_000)
    ap.add_argument("--paths", type=int, default=400)
    ap.add_argument("--episodes", type=int, default=5_000_000)
    ap.add_argument("--chart", action="store_true")
    ap.add_argument("--double", action="store_true")
    args = ap.parse_args()

    if args.phase == "dp":
        run_dp(args.chart, args.double)
    elif args.phase == "ql":
        run_ql(args.episodes)
    elif args.phase == "omega":
        run_omega()
    elif args.phase == "count":
        run_count(args.hands)
    elif args.phase == "risk":
        run_risk(args.paths, 5000)
    else:
        run_figures()