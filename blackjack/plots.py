"""The five figures. Run ``python main.py figures`` to regenerate them."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .qlearning import T_LO, T_HI

UPCARDS = ["A"] + [str(i) for i in range(2, 11)]


def convergence(history: list[dict], path: str) -> None:
    """Value error and decision agreement against training length.

    The reference line is the n^-1 rate predicted for a sample-mean step
    size, drawn so the measured curve can be read against it rather than
    just admired.
    """
    ep = np.array([h["episode"] for h in history])
    mse = np.array([h["mse"] for h in history])
    agree = np.array([h["agreement"] for h in history]) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.loglog(ep, mse, lw=1.6, label="measured")
    ax1.loglog(ep, mse[0] * (ep / ep[0]) ** -1.0, "--", lw=1.0,
               color="grey", label=r"$n^{-1}$ reference")
    ax1.set_xlabel("hands trained")
    ax1.set_ylabel(r"mean squared error against $V^*$")
    ax1.set_title("Value error")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3, which="both")

    ax2.semilogx(ep, agree, lw=1.6)
    ax2.axhline(99, ls="--", lw=1.0, color="grey")
    ax2.set_xlabel("hands trained")
    ax2.set_ylabel("decisions matching $\\pi^*$ (%)")
    ax2.set_title("Decision agreement, 200 states")
    ax2.set_ylim(85, 100.6)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def policy_heatmaps(pi_star: np.ndarray, pi_rl: np.ndarray,
                    gaps: np.ndarray, path: str) -> None:
    """Optimal and learned charts side by side, plus where they can differ.

    The third panel is the point of the figure: it shows |Q(stand) - Q(hit)|
    under the exact solution, so the cells where the two policies disagree
    can be read against how little is at stake there.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    rows = np.arange(T_LO, T_HI + 1)

    for ax, grid, title, cmap in (
            (axes[0], pi_star[T_LO:T_HI + 1, :, 0], "Optimal (hard totals)", "RdYlBu"),
            (axes[1], pi_rl[T_LO:T_HI + 1, :, 0], "Q-learning (hard totals)", "RdYlBu")):
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(10), UPCARDS)
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel("dealer upcard")
        for i in range(len(rows)):
            for j in range(10):
                ax.text(j, i, "H" if grid[i, j] else "S",
                        ha="center", va="center", fontsize=7)
    axes[0].set_ylabel("player total")

    im = axes[2].imshow(gaps[T_LO:T_HI + 1, :, 0], cmap="viridis",
                        aspect="auto", vmin=0, vmax=0.6)
    axes[2].set_title("|Q(stand) - Q(hit)|, exact", fontsize=10)
    axes[2].set_xticks(range(10), UPCARDS)
    axes[2].set_yticks(range(len(rows)), rows)
    axes[2].set_xlabel("dealer upcard")
    fig.colorbar(im, ax=axes[2])

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def omega_sweep(omegas: list[float], mses: list[float], path: str) -> None:
    """Measured error against the step-size exponent, at a fixed budget.

    No theory curve is drawn here, deliberately. The argument that the error
    decays like n^-omega describes the rate in n at a fixed omega; it says
    nothing about the prefactor, which also depends on omega. Extrapolating
    across omega at one fixed n as though the prefactor were constant
    predicts a slope of -6 per unit omega, whereas the measured slope is
    about -2.5. The law is tested properly in the convergence figure, where
    the error is followed in n at a single omega.
    """
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.semilogy(omegas, mses, "o-", lw=1.6)
    slope = np.polyfit(omegas, np.log10(mses), 1)[0]
    ax.set_xlabel(r"$\omega$ in $\alpha_t(s,a) = N(s,a)^{-\omega}$")
    ax.set_ylabel(r"mean squared error against $V^*$")
    ax.set_title(f"Error after 1M hands  (slope {slope:.1f} per unit "
                 r"$\omega$)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def edge_by_count(stats: list[dict], infinite_ev: float, path: str) -> None:
    """Measured edge per true-count bin, with 95% intervals and sample sizes.

    The horizontal line is the infinite-deck expected value from Phase 1.
    Where the measured curve crosses it is where the finite shoe is behaving
    like a neutral one, and that crossing is an independent cross-check: the
    two numbers come from completely different calculations.
    """
    rows = [s for s in stats if s["n"] > 0]
    x = np.arange(len(rows))
    mean = np.array([s["mean"] for s in rows]) * 100
    err = np.array([s["mean"] - s["ci_low"] for s in rows]) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.errorbar(x, mean, yerr=err, fmt="o-", lw=1.6, capsize=4)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axhline(infinite_ev * 100, ls="--", lw=1.0, color="grey",
                label=f"infinite-deck EV ({infinite_ev*100:.3f}%)")
    ax1.set_ylabel("edge, % of bet")
    ax1.set_title("Player edge against the Hi-Lo true count")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    ax2.bar(x, [s["n"] for s in rows], color="grey", alpha=0.6)
    ax2.set_ylabel("hands")
    ax2.set_xticks(x, [s["label"] for s in rows], rotation=45, ha="right")
    ax2.set_xlabel("true count bin")
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bankroll_fan(paths: np.ndarray, initial: float, var_: float, cvar_: float,
                 path: str) -> None:
    """Bankroll trajectories and the distribution of final outcomes.

    The right panel marks VaR and CVaR on the loss distribution so the two
    can be read against each other: VaR is a cutoff, CVaR is the average
    beyond it, and CVaR always sits further out.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    n_show = min(120, paths.shape[0])
    for p in paths[:n_show]:
        ax1.plot(p, lw=0.4, alpha=0.35, color="steelblue")
    ax1.plot(np.median(paths, axis=0), lw=2.0, color="black", label="median")
    ax1.axhline(initial, ls="--", lw=1.0, color="grey")
    ax1.axhline(0.5 * initial, ls=":", lw=1.0, color="crimson",
                label="ruin threshold")
    ax1.set_xlabel("hands played")
    ax1.set_ylabel("bankroll")
    ax1.set_title(f"{paths.shape[0]} bankroll paths")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    losses = initial - paths[:, -1]
    ax2.hist(losses, bins=40, color="steelblue", alpha=0.7)
    ax2.axvline(var_, color="darkorange", lw=1.8, label=f"VaR 99% = {var_:.0f}")
    ax2.axvline(cvar_, color="crimson", lw=1.8, label=f"CVaR 99% = {cvar_:.0f}")
    ax2.set_xlabel("loss at the end (units)")
    ax2.set_ylabel("paths")
    ax2.set_title("Where the tail is")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)