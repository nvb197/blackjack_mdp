# Does reinforcement learning actually work on blackjack?

That question is usually unanswerable. An RL agent produces a policy, the
policy looks sensible, and there is nothing to check it against — so "it works"
means "the loss curve went down" and nobody can say by how much it is wrong.

Blackjack dealt with replacement is the rare case where the question is
**decidable**. It is a finite Markov decision process, small enough to solve
exactly by dynamic programming: the transition probabilities come from the card
distribution rather than from simulation, so the only error is floating-point
and the stopping threshold. (Tightening that threshold from 1e-9 to 1e-12 moves
the value function by 6e-12.) That gives a ground-truth `Q*` to hold a learned
`Q` against.

So the project is really one experiment run four times, each time asking the
same question of a harder problem:

1. **Tabular Q-learning against `Q*`.** Off-policy TD control, 200 states, no
   knowledge of the rules. Does it converge to the optimum, and if not, what
   does the gap cost — in money, not in "percentage of cells that agree"?
2. **Does the step size matter more than the algorithm?** A convergence
   plateau that turned out to be bias rather than noise, and what diagnosing
   it required.
3. **What happens when the state stops being sufficient.** A six-deck shoe
   makes the process non-stationary; a Hi-Lo count is a lossy compression of
   the state. Learning over the compressed state produced a clean, publishable
   result — which turned out to be a **data leak**, and did not survive the
   fix.
4. **From a value function to a bet size.** Kelly on a measured edge, and the
   statistics needed to decide when an edge is real enough to size on at all.

The short answers, spelled out in
[the verdict](#so-does-it-work) below: **yes** for the tabular case, to within
2.47 basis points; **no** for the count-augmented case, at any sample size this
project can reach.

```
159 tests    ~659 lines of logic    1,844 lines of tests
```

The source is 2,168 lines, but most of that is documentation: stripping
docstrings and comments leaves roughly 650 lines of logic. The problem is
small. The point was never the size.

---

## Contents

- [Quick start](#quick-start)
- [So does it work?](#so-does-it-work)
- [The ground truth everything is measured against](#the-ground-truth-everything-is-measured-against)
- [Checking the answer](#checking-the-answer)
- [What the mismatches cost depends on which ones they are](#what-the-mismatches-cost-depends-on-which-ones-they-are)
- [Two things I got wrong first](#two-things-i-got-wrong-first)
- [Phase 3: a finite shoe, and card counting](#phase-3-a-finite-shoe-and-card-counting)
- [Phase 3b: index plays, and a finding that did not survive](#phase-3b-index-plays-and-a-finding-that-did-not-survive)
- [Phase 4: how much to bet](#phase-4-how-much-to-bet)
- [Defects found by review, and what they changed](#defects-found-by-review-and-what-they-changed)
- [A leak that invalidated Phase 3b, and four smaller defects](#a-leak-that-invalidated-phase-3b-and-four-smaller-defects)
- [Design notes](#design-notes)
- [Figures](#figures)
- [Layout](#layout)
- [Limitations](#limitations)

---

## Quick start

Requires **Python 3.10 or later** — the code uses PEP 604 union syntax
(`np.random.Generator | None`), which 3.9 does not parse.

```bash
pip install -r requirements.txt

pytest -m "not slow"     # 158 tests, about 8 seconds
pytest                   # 159, adds the 1M-hand convergence check
```

Each command below reproduces one section of this README:

```bash
python main.py dp --chart --double     # exact solution and strategy chart
python main.py ql --episodes 5000000   # Q-learning against that solution
python main.py omega                   # step-size exponent sweep
python main.py count --hands 3000000   # edge by true count
python main.py risk --paths 400        # Kelly sizing and risk metrics
python main.py figures                 # regenerate all five figures
```

Every number in this document is the output of one of those commands, on the
seeds committed in the code.

---

## So does it work?

Four questions, four answers, all measured rather than asserted.

### 1. Does tabular Q-learning reach the optimum? — Yes, to 2.47 bps

Five million hands of off-policy TD control, starting from a table of zeros and
knowing nothing about card probabilities or the dealer's rule:

| | |
|---|---|
| mean squared error against `V*` | 6.5 × 10⁻⁵ |
| decisions matching `π*` | 98.5% (197 of 200) |
| **cost of the remaining mismatches** | **2.47 bps** |

The last line is what "works" should mean. Three cells still disagree — but
evaluating the learned policy exactly (the same Bellman backup without the max)
prices that disagreement at 2.47 basis points against a house edge of 242. The
agent recovers about 99% of the available value from sampled rewards alone.

### 2. Does the step size matter? — More than expected

With the conventional α = n^−0.6, the error plateaued and only 96% of decisions
matched. The diagnosis mattered more than the fix: **every mismatch went the
same way**, and a one-sided error is bias, not variance. No amount of extra
training would have helped. Moving to α = n^−1 — the exact running sample mean,
still satisfying Robbins–Monro — improved the error ninefold.

The sweep also revealed something the theory does not predict: the curve
**flattens and slightly reverses** at ω = 1, because at that value the earliest
updates keep full weight forever, and their targets were bootstrapped from an
empty table. That is a TD-specific effect; Monte Carlo averaging would not show
it.

### 3. Does it still work when the state is no longer sufficient? — No

A six-deck shoe makes the process non-stationary. Keeping it Markov would need
the full shoe composition in the state — about 7.4 × 10¹⁶ states — so the
standard move is to compress it into one number, the Hi-Lo true count, and
learn over that instead. It is a lossy compression: the count cannot
distinguish which small card left the shoe.

Learning over the augmented state (200 → 1600 states, 8 million hands) produced
a result that looked like a success — the agent appeared to recover a known
index play at exactly the published threshold. It was an artefact of **a data
leak in the state representation**: the dealer's face-down card was being
folded into the count before the agent decided, changing the observed bin on
19.9% of hands. The agent was not learning to count. It was learning to read
through the back of a card.

Rerun without the leak, the finding disappears. Of 46 deviations, 13 point the
wrong way; the largest deviation in the entire run is one of them. The
direction reproduces where data is thick (18 of the 20 best-supported cells)
and stops where it is thin.

**The honest answer is that this scale of experiment cannot resolve the
question.** The quantities being estimated are around 0.0006 to 0.02; the
estimator's noise floor, established in step 1, is about 0.007. More hands do
not fix a two-order-of-magnitude mismatch.

### 4. Does a value function turn into a bet size? — Yes, with statistics attached

Kelly sizing on the measured edge, evaluated out of sample:

| | return on capital wagered | t |
|---|---|---|
| flat betting, must bet every hand | −0.857% | −8.32 |
| half Kelly, allowed to decline bad counts | **+1.165%** | **+3.51** |

But the significance gate refuses **7 of 8** count bins: an edge estimated from
a finite sample is mostly noise, and Kelly applied to a noisy estimate does not
merely add variance — it systematically overbets. Deciding when *not* to act on
a learned value is as much of the problem as learning it.

### The pattern across all four

RL worked where the state was genuinely sufficient and the signal was well
above the noise floor. It failed where the state was a lossy compression and
the signal sat below it — and in that second case it failed **while producing
output that looked like success**, until a data leak was found by someone
reading the code from outside.

That is not a blackjack result. It is the ordinary failure mode of applied RL,
made visible here only because an exact answer existed to check against.

---

## The ground truth everything is measured against

Value iteration on the infinite-deck MDP, 200 decision states, no discounting:

| | computed | published |
|---|---|---|
| Value iteration converges in | 13 sweeps | |
| Expected return, hit/stand only | **−2.421%** | −2.421% |
| Expected return, doubling allowed | **−1.087%** | −1.087% |
| Value of the option to double | +1.334% | |

γ = 1 costs the Bellman operator its contraction property, so convergence does
not follow from the usual fixed-point argument. It follows instead from the
episode structure: every hit strictly increases the player's total, which is
bounded at 21, so no policy can play forever. That makes this a stochastic
shortest-path problem with every policy proper, where the fixed point is still
unique. Thirteen sweeps to a sup-norm change below 1e-9.

Matching published figures to five significant figures is what licenses using
this as ground truth. Two further independent checks follow.

---

## Checking the answer

The expected value is a number that could be wrong in ways that still look
reasonable, so it is checked three independent ways:

1. **Against published figures.** −2.421% and −1.087% for these rules, to five
   significant figures.
2. **Against simulation.** The solver computes transition probabilities by hand
   and never simulates; the environment simulates and never uses those
   probabilities. Playing the optimal policy for 300,000 hands lands within
   three standard errors of the exact value. An error would have to appear in
   both, the same way, independently.
3. **Against a model-free method.** Q-learning starts from nothing but sampled
   rewards and arrives at the same value function.

A fourth check appears in Phase 3: the edge measured over a finite shoe
aggregates to −1.074%, against −1.087% computed exactly here — 1.3 bps apart,
inside one standard error, from two calculations that share no code.

---

## What the mismatches cost depends on which ones they are

Three seeds, five million hands each:

| seed | matching | cost | cells that disagree |
|---|---|---|---|
| 42 | 98.5% | 2.47 bps | soft 18 v A, soft 18 v 2, hard 12 v 6 |
| 7 | 99.0% | **0.44 bps** | hard 16 v 10, hard 12 v 4 |
| 2024 | 98.5% | 1.90 bps | soft 18 v 5, hard 12 v 4, soft 18 v 4 |

The count of wrong cells barely moves; the cost moves by a factor of **5.6**.
Seed 7 happens to miss two of the cheapest cells on the chart — hard 16 against
a ten, where the exact gap between standing and hitting is 0.0006, and hard 12
against a four, where it is 0.0025 — so being wrong there is almost free.

That is the argument for reporting basis points: "three cells wrong" is
compatible with anything from 0.44 to 2.47 bps, so the cell count does not
answer the question anyone actually has.

It is **not** an argument that the basis-point figure is more stable. An
earlier version of this analysis, run before the step-size definition was
corrected, appeared to show exactly that. The pattern did not survive the
correction, and quoting it would have been reading a coincidence as a finding.
Both metrics are noisy at this sample size. One of them is in units that mean
something.

---

## Two things I got wrong first

**The dealer's peek.** When the dealer shows an ace or a ten they check for
blackjack before the player acts, so every decision is made conditional on the
dealer not having one. The hole-card distribution has to be renormalised by
that event. My first version did not, which shifted the expected value by 0.16
percentage points — from −1.087% to −1.249% — while still producing a
strategy chart that looked exactly like the published one. Being wrong by a
visible amount while looking fine is what makes it worth a comment in the code.

**The step size.** With the usual choice α = (1+N)^−0.6 the error plateaued and
only 96% of decisions matched. What made it diagnosable was that every mismatch
went the same way — the agent hit where the optimum stands. A one-sided error
is bias, not noise, so this was not a matter of training longer.

The step size α_n = n^−ω averages over an effective window of about n^ω
samples, so residual noise scales like σn^−ω/2 and the mean squared error like
n^−ω. Sweeping ω at a fixed budget of one million hands:

| ω | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|
| MSE at 1M hands | 2.0e−3 | 9.0e−4 | 4.2e−4 | **2.2e−4** | 2.3e−4 |

ω = 1 gives α = 1/n, the exact sample mean, and still satisfies Robbins–Monro
since Σ1/n diverges while Σ1/n² does not. Roughly a nine-fold improvement over
ω = 0.6.

![omega sweep](figures/omega.png)

Note the curve **flattens and slightly reverses** at the top: ω = 0.9 and
ω = 1.0 are indistinguishable. There is a reason. At ω = 1 every sample carries
equal weight forever, including the earliest ones — whose targets were
bootstrapped from a Q table that was still essentially empty. Slightly ω < 1
decays that early contamination away. Pure Monte Carlo averaging of terminal
rewards would show no such effect, so this is specifically a consequence of
bootstrapping.

**A caveat on testing that law.** It is tempting to check n^−ω by
extrapolating across the table above — raising ω from 0.6 to 1.0 at n = 10⁶
should divide the error by 10⁶·⁰·⁴, about 400. It does not; the measured
improvement is about 9. The law describes decay in n at a **fixed** ω and says
nothing about the prefactor, which also depends on ω. Tested properly —
following the error in n at fixed ω = 1 — the fitted slope of log MSE against
log n is **−1.09, against −1.00 predicted**. That is the confirmation; the
sweep is a budget comparison, not a test of the law.

---

## Phase 3: a finite shoe, and card counting

With cards drawn with replacement the edge is identical on every hand, so there
is nothing to bet on. A six-deck shoe dealt without replacement changes that: as
cards leave, what remains gets richer or poorer in the cards that favour the
player, and a Hi-Lo running count normalised by decks remaining ("true count")
summarises that in one number.

Playing the *fixed* Phase 1 strategy against the shoe for 3 million hands,
recording each outcome under the count that preceded the deal:

| true count | hands | edge | 95% interval |
|---|---|---|---|
| < −3 | 262,112 | −3.429% | [−3.861, −2.998] |
| −3..−2 | 202,822 | −2.211% | [−2.698, −1.724] |
| −2..−1 | 373,115 | −1.890% | [−2.248, −1.533] |
| −1..0 | 567,873 | −0.973% | [−1.262, −0.684] |
| 0..1 | 796,606 | −0.942% | [−1.186, −0.699] |
| 1..2 | 352,521 | −0.404% | [−0.769, −0.039] |
| 2..3 | 193,357 | **+0.267%** | [−0.222, +0.757] |
| ≥ 3 | 251,594 | **+0.890%** | [+0.463, +1.317] |

The edge is monotone across all eight bins and crosses zero near a true count of
+2. Aggregated it comes to **−1.074%**, against **−1.087%** computed exactly in
Phase 1 — 1.3 bps apart, well inside one standard error.

![edge by count](figures/edge_by_count.png)

### The bug this was designed to avoid

A bet is placed before any card of the hand is visible, so the only count that
may be used for sizing is the one that preceded the deal. Reading the count
*after* the hand's cards are out, and attributing the outcome to it, uses
information that did not exist at decision time — look-ahead bias, the same
error as a backtest that reads tomorrow's prices.

It raises no exception. It produces a smooth, convincing, entirely fictitious
edge curve. The defence is structural rather than disciplinary: the environment
snapshots the count as the first statement of `reset()`, before dealing
anything, so a caller cannot obtain the wrong one. A second, subtler version —
sizing on the stale count of a shoe that has just reached penetration and is
about to be shuffled — is handled by `pre_deal_true_count()`.

Both are pinned by tests. I also checked those tests have teeth by
reintroducing each bug by hand and confirming tests turn red — but that check
was run locally and is not part of the repository, so take it as a claim about
my process rather than something you can reproduce from a clone.

---

## Phase 3b: index plays, and a finding that did not survive

This section reported a clean result, then a leak was found upstream of it, and
rerunning without the leak took the result away. Both versions are shown,
because the difference is the most useful thing here.

Adding the count bin to the state grows it from 200 to 1600 decision states.
Training 8 million hands and listing every cell where the count-aware policy
departs from the infinite-deck one:

| | deviations |
|---|---|
| with the hole card counted at the deal (the leak) | 58 |
| **with the hole card deferred until the hand ends** | **46** |

The leak manufactured about a dozen deviations. More importantly, it moved the
one that mattered.

### What was claimed

> The top row is one of the best-known index plays in the counting literature:
> stand on 16 against a ten once the true count reaches zero. The agent found
> it from nothing but sampled rewards, **at the threshold the published tables
> give.**

That rested on `hard 16 vs 10` deviating in the `0..1` bin, with 68,681 hands
behind it — the most-supported deviation in the whole table.

### What the rerun shows

`hard 16 vs 10`, every bin, after the fix. Basic strategy says hit:

| true count | learned | hands |
|---|---|---|
| < −3 | hit | 27,425 |
| −3..−2 | hit | 20,627 |
| **−2..−1** | **stand** ← deviates | 36,825 |
| −1..0 | hit | 58,729 |
| **0..1** | **hit** | **67,777** |
| 1..2 | stand ← deviates | 32,045 |
| 2..3 | stand ← deviates | 17,616 |
| ≥ 3 | stand ← deviates | 22,460 |

**The `0..1` bin went back to hitting.** The threshold moved up one bin, so the
claim that the agent recovered the published threshold was an artefact of the
leak, not a finding.

Worse, a deviation appeared at `−2..−1` — standing on 16 against a ten at a
*negative* count, which is backwards. That cell holds 36,825 hands, so it
cannot be dismissed as a thin tail. It is simply noise sitting in a
well-sampled cell, which is what happens when the quantity being resolved
(exact gap 0.0006) is two orders of magnitude below the estimator's noise
floor (about 0.007, established in Phase 2).

### What did survive

Counting theory predicts a direction: switch **towards standing** as the count
rises, **towards hitting** as it falls. Classifying all 46 deviations against
that rule, rather than picking a few that agree:

| | count |
|---|---|
| point the predicted way | 33 |
| **point the wrong way** | **13 (28%)** |
| among the 20 best-supported cells | **18 of 20 correct** |

So the direction reproduces **where there is enough data**, and stops
reproducing where there is not. That is the honest summary, and it is weaker
than the one this section originally gave.

An earlier draft of this paragraph claimed the direction was "intact across
every well-sampled hand" and listed four hands that agreed. Four hands chosen
after seeing the results is not evidence; the table above is what checking all
46 actually shows. Two of the twenty best-supported cells still point the wrong
way, including the largest deviation in the whole run.

(The classification is crude near zero: a bin spanning 0..1 is counted as
"high", so a hand deviating there is scored against the rule even though the
shoe is close to neutral. Both of the two wrong-way entries in the top twenty
sit in bins adjacent to zero.)

### The honest conclusion

**The direction of the index plays reproduces. The thresholds do not resolve at
this sample size, and should not be quoted.** The least-visited cell holds 113
hands against a median of 3,181; even a cell with 36,825 hands produced a
deviation pointing the wrong way. Separating a 0.0006 gap from a 0.007 noise
floor is not a matter of a few more million hands — it needs a different
estimator, or the exact conditional DP that Phase 3 explains is computationally
out of reach.

Reporting the direction and refusing to report the threshold is the whole
lesson of this section. The earlier version did the opposite, and it took an
outside reader finding a data leak for that to become visible.

---

## Phase 4: how much to bet

Kelly maximises the expected logarithm of wealth, because gains compound
multiplicatively. For a payoff `X` per unit staked the growth-optimal fraction
is `f* = mu / E[X^2]` — not `(bp−q)/b`, which is the two-outcome special case
and silently drops the 1.5 and ±2 payoffs blackjack actually has.

That formula is **second order in f**, not the exact maximiser of
E[log(1+fX)], which has no closed form here. Measured on the ≥3 count bin
(mu = 0.00890, E[X²] = 1.19474) the Taylor value is 0.00745 against a numerical
optimum of 0.00746 — it **under**-bets by 0.1%. The direction follows from the
sign of the third moment: E[X³] = +0.213 here, positively skewed by the +1.5
natural and the +2 double, so the next Taylor term raises the optimum. A
negatively skewed payoff would reverse that, which is why "the second-order
formula over-bets on skewed distributions" is only right half the time.

**The significance gate refuses 7 of 8 bins.** Only the ≥3 bin has a lower
confidence bound above zero. The 2..3 bin has a positive point estimate
(+0.267%) but an interval straddling zero, so it is not bet on. An edge measured
from a finite sample is partly noise, and Kelly applied to a noisy estimate does
not merely add variance — it systematically overbets.

**Where the loss comes from matters more than where the edge is.** The two
positive bins contribute +0.09% between them; the six negative bins take
−1.17%. So the decisive question is not how hard to bet the good counts but
whether you must play the bad ones:

| | median final | mean drawdown |
|---|---|---|
| flat 1 unit, must bet every hand | 959.0 | 11.5% |
| half Kelly, still must bet every hand | 962.4 | 14.4% |
| half Kelly, allowed to sit out | **1017.3** | 9.0% |
| full Kelly, allowed to sit out | **1027.8** | 17.3% |

These are the exact output of `python main.py risk --paths 400`.

Bet variation alone, with a compulsory minimum bet, does not overcome the house
edge under these rules. Being able to decline the bad counts does.

**But final bankroll is the wrong metric for that claim, and it took an
outside eye to notice.** A strategy that sits out 92% of hands wagers a third
as much money. In a game that is negative-expectation most of the time,
wagering less trivially loses less — so a higher final bankroll proves nothing
on its own. The question is whether the money that *was* put at risk earned a
return.

Measured over 250 paths, return on capital actually wagered. These come from a
separate run — 250 paths rather than 400, and means rather than medians,
because a ratio of two running totals is not something a median of final
bankrolls can be read off. The two tables agree where they overlap: a mean P&L
of −42.87 is a mean final bankroll of 957, against the 959 median above.

| | mean P&L | mean wagered | **return on capital** | t |
|---|---|---|---|---|
| flat 1 unit, must bet | −42.87 | 5,000 | **−0.857%** | −8.32 |
| half Kelly, sit out | +20.36 | 1,581 | **+1.165%** | **+3.51** |
| full Kelly, sit out | +40.42 | 3,182 | **+0.948%** | **+2.85** |

This is the figure that settles it. The sit-out strategies are not merely
losing less by playing less: the capital they deploy earns a positive return,
significantly so, and the +1.165% is consistent with the +0.890% edge measured
for the one bin they bet (the intervals overlap).

A caution that comes with it: at 60 paths the same measurement gave −0.076%,
inside noise of the 250-path figure but of the opposite sign. Anything read off
a few dozen paths here is not a result.

Full Kelly grows faster than half Kelly and has nearly double the drawdown —
the expected ordering, and if it were absent that would signal a bug.

![bankroll](figures/bankroll.png)

### Separating betting from playing

Changing the sizing rule and the playing strategy together and observing an
improvement tells you nothing about which caused it. Running all four
combinations on identical seeds — so every configuration sees the same cards in
the same order — makes the two effects separable:

| mean final bankroll | flat bet | Kelly bet |
|---|---|---|
| fixed strategy | 956.4 (baseline) | 1017.9 |
| count strategy | 960.8 | 1021.0 |

Because the seeds are shared, the paths can be **paired**, which removes most of
the path-to-path variance and is far more powerful than comparing the two
distributions independently:

| effect | paired difference | standard error | t |
|---|---|---|---|
| bet sizing | **+61.45** | 4.80 | **+12.80** |
| count-dependent play | +4.41 | 2.53 | +1.74 |

Bet sizing is unambiguous. Count-dependent play is a small positive effect that
**this sample size cannot resolve** — t = 1.74 does not clear the conventional
threshold, and separating it from zero would need roughly four times as many
paths.

Two methodological notes, both mistakes made first and then corrected:

*Use the paired mean, not the medians.* Comparing medians of the four
distributions independently gave the playing effect as −2.0 — the wrong sign.
The median is a robust summary of one distribution but a noisy estimator of a
difference between two, and here it inverted the conclusion.

*Do not compute a percentage split from a non-significant term.* An earlier
version reported "97% from betting, 3% from playing" against the roughly 90/10
quoted by Griffin and Wong. That figure was not defensible: a share cannot be
taken of a quantity indistinguishable from zero, and the denominator was built
from an estimate whose own sign was uncertain. The honest statement is the
direction and the significance — betting dominates, by a margin this experiment
resolves clearly — with the split left unquoted.

### Where the theory does not apply, and why

The closed-form risk of ruin under fractional Kelly, `P(touch x) =
x^((2−λ)/λ)`, predicts 12.5% for half Kelly and 50% for full Kelly. Measured
ruin here is 0% in every configuration.

That is not a bug — the approximation simply does not hold. It assumes
continuous time, infinitely divisible bets, a known edge, and an infinite
horizon. Here the horizon is 5,000 hands, only about 8% of which are bet above
the minimum, and Kelly asks for well under 1% of bankroll even then. Total
exposure is far too small for the bankroll to approach half its starting value;
reaching the threshold would be roughly a three-sigma event. Reporting the gap
and its cause is more informative than quietly omitting the comparison.

CVaR is quoted with a bootstrap interval — VaR 144, CVaR 173 with a 95%
interval of [143, 188] — because a 99% tail statistic estimated from 400 paths
rests on about four observations, and a point estimate alone would overstate the
precision.

---

## Defects found by review, and what they changed

Every one of these was found *after* the test suite was green, and none of them
crashed anything. They are recorded rather than quietly fixed because the
pattern is the interesting part: the dangerous errors in quantitative code are
the ones that produce plausible numbers.

**The step size was not the sample mean it claimed to be.** The code used
α = (1 + N)^−ω with N counting visits from 1, so the first visit took α = ½ and
the estimate became half the observed reward. Unrolling the recursion gives
Q_n = [n/(n+1)] × (sample mean): permanently shrunk toward the initial zero.
The fix is α = N^−ω.

Worse, a unit test asserted the buggy value, so the suite was **protecting** the
bug rather than catching it. That test now asserts the correct value, alongside
two new ones checking that a single sample gives back that sample and that
repeated updates converge on the arithmetic mean.

**CVaR was understated whenever probability mass sat exactly at VaR.**
Averaging every loss at-or-above VaR is correct only for a continuous
distribution. With 99 break-even paths and one wipeout at 100, VaR at 90% is 0,
so that average sweeps in all 100 paths and returns 1.0 — where the worst 10%
(nine zeros and one 100) averages 10.0, ten times larger. Discrete payoffs make
this ordinary, not exotic. The fix uses the exact expected-shortfall identity.

**A read-only query mutated the agent.** `greedy_policy()` broke ties using the
agent's own generator, so evaluating the agent at training checkpoints perturbed
the training itself: the final Q table depended on how often it had been
inspected (measured maximum divergence 0.48). An observer effect in the middle
of a reproducibility guarantee is worse than the cosmetic bias it was fixing —
which had itself been measured at 0.000000 bps. The tie-break now uses a
locally seeded generator.

**A test asserted a coincidence as though it were an invariant.** The
convergence test required every disagreeing cell to have |Q(stand) − Q(hit)|
below 0.05. That held for one step-size schedule and one seed. It is not a
property of the game: soft 18 against a 2 has an exact gap of 0.0588, so any run
that misses that cell fails a threshold the game itself violates. The test now
bounds the aggregate cost in basis points, which is the claim that actually
matters.

**Four docstring figures did not match measurement.** The peek renormalisation
was described as shifting EV by "about 0.3%" (measured: 0.16 percentage
points); naturals as ending "roughly 8%" of hands (measured: 9.1%); the
per-hand standard deviation in a test constant as 1.14 (measured under optimal
hit/stand play: 0.984 — 1.14 is the figure for the game *with* doubling, which
that test does not play, so its bound was 16% looser than it looked); and the
shoe's state count as exceeding the number of seconds since the Big Bang (it is
6× smaller — the honest comparison is 590 petabytes of storage).

---

## A leak that invalidated Phase 3b, and four smaller defects

An outside review found five more defects after everything above was written.
One of them was serious enough to invalidate a whole section of results.

**The dealer's hole card was counted before the player decided.** Every card
was folded into the running count the moment it left the shoe — including the
dealer's face-down card. The count an agent observed therefore encoded the one
card a real player cannot see. Measured over 3,000 hands, the count **bin** the
agent saw was changed by the hole card on **19.9% of them**.

An agent trained on that state is not learning index plays. It is learning that
a lower-than-expected count means the hole card was high, so the dealer is
strong — reading through the back of a card.

Phase 3b has since been rerun without the leak. The headline finding did not
survive: the deviation the section was built on, `hard 16 vs 10` at a true
count of `0..1` with 68,681 hands behind it, went back to hitting, and the
claim that the agent had recovered the published threshold turned out to be an
artefact. The direction of the index plays survived; the thresholds did not.
See that section for the side-by-side.

Nothing crashed. No test failed. The edge curve looked entirely reasonable.
This is the clearest example in the project of the pattern the section above
describes, and it survived seven rounds of review before an outside reader
found it.

The fix is structural: `draw_hole()` takes the card out of the shoe without
publishing it, and `_reveal_hole()` folds it into the count when the hand ends.
The infinite-deck environment inherits no-op defaults, so Phase 1 and 2 are
untouched — and their numbers are unchanged, which is the check that the fix
did not reach further than intended.

**Four smaller defects, all latent:**

*The standard error divided by n rather than n−1.* `ex2 - mean²` is the
population variance; the unbiased estimate divides by n−1. The gap is 0.1% at
n = 1000 but **41% at n = 2**, and it understates the error — the direction
that lets noise through the significance gate.

*Two hands could open the significance gate.* With n = 2 and both hands won,
the measured variance is exactly zero, so the standard error is zero and the
lower confidence bound equals the mean. The interval test passes and Kelly goes
to its cap on the strength of two hands. The confidence interval rests on the
central limit theorem, which says nothing at n = 2. Fixed with a hard floor of
30 observations — a convention, not a number this project derived, and far
below the ~38,000 per bin actually needed to resolve a 1% edge.

*Growth rate was reported over survivors only.* `np.median(growth[survivors])`
discards ruined paths before taking the median. With three paths wiped out and
one tripled, the honest median is −∞; the filtered version reports the
survivor's growth. Textbook survivorship bias. Now reported over every path,
with the survivor-only figure kept beside it for diagnosis and never quoted
alone.

*A bankroll could place a double it could not fund.* `play_hand` did not know
the bankroll, so a path holding 1.5 units could double a 1-unit stake and lose
2.0 — clamped to zero, with the shortfall silently absorbed. That is
uncollateralised credit, and it flatters both drawdown and risk of ruin. At a
starting bankroll of 1000 it never fires; start at 2 units and it fires on
about 7% of doubles.

All five are pinned by regression tests, and all five were verified by
mutation: reintroducing each one turns a test red. The first attempt at the
double-funding test did **not** catch its mutant — it exercised the parameter
rather than the caller — which is itself a reminder that a test passing is not
the same as a test working.

### The question none of the earlier reviews asked

Seven rounds of review missed the hole-card leak. Every one of them asked
*does the code compute what it claims?* None asked *what is this
decision-maker allowed to know at the moment it decides?*

That second question now has its own tests. Every decision point in the
project is enumerated and its information set checked against what a real
player would hold:

| decision | may know | checked |
|---|---|---|
| bet sizing | completed hands only | `test_bet_sizing_sees_only_completed_hands` |
| playing, count-aware | + own cards, dealer upcard | `test_the_playing_decision_never_sees_the_hole_card` |
| after a hit | + the card just drawn | `test_a_card_the_player_draws_is_immediately_public` |

The third is the mirror image of the first two and matters just as much:
withholding information the player *does* have is as wrong as supplying
information they do not.

That audit found one further defect — introduced by the hole-card fix itself.
Deferring the card meant an abandoned hand (reset called twice without playing
out) left it pending, the next deal overwrote the slot, and the card left the
shoe without ever entering the count. Five abandoned hands were enough for the
count to drift. `reset()` now publishes any pending card first.

The same audit also cleared a suspected sixth defect. The count bin an agent
sees correlates with the Hi-Lo value of the hole card (0.24 bins between
extremes), which looks exactly like residual leakage. It is not: the same
correlation, 0.20 bins, is present in the **pre-deal** count, which cannot
contain any information about a card not yet dealt. A ten-rich shoe both
raises the count and makes the hole card more likely to be a ten. That
correlation is the mechanism card counting runs on, not a bug — and reporting
it as one would have been the same error in the opposite direction.

---

## Design notes

**The step size decays per state-action pair, not on a global clock.** Visit
frequencies are very uneven — measured 135× between the most and least visited
cell of the decision region. On a global clock the step size would collapse at
rare states that have barely been updated, breaking the Robbins–Monro condition
exactly where the estimates are worst.

**Exploration decays to a floor, not to zero.** This keeps every pair reachable.
It is safe only because Q-learning is off-policy: the max in the target means it
learns the value of the greedy policy whatever the behaviour policy does. SARSA
under the same schedule would converge to the value of the ε-greedy policy.

**The shoe is an explicit 312-card array, not a count vector.** Dealing a card
that was never in the shoe becomes structurally impossible rather than merely
detectable. Prefer designs where a bug cannot happen over designs where it can
be caught.

**Every random source is created from a seed and passed explicitly.** Nothing
touches the global numpy state, so a run is reproducible from its seed alone.
Every command in the quick start was run on both Windows (3.12.8) and Linux
(3.12.3) and compared digit by digit — including the 5-million-hand training
run and the 400-path bankroll simulation. That is strong evidence, not a
guarantee: only these commands on these two platforms were compared.

---

## Figures

![convergence](figures/convergence.png)

Value error and decision agreement against training length, with the n^−1
reference line the step size predicts.

![policy](figures/policy.png)

The exact and learned charts, and beside them the exact gap between the value of
standing and of hitting. The cells that disagree sit in the dark region of the
third panel, where the gap is smallest.

---

## Layout

```
blackjack/
  rules.py            card distribution, hand arithmetic, seeded generators
  hand.py             blackjack rules shared by both environments
  env.py              infinite-deck simulator
  dp.py               dealer distribution, value iteration, policy evaluation
  qlearning.py        tabular agent, training loop, comparison against V*
  shoe.py             six-deck shoe, dealt without replacement
  counting.py         Hi-Lo, true count, look-ahead-safe payoff tracker
  finite_env.py       finite-shoe simulator, count exposed safely
  qlearning_count.py  agent over the count-augmented state
  simulate.py         playing hands, measuring edge, bankroll paths
  sizing.py           Kelly fraction, fractional Kelly, significance gate
  risk.py             VaR, CVaR, drawdown, risk of ruin, bootstrap
  plots.py            the five figures

tests/                159 tests
  test_rules.py             13    card distribution and hand arithmetic
  test_dp.py                16    exact solution, including a 300k-hand cross-check
  test_qlearning.py         15    update rule, convergence, side-effect freedom
  test_shoe.py              15    composition, conservation, reshuffling
  test_counting.py          38    Hi-Lo, true count, information sets at each decision
  test_kelly.py             30    Kelly formula, sizing gate, risk measures
  test_qlearning_count.py   18    count-augmented agent
  test_simulate.py          14    bet sizing, bankroll mechanics, double funding

main.py               command-line entry point
```

---

## Limitations

Splitting is not implemented, which is worth roughly +0.6% of expected value, so
−1.087% is not the full house edge for these rules.

`main.py` has no tests. It is the only layer without coverage, and it is exactly
where one defect slipped through: a stale function signature that made
`main.py risk` crash while all 117 tests stayed green.

Index-play thresholds do not resolve at 8 million hands and are not quoted.
Only the direction is reported; see Phase 3b for why, including a deviation
pointing the wrong way in a cell holding 36,825 hands.

The risk-of-ruin comparison against the closed form is not a meaningful test at
this horizon, for the reasons given above.

Wonging — declining to bet at unfavourable counts — is modelled as staking zero
while the hand is still dealt. A real player must also avoid being noticed doing
it, which is outside the scope of anything measurable here.