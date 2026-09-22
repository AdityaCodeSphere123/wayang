# Multi-objective cost in Apache Wayang

Base path: `wayang-commons/wayang-core/`  
`PlanEnumerator.java` is **unchanged**. Only cost, prune, and final pick changed. Costs are treated as **monotonic and additive**. Channel conversion is still a scalar shortest path.

---

## What was implemented

Two-objective vector \((c_1, c_2)\) = (latency, monetary). After each concatenate, pruning keeps an exact or \((1+\varepsilon)\)-approximate Pareto set of partial plans that share the same open interface. At the end, one plan is picked: min latency subject to money \(\le B\) (or a weighted sum).

---

## New files

| File | What it implements |
|---|---|
| `src/main/java/.../optimizer/costs/VectorCost.java` | Cost vector `(latency, monetary)`: Pareto dominance, \(\alpha\)-dominance, log-buckets for \((1+\varepsilon)\) coarsening. |
| `src/main/java/.../optimizer/costs/VectorEstimatableCost.java` | Cost model: \(c_1\) from Wayang time estimates, \(c_2\) from existing `TimeToCostConverter` (`wayang.<platform>.costs.per-ms` / `costs.fix`). Final pick via `VectorPlanSelection`. |
| `src/main/java/.../optimizer/enumeration/ParetoFront.java` | Exact 2-D Pareto (sort by latency, scan decreasing money). If \(\varepsilon>0\), keep one plan per log-bucket first (Trummer–Koch RTA). |
| `src/main/java/.../optimizer/enumeration/ParetoPruningStrategy.java` | Default prune hook: same grouping as latent-operator pruning (platforms + open operators), keep a Pareto set instead of one scalar winner. |
| `src/main/java/.../optimizer/enumeration/VectorPlanSelection.java` | `pick`: min latency s.t. money \(\le B\); infeasible \(B\) falls back to cheapest money. `selectBest`: paper SelectBest (weighted cost; ignore bounds if none fit). |

---

## Edited files

| File | What changed |
|---|---|
| `src/main/java/.../optimizer/costs/EstimatableCost.java` | Default `getVectorEstimate()` maps the old scalar onto both axes so other cost models still compile. |
| `src/main/java/.../optimizer/enumeration/PlanImplementation.java` | `getVectorCostEstimate()` delegates to the configured cost model. |
| `src/main/java/.../optimizer/enumeration/LatentOperatorPruningStrategy.java` | If `wayang.core.optimizer.objectives.vector=true`, keep a Pareto set per group; otherwise the old single best plan. |
| `src/main/java/.../api/Configuration.java` | `wayang.core.optimizer.cost.model` (`default` / `vector`); `bootstrapCostModel` / `applyCostModel` after properties load. |
| `src/main/resources/wayang-core-defaults.properties` | `pruning.strategies = ParetoPruningStrategy`, `cost.model=vector`, `objectives.vector=true`, `epsilon=0.1`, latency/monetary weights; optional `objectives.budget`. |

Removed: `HardcodedPlatformPrices.java` (invented $/ms). Money uses Wayang’s converters.

---

## Running the tests

Synthetic vectors only — not a live Spark/Java job. Java 17 + Maven, from this directory (`wayang/`):

```bash
mvn -pl wayang-commons/wayang-core -am test \
  -Dtest=VectorCostTest,ParetoFrontTest,CombinationSweepTest,MultiObjectiveSimulationTest,TrummerKochCorrectnessTest,HybridMultiPlatformPipelineTest \
  -Dsurefire.failIfNoSpecifiedTests=false
```

| Test | What it checks |
|---|---|
| `.../costs/VectorCostTest.java` | Dominance, \(\alpha\)-dominance, log-buckets |
| `.../enumeration/ParetoFrontTest.java` | Exact front drops dominated plans; bucketing shrinks the set |
| `.../enumeration/MultiObjectiveSimulationTest.java` | Prints Pareto sets while \(\varepsilon\), budget, weights, and platform rates change |
| `.../enumeration/CombinationSweepTest.java` | Grid of \(\varepsilon \times B\), weights, and Spark `$/`ms |
| `.../enumeration/TrummerKochCorrectnessTest.java` | Paper invariants: \(\alpha\)-coverage, weighted \(\rho \le \alpha\), additive prune-after-concat, Fig. 8 bounded gap |
| `.../enumeration/HybridMultiPlatformPipelineTest.java` | 5 engines × 5 operators (+ diamond join); hybrid must beat uniform/greedy and match exhaustive search |

For a real job, keep the defaults above. Set `wayang.java.costs.per-ms` vs `wayang.spark.costs.per-ms` so the two axes conflict, and optionally `wayang.core.optimizer.objectives.budget`.

---

## Result of the last run (4 Sep 2026)

**20 tests, 0 failures, BUILD SUCCESS** (~6 s). Hybrid pipeline tests were added after that run.

- **Exact Pareto** drops `dominated (90, 90)`. Front: cheap-slow → java → hybrid → spark → fast-expensive.
- **Budget** walks that front: \(B=10\) cheap-slow, \(B=20\) java, \(B=50\) hybrid, \(B=80\) spark, unconstrained → fastest.
- **Weights:** \((1,0)\) fastest, \((0,1)\) cheapest, \((1,1)\) hybrid.
- **Coarser \(\varepsilon\):** `0.1` matches exact; \(\varepsilon=1\) front size 5 → 4; \(\varepsilon=2\) leaves 3.
- **Spark rate:** \(B=40\): cheap Spark stays Spark; expensive Spark switches to Postgres.
- **Trummer–Koch:** weighted pick from the \(\alpha\)-front stays within factor \(\alpha\) on additive DP. A hard bound can miss that (Fig. 8); IRA is not implemented.

---

## New updates (18 Sep 2026)

Corner cases in scoring, α-Pareto bucketing, and the final pick are fixed. New tests compare exact Pareto DP (`ε = 0`) with α-Pareto on chains, diamonds, bushy trees, fan-in, and a branching DAG.

### Corner cases that were wrong

- A broken time estimate (`NaN` / `Inf`) was stored as **0**, so that plan looked free and could wipe the front. It is now **+∞**, so it loses.
- Tiny `ε` made `log(1+ε) = 0` and blew up bucket ids. Tiny / negative / `NaN` `ε` now means **exact Pareto**.
- Also handled: null plans, duplicate costs, weakly dominated prefixes, budget `0` or negative, and both weights unset (falls back to min latency).

Fixes live in [`VectorCost.java`](wayang-commons/wayang-core/src/main/java/org/apache/wayang/core/optimizer/costs/VectorCost.java), [`ParetoFront.java`](wayang-commons/wayang-core/src/main/java/org/apache/wayang/core/optimizer/enumeration/ParetoFront.java), and [`VectorPlanSelection.java`](wayang-commons/wayang-core/src/main/java/org/apache/wayang/core/optimizer/enumeration/VectorPlanSelection.java). Coverage is in [`MultiObjectiveCornerCaseTest.java`](wayang-commons/wayang-core/src/test/java/org/apache/wayang/core/optimizer/enumeration/MultiObjectiveCornerCaseTest.java), plus extra cases in [`VectorCostTest.java`](wayang-commons/wayang-core/src/test/java/org/apache/wayang/core/optimizer/costs/VectorCostTest.java) and [`ParetoFrontTest.java`](wayang-commons/wayang-core/src/test/java/org/apache/wayang/core/optimizer/enumeration/ParetoFrontTest.java).

### Tests they asked for

[`ExactVsAlphaPlanShapeTest.java`](wayang-commons/wayang-core/src/test/java/org/apache/wayang/core/optimizer/enumeration/ExactVsAlphaPlanShapeTest.java) runs exact Pareto DP (`ε = 0`) against α-Pareto (`ε = 0.1`) on:

- chains (4, 8, 12 ops)
- diamond and two diamonds
- bushy join tree
- wide fan-in
- a 10-op branching DAG

Small graphs are also checked against brute force. Dense cases use 8–12 platforms so the front is actually long.

From this directory:

```bash
mvn -pl wayang-commons/wayang-core -am test -Dtest=ExactVsAlphaPlanShapeTest,MultiObjectiveCornerCaseTest
```

### Trade-off (slide)

On a 12-op chain, 12 platforms (cartesian about \(10^{12}\)):

| ε | concatenations | final subplans | time | latency ratio ρ |
|---|---|---|---|---|
| 0 (exact) | 103k | 134 | 24 ms | 1.00 |
| 0.1 | 20k | 21 | 6 ms | 1.00 |
| 0.25 | 10k | 11 | 2 ms | 1.06 |
| 0.5 | 6k | 5 | 1.4 ms | 1.28 |
| 1.0 | 4k | 3 | 1.1 ms | 1.18 |

Bigger `ε` means fewer subplans, faster search, a slightly worse pick. Default `ε = 0.1` kept the same pick as exact here, with 80% fewer concatenations.

Both DPs already cut the cartesian product down to almost nothing. α-Pareto mainly helps when the exact front is long (chains, sequential diamonds). On wide fan-in, grouping by the cut of all open sources keeps many groups of size 1, so α has little left to prune until the join. That is a real shape effect, not a bug.

Exact DP matched brute force on the small chains, diamond, and bushy tree.
