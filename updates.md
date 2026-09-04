# Running the multi-objective optimizer tests

These tests exercise the two-objective cost vector `(latency, monetary)`, Pareto pruning (exact and \((1+\varepsilon)\)-approximate), and the final pick (min latency subject to a budget, or a weighted sum). They use **synthetic cost vectors**, not a live Spark/Java job.

Requires **Java 17** and Maven. From this directory (`wayang/`):

```bash
mvn -pl wayang-commons/wayang-core -am test \
  -Dtest=VectorCostTest,ParetoFrontTest,CombinationSweepTest,MultiObjectiveSimulationTest,TrummerKochCorrectnessTest,HybridMultiPlatformPipelineTest \
  -Dsurefire.failIfNoSpecifiedTests=false
```

| Class | What it checks |
|---|---|
| `VectorCostTest` | Dominance, \(\alpha\)-dominance, log-buckets |
| `ParetoFrontTest` | Exact front drops dominated plans; bucketing shrinks the set |
| `MultiObjectiveSimulationTest` | Prints Pareto sets while \(\varepsilon\), budget, weights, and platform rates change |
| `CombinationSweepTest` | Grid of \(\varepsilon \times B\), weights, and Spark `$/`ms |
| `TrummerKochCorrectnessTest` | Paper invariants: \(\alpha\)-coverage, weighted \(\rho \le \alpha\), additive prune-after-concat, Fig. 8 bounded gap |
| `HybridMultiPlatformPipelineTest` | 5 engines × 5 operators (+ diamond join); hybrid must beat uniform/greedy and match exhaustive search |

To print the sweep tables, Surefire must not hide stdout (default Maven output already shows `System.out` from these tests).

Enable the vector optimizer in a real job via `wayang-core-defaults.properties` (`cost.model=vector`, `ParetoPruningStrategy`, `objectives.epsilon`). Set `wayang.java.costs.per-ms` vs `wayang.spark.costs.per-ms` so the two axes actually conflict, and optionally `wayang.core.optimizer.objectives.budget`.

---

## Result of the last run (4 Sep 2026)

**20 tests, 0 failures, BUILD SUCCESS** (~6 s).

- **Exact Pareto** always drops `dominated (90, 90)`. The front is cheap-slow → java → hybrid → spark → fast-expensive.
- **Budget** walks that front: \(B=10\) cheap-slow, \(B=20\) java, \(B=50\) hybrid, \(B=80\) spark, unconstrained → fastest.
- **Weights:** \((1,0)\) fastest, \((0,1)\) cheapest, \((1,1)\) hybrid.
- **Coarser \(\varepsilon\):** default \(0.1\) matches exact; \(\varepsilon=1\) merges spark with fast-expensive (front size 5 → 4); \(\varepsilon=2\) leaves 3 plans.
- **Spark rate:** same runtimes, \(B=40\): cheap Spark stays Spark; expensive Spark switches to Postgres.
- **Trummer–Koch:** weighted pick from the \(\alpha\)-front stays within factor \(\alpha\) of the exhaustive optimum, including 4-stage **additive** DP. A hard bound can miss that guarantee (Fig. 8); IRA is not implemented. Wayang `pick` still falls back to cheapest money if \(B\) is infeasible.

Enumeration is unchanged; only prune/score after concatenate. Costs are assumed **monotonic and additive**.
