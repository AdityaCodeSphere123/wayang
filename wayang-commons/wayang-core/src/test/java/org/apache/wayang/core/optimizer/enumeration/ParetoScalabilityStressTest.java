/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.wayang.core.optimizer.enumeration;

import org.apache.wayang.core.optimizer.costs.VectorCost;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Large plans with the same additive DP and grouping as the optimizer: how many partial plans are alive, how long
 * pruning takes, and how far each mode is from the exact front. Opt-in: {@code -Dwayang.stress=true};
 * {@code -Dwayang.stress.only=chains|dags|joins} runs one family.
 *
 * <p>Modes: {@code wayang} = original Wayang (one min-latency plan per group); {@code exact} = Pareto front per group;
 * {@code +xg} = plus exact cross-group pruning ({@link PlanDagSimulator#crossGroupPrune}); {@code +cap} = at most 64
 * plans per group.</p>
 */
@EnabledIfSystemProperty(named = "wayang.stress", matches = "true")
class ParetoScalabilityStressTest {

    private static final int CAP = 64;
    /** Plain exact and the original-Wayang emulation run out of heap beyond this size on random DAGs with 4 platforms. */
    private static final int MAX_PLAIN_DAG = 22;

    @Test
    void largePlans() {
        final String only = System.getProperty("wayang.stress.only", "all");
        final List<PlanDagSimulator.Graph> graphs = new ArrayList<>();
        if (only.equals("all") || only.equals("chains")) {
            for (int n : new int[]{50, 100, 200}) graphs.add(PlanDagSimulator.chain("chain", n, 8));
        }
        if (only.equals("all") || only.equals("dags")) {
            for (int n : new int[]{25, 30, 35, 40}) {
                for (long seed = 1; seed <= 3; seed++) graphs.add(PlanDagSimulator.randomConnected("dag", n, 4, seed));
            }
        }
        if (only.equals("huge")) {
            for (int n : new int[]{30, 35, 40, 50}) {
                for (long seed = 1; seed <= 3; seed++) graphs.add(PlanDagSimulator.randomConnected("dag", n, 4, seed));
            }
        }
        if (only.equals("all") || only.equals("joins")) {
            for (int sources : new int[]{8, 10, 12}) graphs.add(PlanDagSimulator.leftDeep("leftdeep", sources, 4));
        }

        System.out.printf("%-9s %4s %3s | %-14s %9s %10s %10s %6s %7s%n",
                "shape", "ops", "pl", "mode", "time_ms", "peak", "sum", "front", "cover");
        for (PlanDagSimulator.Graph g : graphs) {
            final boolean plainOk = !g.name.equals("dag") || g.nOps() <= MAX_PLAIN_DAG;
            final Run wayang = plainOk ? run(g, -1, false) : null;
            final boolean huge = System.getProperty("wayang.stress.only", "").equals("huge");
            if (huge) {
                final Run approx = run(g, CAP, 0.05);
                System.out.printf("%-9s %4d %3d | %-14s %9d %10d %10d %6d%n", g.name, g.nOps(), g.nPlats,
                        "xg~0.05+cap", approx.ms, approx.result.peakKept, approx.result.keptSum, approx.result.finalKept);
                continue;
            }
            final Run xg = run(g, 0, true);
            final Run exact = plainOk ? run(g, 0, false) : null;
            final Run xgCap = run(g, CAP, true);
            final Run xgApprox = run(g, CAP, 0.05);
            final List<PlanDagSimulator.Partial> reference = xg.result.front;   // exact front (CrossGroupPruningTest)

            if (wayang != null) print(g, "wayang", wayang, reference);
            if (exact != null) print(g, "exact", exact, reference);
            print(g, "exact+xg", xg, reference);
            print(g, "exact+xg+cap", xgCap, reference);
            print(g, "xg~0.05+cap", xgApprox, reference);

            if (exact != null) {
                assertTrue(cover(exact.result.front, xg.result.front) <= 1 + 1e-9, g.name + ": +xg lost a front point");
                assertTrue(cover(xg.result.front, exact.result.front) <= 1 + 1e-9, g.name + ": +xg added a point");
            }
            for (Run r : new Run[]{wayang, xgCap}) {
                if (r == null) continue;
                assertTrue(Math.abs(min(r.result.front) - min(reference)) < 1e-6, g.name + ": fastest drifted");
            }
        }
    }

    private static final class Run {
        PlanDagSimulator.SearchResult result;
        long ms;
    }

    private static Run run(PlanDagSimulator.Graph g, int maxFront, boolean crossGroup) {
        return run(g, maxFront, crossGroup ? 0d : -1d);
    }

    private static Run run(PlanDagSimulator.Graph g, int maxFront, double crossGroupEps) {
        final Run r = new Run();
        final long t = System.nanoTime();
        r.result = PlanDagSimulator.optimize(g, 0d, maxFront, crossGroupEps);
        r.ms = (System.nanoTime() - t) / 1_000_000;
        return r;
    }

    private static void print(PlanDagSimulator.Graph g, String mode, Run r, List<PlanDagSimulator.Partial> reference) {
        System.out.printf("%-9s %4d %3d | %-14s %9d %10d %10d %6d %7.4f%n",
                g.name, g.nOps(), g.nPlats, mode, r.ms, r.result.peakKept, r.result.keptSum, r.result.finalKept,
                cover(reference, r.result.front));
    }

    private static double min(List<PlanDagSimulator.Partial> front) {
        return front.stream().mapToDouble(p -> p.cost.getLatency()).min().orElse(Double.NaN);
    }

    /** Smallest α such that every point of {@code exact} is α-dominated by some point of {@code approx}. */
    static double cover(List<PlanDagSimulator.Partial> exact, List<PlanDagSimulator.Partial> approx) {
        double worst = 1;
        for (PlanDagSimulator.Partial e : exact) {
            double best = Double.POSITIVE_INFINITY;
            for (PlanDagSimulator.Partial a : approx) {
                final VectorCost ac = a.cost, ec = e.cost;
                best = Math.min(best, Math.max(ac.getLatency() / ec.getLatency(), ac.getMonetary() / ec.getMonetary()));
            }
            worst = Math.max(worst, best);
        }
        return worst;
    }
}
