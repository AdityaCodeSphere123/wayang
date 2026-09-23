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

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import static org.apache.wayang.core.optimizer.enumeration.PlanDagSimulator.Graph;
import static org.apache.wayang.core.optimizer.enumeration.PlanDagSimulator.Partial;
import static org.apache.wayang.core.optimizer.enumeration.PlanDagSimulator.SearchResult;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Meeting action items (8 Sep 2026): many plan shapes and sizes, exact Pareto DP
 * vs α-Pareto, time / subplans kept / pruning cut, and the speed–quality trade-off.
 */
class ExtensiveMeetingActionItemsTest {

    private static final double EPS = 0.1d;
    private static final double ALPHA = 1.1d;
    private static final long EXHAUSTIVE_LIMIT = 20_000L;

    @Test
    void exactDpMatchesExhaustiveOnManySmallShapes() {
        List<Graph> graphs = new ArrayList<>();
        for (int n : Arrays.asList(3, 4, 5, 6)) {
            for (int p : Arrays.asList(2, 3, 4)) {
                if (pow(p, n) <= EXHAUSTIVE_LIMIT) {
                    graphs.add(PlanDagSimulator.chain("chain-" + n + "x" + p, n, p));
                }
            }
        }
        graphs.add(PlanDagSimulator.diamond("diamond-2", 2));
        graphs.add(PlanDagSimulator.diamond("diamond-3", 3));
        graphs.add(PlanDagSimulator.diamond("diamond-4", 4));
        graphs.add(PlanDagSimulator.bushy("bushy-2", 2));
        graphs.add(PlanDagSimulator.bushy("bushy-3", 3));
        graphs.add(PlanDagSimulator.fanIn("fan2-3", 2, 3));
        graphs.add(PlanDagSimulator.fanIn("fan3-3", 3, 3));
        graphs.add(PlanDagSimulator.fanOut("fanout-3", 3));
        graphs.add(PlanDagSimulator.skipEdges("skip-5-3", 5, 3));
        graphs.add(PlanDagSimulator.leftDeep("leftdeep-3-2", 3, 2));
        graphs.add(PlanDagSimulator.twoDiamonds("twodiamond-2", 2));
        for (int seed = 0; seed < 12; seed++) {
            Graph g = PlanDagSimulator.randomConnected("rand-" + seed, 5, 3, 1000L + seed);
            if (g.cartesian() <= EXHAUSTIVE_LIMIT) {
                graphs.add(g);
            }
        }

        int checked = 0;
        for (Graph g : graphs) {
            if (g.cartesian() > EXHAUSTIVE_LIMIT) {
                continue;
            }
            SearchResult exact = PlanDagSimulator.optimize(g, 0d);
            SearchResult brute = PlanDagSimulator.exhaustive(g);
            for (double frac : Arrays.asList(0.1d, 0.4d, 0.8d, 2d)) {
                double b = frac > 1d ? Double.POSITIVE_INFINITY : PlanDagSimulator.budgetAt(exact.front, frac);
                Partial dp = PlanDagSimulator.pick(exact.front, b);
                Partial ex = PlanDagSimulator.pick(brute.front, b);
                assertEquals(ex.cost.getLatency(), dp.cost.getLatency(), 1e-9, g.name + " B=" + b);
                assertEquals(ex.cost.getMonetary(), dp.cost.getMonetary(), 1e-9, g.name + " B=" + b);
            }
            checked++;
        }
        assertTrue(checked >= 25, "need a wide exhaustive suite, got " + checked);
        System.out.println("exhaustive vs exact DP: " + checked + " graphs, all picks matched");
    }

    @Test
    void shapeSizeMatrixExactVersusAlpha() {
        List<Graph> graphs = new ArrayList<>();
        for (int n : Arrays.asList(4, 6, 8, 10, 12, 14, 16)) {
            for (int p : Arrays.asList(3, 4, 6, 8, 12)) {
                if (n >= 14 && p >= 8) {
                    continue;
                }
                graphs.add(PlanDagSimulator.chain("chain-" + n + "x" + p, n, p));
            }
        }
        for (int p : Arrays.asList(3, 4, 6, 8, 10)) {
            graphs.add(PlanDagSimulator.diamond("diamond-x" + p, p));
            graphs.add(PlanDagSimulator.twoDiamonds("2dia-x" + p, p));
            graphs.add(PlanDagSimulator.bushy("bushy-x" + p, p));
            graphs.add(PlanDagSimulator.fanOut("fanout-x" + p, p));
            graphs.add(PlanDagSimulator.complexBranching("branch-x" + p, p));
        }
        graphs.add(PlanDagSimulator.threeDiamonds("3dia-x4", 4));
        graphs.add(PlanDagSimulator.threeDiamonds("3dia-x8", 8));
        graphs.add(PlanDagSimulator.skipEdges("skip-10x6", 10, 6));
        graphs.add(PlanDagSimulator.skipEdges("skip-14x4", 14, 4));
        graphs.add(PlanDagSimulator.leftDeep("left-4x4", 4, 4));
        graphs.add(PlanDagSimulator.leftDeep("left-5x3", 5, 3));
        graphs.add(PlanDagSimulator.fanIn("fan-3x4", 3, 4));
        graphs.add(PlanDagSimulator.fanIn("fan-4x4", 4, 4));
        graphs.add(PlanDagSimulator.fanIn("fan-5x4", 5, 4));

        System.out.println();
        System.out.println("=== shape × size matrix, ε=0 vs ε=0.1, B=40th percentile money ===");
        System.out.printf(Locale.ROOT,
                "%-18s %3s %5s %12s %8s %8s %9s %9s %7s %7s %7s %6s%n",
                "shape", "ops", "plats", "cartesian", "ex_ms", "a_ms",
                "ex_gen", "a_gen", "ex_k", "a_k", "cut%", "ρ");

        List<Double> rhos = new ArrayList<>();
        int compared = 0;
        for (Graph g : graphs) {
            Row row = compare(g, EPS, 0.4d);
            rhos.add(row.rho);
            compared++;
            System.out.printf(Locale.ROOT,
                    "%-18s %3d %5d %12s %8.2f %8.2f %9d %9d %7d %7d %6.1f %6.3f%n",
                    row.graph.name, row.graph.nOps(), row.graph.nPlats,
                    cart(row.graph.cartesian()),
                    row.exactNs / 1e6d, row.alphaNs / 1e6d,
                    row.exact.generated, row.alpha.generated,
                    row.exact.finalKept, row.alpha.finalKept,
                    row.cutPct, row.rho);
            assertRow(row, ALPHA);
        }
        summarize("matrix", rhos);
        assertTrue(compared >= 40, "matrix too small: " + compared);
    }

    @Test
    void randomBranchingDagsManySeeds() {
        System.out.println();
        System.out.println("=== 40 random connected DAGs (6–9 ops, 3–6 platforms) ===");
        List<Double> rhos = new ArrayList<>();
        int exhaustiveHits = 0;
        for (int seed = 0; seed < 40; seed++) {
            int n = 6 + (seed % 4);
            int p = 3 + (seed % 4);
            Graph g = PlanDagSimulator.randomConnected("rand-" + seed, n, p, 42L + 17 * seed);
            Row row = compare(g, EPS, 0.4d);
            rhos.add(row.rho);
            assertRow(row, ALPHA);
            if (g.cartesian() <= EXHAUSTIVE_LIMIT) {
                SearchResult brute = PlanDagSimulator.exhaustive(g);
                double b = PlanDagSimulator.budgetAt(row.exact.front, 0.4d);
                Partial dp = PlanDagSimulator.pick(row.exact.front, b);
                Partial ex = PlanDagSimulator.pick(brute.front, b);
                assertEquals(ex.cost.getLatency(), dp.cost.getLatency(), 1e-9, g.name);
                exhaustiveHits++;
            }
        }
        summarize("random DAGs", rhos);
        assertTrue(exhaustiveHits >= 8, "random exhaustive checks: " + exhaustiveHits);
    }

    @Test
    void budgetTightnessOnSeveralShapes() {
        List<Graph> graphs = Arrays.asList(
                PlanDagSimulator.chain("chain-10x8", 10, 8),
                PlanDagSimulator.diamond("diamond-x8", 8),
                PlanDagSimulator.twoDiamonds("2dia-x6", 6),
                PlanDagSimulator.bushy("bushy-x8", 8),
                PlanDagSimulator.complexBranching("branch-x8", 8),
                PlanDagSimulator.skipEdges("skip-12x6", 12, 6)
        );
        System.out.println();
        System.out.println("=== budget tightness (10% / 40% / 80% / inf) at ε=0.1 ===");
        List<Double> rhos = new ArrayList<>();
        for (Graph g : graphs) {
            SearchResult exact = PlanDagSimulator.optimize(g, 0d);
            SearchResult alpha = PlanDagSimulator.optimize(g, EPS);
            for (double frac : Arrays.asList(0.1d, 0.4d, 0.8d)) {
                double b = PlanDagSimulator.budgetAt(exact.front, frac);
                double rho = PlanDagSimulator.rho(
                        PlanDagSimulator.pick(alpha.front, b),
                        PlanDagSimulator.pick(exact.front, b),
                        b);
                rhos.add(rho);
                System.out.printf(Locale.ROOT, "  %-16s  B@%.0f%%  ρ=%.3f  B=%.1f%n",
                        g.name, 100d * frac, rho, b);
                assertTrue(Double.isInfinite(rho) || rho <= Math.pow(ALPHA, g.nOps()) + 1e-6, g.name + " frac=" + frac);
            }
            double rhoInf = PlanDagSimulator.rho(
                    PlanDagSimulator.pick(alpha.front, Double.POSITIVE_INFINITY),
                    PlanDagSimulator.pick(exact.front, Double.POSITIVE_INFINITY),
                    Double.POSITIVE_INFINITY);
            rhos.add(rhoInf);
            System.out.printf(Locale.ROOT, "  %-16s  B=inf   ρ=%.3f%n", g.name, rhoInf);
        }
        summarize("budgets", rhos);
    }

    @Test
    void epsilonGridOnChainDiamondBushyAndBranch() {
        List<Graph> graphs = Arrays.asList(
                PlanDagSimulator.chain("chain-12x12", 12, 12),
                PlanDagSimulator.diamond("diamond-x10", 10),
                PlanDagSimulator.bushy("bushy-x10", 10),
                PlanDagSimulator.complexBranching("branch-x10", 10),
                PlanDagSimulator.twoDiamonds("2dia-x10", 10)
        );
        System.out.println();
        System.out.println("=== ε grid on four shapes ===");
        System.out.printf(Locale.ROOT, "%-16s %6s %8s %8s %8s %8s %6s%n",
                "shape", "ε", "gen", "peak", "final", "cut%", "ρ");
        List<Double> rhos = new ArrayList<>();
        for (Graph g : graphs) {
            SearchResult exact = PlanDagSimulator.optimize(g, 0d);
            double b = PlanDagSimulator.budgetAt(exact.front, 0.4d);
            Partial exactPick = PlanDagSimulator.pick(exact.front, b);
            int lastKept = Integer.MAX_VALUE;
            for (double eps : Arrays.asList(0d, 0.05d, 0.1d, 0.25d, 0.5d, 1d, 2d)) {
                SearchResult r = PlanDagSimulator.optimize(g, eps);
                double rho = PlanDagSimulator.rho(PlanDagSimulator.pick(r.front, b), exactPick, b);
                rhos.add(rho);
                double cut = 100d * (1d - (double) r.generated / Math.max(1, exact.generated));
                System.out.printf(Locale.ROOT, "%-16s %6.2f %8d %8d %8d %7.1f %6.3f%n",
                        g.name, eps, r.generated, r.peakKept, r.finalKept, cut, rho);
                assertTrue(r.finalKept <= exact.finalKept + 2, g.name + " ε=" + eps);
                assertTrue(Double.isInfinite(rho) || rho <= Math.pow(1d + Math.max(eps, 0.05d), g.nOps()) + 1e-6,
                        g.name + " ε=" + eps + " ρ=" + rho);
                if (eps > 0d) {
                    assertTrue(r.finalKept <= lastKept + 2, "front should not grow with coarser ε");
                }
                lastKept = r.finalKept;
            }
        }
        summarize("ε grid", rhos);
    }

    @Test
    void repeatedSearchIsDeterministic() {
        Graph g = PlanDagSimulator.complexBranching("det-branch", 6);
        SearchResult a = PlanDagSimulator.optimize(g, 0.1d);
        SearchResult b = PlanDagSimulator.optimize(g, 0.1d);
        assertEquals(a.generated, b.generated);
        assertEquals(a.finalKept, b.finalKept);
        assertEquals(a.front.get(0).path, b.front.get(0).path);
        assertEquals(a.front.get(0).cost.getLatency(), b.front.get(0).cost.getLatency(), 1e-12);
    }

    @Test
    void pruningAlwaysBeatsCartesianOnMediumGraphs() {
        for (Graph g : Arrays.asList(
                PlanDagSimulator.chain("c8x4", 8, 4),
                PlanDagSimulator.chain("c12x6", 12, 6),
                PlanDagSimulator.diamond("d6", 6),
                PlanDagSimulator.bushy("b6", 6),
                PlanDagSimulator.complexBranching("br6", 6),
                PlanDagSimulator.twoDiamonds("td6", 6)
        )) {
            SearchResult exact = PlanDagSimulator.optimize(g, 0d);
            SearchResult alpha = PlanDagSimulator.optimize(g, 0.1d);
            assertTrue(exact.generated < g.cartesian(), g.name + " exact");
            assertTrue(alpha.generated <= exact.generated, g.name + " alpha");
            assertTrue(alpha.finalKept <= exact.finalKept, g.name + " front");
        }
    }

    private static Row compare(Graph g, double epsilon, double budgetFrac) {
        PlanDagSimulator.optimize(g, 0d);
        long t0 = System.nanoTime();
        SearchResult exact = PlanDagSimulator.optimize(g, 0d);
        long exactNs = System.nanoTime() - t0;
        t0 = System.nanoTime();
        SearchResult alpha = PlanDagSimulator.optimize(g, epsilon);
        long alphaNs = System.nanoTime() - t0;
        double b = PlanDagSimulator.budgetAt(exact.front, budgetFrac);
        double rho = PlanDagSimulator.rho(
                PlanDagSimulator.pick(alpha.front, b),
                PlanDagSimulator.pick(exact.front, b),
                b);
        double cutPct = 100d * (1d - (double) alpha.generated / Math.max(1, exact.generated));
        return new Row(g, exact, alpha, exactNs, alphaNs, rho, cutPct);
    }

    private static void assertRow(Row row, double alpha) {
        assertTrue(row.alpha.finalKept <= row.exact.finalKept + 1, row.graph.name + " front");
        assertTrue(row.alpha.generated <= row.exact.generated, row.graph.name + " gen");
        if (row.graph.cartesian() >= 1_000L) {
            assertTrue(row.alpha.generated < row.graph.cartesian(), row.graph.name + " vs cartesian");
        }
        if (!Double.isInfinite(row.rho)) {
            assertTrue(row.rho <= Math.pow(alpha, row.graph.nOps()) + 1e-6,
                    row.graph.name + " ρ=" + row.rho);
        }
    }

    private static void summarize(String label, List<Double> rhos) {
        double sum = 0d;
        double max = 0d;
        int within = 0;
        int finite = 0;
        int missedBudget = 0;
        for (double r : rhos) {
            if (!Double.isFinite(r)) {
                missedBudget++;
                continue;
            }
            finite++;
            sum += r;
            max = Math.max(max, r);
            if (r <= ALPHA + 1e-6) {
                within++;
            }
        }
        double mean = finite == 0 ? 1d : sum / finite;
        System.out.printf(Locale.ROOT,
                "  [%s] n=%d  mean ρ=%.3f  max ρ=%.3f  within one-shot α: %d/%d  budget-miss: %d%n",
                label, rhos.size(), mean, max, within, finite, missedBudget);
        assertTrue(mean < 1.35d, label + " mean ρ=" + mean);
    }

    private static String cart(long n) {
        if (n >= 1_000_000_000L) {
            return String.format(Locale.ROOT, "%.1e", (double) n);
        }
        return Long.toString(n);
    }

    private static long pow(int a, int b) {
        long n = 1L;
        for (int i = 0; i < b; i++) {
            if (n > Long.MAX_VALUE / a) {
                return Long.MAX_VALUE;
            }
            n *= a;
        }
        return n;
    }

    private static final class Row {
        final Graph graph;
        final SearchResult exact;
        final SearchResult alpha;
        final long exactNs;
        final long alphaNs;
        final double rho;
        final double cutPct;

        Row(Graph graph, SearchResult exact, SearchResult alpha, long exactNs, long alphaNs, double rho, double cutPct) {
            this.graph = graph;
            this.exact = exact;
            this.alpha = alpha;
            this.exactNs = exactNs;
            this.alphaNs = alphaNs;
            this.rho = rho;
            this.cutPct = cutPct;
        }
    }
}
