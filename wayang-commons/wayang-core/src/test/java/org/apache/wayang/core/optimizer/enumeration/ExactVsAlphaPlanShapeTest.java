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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Compares exact Pareto DP ({@code ε = 0}) with α-Pareto DP ({@code ε > 0}) on
 * chains, diamonds, bushy trees, and wide fan-in DAGs. Ground truth on small
 * graphs is brute force over every platform assignment.
 */
class ExactVsAlphaPlanShapeTest {

    private static final String[] NAMED_PLATFORMS = {"JAVA", "SPARK", "FLINK", "GPU"};
    private static final double ALPHA_EPS = 0.1d;
    private static final double ALPHA = 1d + ALPHA_EPS;
    private static final Comparator<Partial> BY_PATH = Comparator.comparing(p -> p.path);

    private enum Kind {SCAN, FILTER, MAP, JOIN, AGG, SINK}

    @Test
    void smallShapesExactDpMatchesExhaustive() {
        for (Graph g : Arrays.asList(chain("chain-s", 4, 4), diamond("diamond-s", 4), bushy("bushy-s", 4))) {
            SearchResult exact = optimize(g, 0d);
            SearchResult brute = exhaustive(g);
            Partial dpPick = pick(exact.front, budgetOf(exact.front));
            Partial brutePick = pick(brute.front, budgetOf(exact.front));
            assertEquals(brutePick.cost.getLatency(), dpPick.cost.getLatency(), 1e-9, g.name);
            assertEquals(brutePick.cost.getMonetary(), dpPick.cost.getMonetary(), 1e-9, g.name);
        }
    }

    @Test
    void exactVersusAlphaOnChainsDagsAndBranching() {
        List<Graph> graphs = Arrays.asList(
                chain("chain-small", 4, 4),
                chain("chain-medium", 8, 4),
                chain("chain-large", 12, 4),
                diamond("diamond", 4),
                twoDiamonds("two-diamonds", 4),
                bushy("bushy-join", 4),
                fanIn("fan-in-6", 6, 4),
                complexBranching("complex-branch", 4),
                chain("chain-dense", 10, 12),
                bushy("bushy-dense", 10),
                fanIn("fan-in-dense", 5, 8),
                complexBranching("complex-dense", 10)
        );

        System.out.println();
        System.out.println("Exact Pareto DP vs α-Pareto (ε=0.1, α=1.1). Budget = 40th percentile money on the exact front.");
        System.out.println(header());

        List<Row> rows = new ArrayList<>();
        for (Graph g : graphs) {
            optimize(g, 0d);
            optimize(g, ALPHA_EPS);

            long t0 = System.nanoTime();
            SearchResult exact = optimize(g, 0d);
            long exactNs = System.nanoTime() - t0;

            t0 = System.nanoTime();
            SearchResult alpha = optimize(g, ALPHA_EPS);
            long alphaNs = System.nanoTime() - t0;

            SearchResult brute = g.cartesian() <= 80_000 ? exhaustive(g) : null;
            double budget = budgetOf(exact.front);
            Partial exactPick = pick(exact.front, budget);
            Partial alphaPick = pick(alpha.front, budget);
            double rho = exactPick.cost.getLatency() <= 0d
                    ? 1d
                    : alphaPick.cost.getLatency() / exactPick.cost.getLatency();

            Row row = new Row(g, exact, alpha, brute, exactNs, alphaNs, budget, exactPick, alphaPick, rho);
            rows.add(row);
            System.out.println(format(row));

            assertTrue(alpha.finalKept <= exact.finalKept,
                    g.name + ": α-front should not be larger than the exact front");
            assertTrue(alpha.generated <= exact.generated,
                    g.name + ": α-Pareto should concatenate fewer or equal subplans");
            assertTrue(alpha.generated < g.cartesian() || g.ops.length >= 10,
                    g.name + ": pruning should cut the cartesian product");
            if (brute != null) {
                Partial brutePick = pick(brute.front, budget);
                assertEquals(brutePick.cost.getLatency(), exactPick.cost.getLatency(), 1e-9, g.name);
                assertEquals(brutePick.cost.getMonetary(), exactPick.cost.getMonetary(), 1e-9, g.name);
            }
            assertTrue(rho <= Math.pow(ALPHA, g.ops.length) + 1e-6,
                    g.name + " latency ratio " + rho + " vs α^ops=" + Math.pow(ALPHA, g.ops.length));
        }

        System.out.println();
        System.out.println("Trade-off summary:");
        for (Row row : rows) {
            double cut = 1d - (double) row.alpha.generated / (double) row.exact.generated;
            System.out.printf(Locale.ROOT,
                    "  %-16s  exact %d gen / %d kept, α %d gen / %d kept (%.0f%% fewer concatenations), ρ=%.3f, cartesian=%d%n",
                    row.graph.name,
                    row.exact.generated,
                    row.exact.finalKept,
                    row.alpha.generated,
                    row.alpha.finalKept,
                    100d * cut,
                    row.rho,
                    row.graph.cartesian());
        }

        double meanRho = rows.stream().mapToDouble(r -> r.rho).average().orElse(1d);
        assertTrue(meanRho < 1.25d, "mean latency ratio should stay close to 1, was " + meanRho);
    }

    @Test
    void epsilonSweepShowsSpeedSubplanQualityTradeoff() {
        Graph g = chain("chain-dense-sweep", 12, 12);
        optimize(g, 0d);
        SearchResult exact = optimize(g, 0d);
        double budget = budgetOf(exact.front);
        Partial exactPick = pick(exact.front, budget);

        System.out.println();
        System.out.println("ε sweep on " + g.name + " (" + g.ops.length + " ops, " + g.nPlats
                + " platforms, cartesian=" + g.cartesian() + "). Exact pick latency="
                + String.format(Locale.ROOT, "%.1f", exactPick.cost.getLatency())
                + " under B=" + String.format(Locale.ROOT, "%.0f", budget));
        System.out.println(String.format(Locale.ROOT,
                "%6s %8s %8s %8s %8s %8s %8s %6s",
                "ε", "ms", "generated", "peak", "final_k", "vs_exact%", "vs_cart%", "ρ"));

        boolean sawSmallerFront = false;
        int lastKept = exact.finalKept;
        for (double eps : Arrays.asList(0d, 0.1d, 0.25d, 0.5d, 1d)) {
            long t0 = System.nanoTime();
            SearchResult r = optimize(g, eps);
            long ns = System.nanoTime() - t0;
            Partial picked = pick(r.front, budget);
            double rho = exactPick.cost.getLatency() <= 0d
                    ? 1d
                    : picked.cost.getLatency() / exactPick.cost.getLatency();
            double vsExact = 100d * (1d - (double) r.generated / (double) exact.generated);
            double vsCart = 100d * (1d - (double) r.generated / (double) g.cartesian());
            System.out.printf(Locale.ROOT,
                    "%6.2f %8.2f %8d %8d %8d %8.1f %8.1f %6.3f%n",
                    eps, ns / 1e6d, r.generated, r.peakKept, r.finalKept, vsExact, vsCart, rho);
            assertTrue(r.finalKept <= exact.finalKept + 1, "ε=" + eps);
            assertTrue(rho <= Math.pow(1d + Math.max(eps, ALPHA_EPS), g.ops.length) + 1e-6, "ε=" + eps);
            if (r.finalKept < lastKept) {
                sawSmallerFront = true;
            }
            lastKept = r.finalKept;
        }
        assertTrue(sawSmallerFront || exact.finalKept <= 4,
                "coarser ε should shrink the front when the exact front is large");
    }

    private static String header() {
        return String.format(Locale.ROOT,
                "%-16s %3s %5s %12s %8s %8s %8s %8s %8s %8s %8s %6s",
                "shape", "ops", "plats", "cartesian", "exact_ms", "alpha_ms",
                "exact_gen", "alpha_gen", "exact_k", "alpha_k", "B", "ρ");
    }

    private static String format(Row row) {
        return String.format(Locale.ROOT,
                "%-16s %3d %5d %12s %8.2f %8.2f %8d %8d %8d %8d %8.0f %6.3f",
                row.graph.name,
                row.graph.ops.length,
                row.graph.nPlats,
                cartesianLabel(row.graph.cartesian()),
                row.exactNs / 1e6d,
                row.alphaNs / 1e6d,
                row.exact.generated,
                row.alpha.generated,
                row.exact.finalKept,
                row.alpha.finalKept,
                row.budget,
                row.rho);
    }

    private static String cartesianLabel(long n) {
        if (n >= 1_000_000_000L) {
            return String.format(Locale.ROOT, "%.1e", (double) n);
        }
        return Long.toString(n);
    }

    private static SearchResult optimize(Graph g, double epsilon) {
        List<Partial> front = new ArrayList<>();
        int generated = 0;
        int peak = 0;
        int keptSum = 0;
        for (int p = 0; p < g.nPlats; p++) {
            front.add(seed(g, p));
            generated++;
        }
        front = pruneByCut(g, front, 0, epsilon);
        peak = Math.max(peak, front.size());
        keptSum += front.size();

        for (int i = 1; i < g.ops.length; i++) {
            List<Partial> next = new ArrayList<>(front.size() * g.nPlats);
            for (Partial left : front) {
                for (int p = 0; p < g.nPlats; p++) {
                    next.add(extend(g, left, i, p));
                }
            }
            generated += next.size();
            front = pruneByCut(g, next, i, epsilon);
            peak = Math.max(peak, front.size());
            keptSum += front.size();
        }
        return new SearchResult(front, generated, peak, keptSum, front.size());
    }

    private static List<Partial> pruneByCut(Graph g, List<Partial> items, int placedUpTo, double epsilon) {
        Map<String, List<Partial>> groups = new LinkedHashMap<>();
        for (Partial item : items) {
            groups.computeIfAbsent(cutKey(g, item, placedUpTo), k -> new ArrayList<>()).add(item);
        }
        List<Partial> kept = new ArrayList<>();
        for (List<Partial> group : groups.values()) {
            kept.addAll(ParetoFront.retain(group, Partial::cost, epsilon, BY_PATH));
        }
        return kept;
    }

    private static String cutKey(Graph g, Partial item, int placedUpTo) {
        StringBuilder sb = new StringBuilder();
        for (int j = 0; j <= placedUpTo; j++) {
            boolean open = false;
            for (int s : g.succs[j]) {
                if (s > placedUpTo) {
                    open = true;
                    break;
                }
            }
            if (open) {
                sb.append(j).append('=').append(item.plat[j]).append(';');
            }
        }
        return sb.toString();
    }

    private static Partial seed(Graph g, int platform) {
        byte[] plat = new byte[g.ops.length];
        Arrays.fill(plat, (byte) -1);
        plat[0] = (byte) platform;
        return new Partial(
                plat,
                g.lat[0][platform],
                g.money[0][platform],
                platName(g, platform)
        );
    }

    private static Partial extend(Graph g, Partial left, int op, int platform) {
        byte[] plat = left.plat.clone();
        plat[op] = (byte) platform;
        double lat = left.lat + g.lat[op][platform];
        double money = left.money + g.money[op][platform];
        for (int pred : g.preds[op]) {
            VectorCost conv = conversion(left.plat[pred], platform);
            lat += conv.getLatency();
            money += conv.getMonetary();
        }
        return new Partial(plat, lat, money, left.path + "-" + platName(g, platform));
    }

    private static SearchResult exhaustive(Graph g) {
        List<Partial> all = new ArrayList<>();
        byte[] plat = new byte[g.ops.length];
        Arrays.fill(plat, (byte) -1);
        enumerate(g, 0, plat, all);
        List<Partial> front = ParetoFront.retain(all, Partial::cost, 0d, BY_PATH);
        return new SearchResult(front, all.size(), all.size(), all.size(), front.size());
    }

    private static void enumerate(Graph g, int op, byte[] plat, List<Partial> out) {
        if (op == g.ops.length) {
            out.add(score(g, plat));
            return;
        }
        for (int p = 0; p < g.nPlats; p++) {
            plat[op] = (byte) p;
            enumerate(g, op + 1, plat, out);
        }
    }

    private static Partial score(Graph g, byte[] plat) {
        double lat = 0d;
        double money = 0d;
        StringBuilder path = new StringBuilder();
        for (int i = 0; i < g.ops.length; i++) {
            int p = plat[i];
            lat += g.lat[i][p];
            money += g.money[i][p];
            for (int pred : g.preds[i]) {
                VectorCost conv = conversion(plat[pred], p);
                lat += conv.getLatency();
                money += conv.getMonetary();
            }
            if (i > 0) {
                path.append('-');
            }
            path.append(platName(g, p));
        }
        return new Partial(plat.clone(), lat, money, path.toString());
    }

    private static String platName(Graph g, int p) {
        if (g.nPlats == NAMED_PLATFORMS.length && p < NAMED_PLATFORMS.length) {
            return NAMED_PLATFORMS[p];
        }
        return "p" + p;
    }

    private static VectorCost conversion(int from, int to) {
        if (from == to) {
            return new VectorCost(0, 0);
        }
        int dist = Math.abs(from - to);
        return new VectorCost(5d * dist, 4d * dist);
    }

    private static Partial pick(List<Partial> front, double budget) {
        return VectorPlanSelection.pick(front, Partial::cost, budget, 1d, 0d, BY_PATH);
    }

    private static double budgetOf(List<Partial> front) {
        double min = Double.POSITIVE_INFINITY;
        double max = 0d;
        for (Partial p : front) {
            min = Math.min(min, p.cost.getMonetary());
            max = Math.max(max, p.cost.getMonetary());
        }
        if (!Double.isFinite(min) || max <= min) {
            return Double.POSITIVE_INFINITY;
        }
        return min + 0.4d * (max - min);
    }

    private static Graph chain(String name, int n, int nPlats) {
        Kind[] kinds = new Kind[n];
        kinds[0] = Kind.SCAN;
        kinds[n - 1] = Kind.SINK;
        Kind[] mid = {Kind.FILTER, Kind.MAP, Kind.JOIN, Kind.AGG};
        for (int i = 1; i < n - 1; i++) {
            kinds[i] = mid[(i - 1) % mid.length];
        }
        Builder b = new Builder(name, nPlats, kinds);
        for (int i = 1; i < n; i++) {
            b.edge(i - 1, i);
        }
        return b.build();
    }

    private static Graph diamond(String name, int nPlats) {
        Builder b = new Builder(name, nPlats, Kind.SCAN, Kind.FILTER, Kind.MAP, Kind.JOIN, Kind.SINK);
        b.edge(0, 1).edge(0, 2).edge(1, 3).edge(2, 3).edge(3, 4);
        return b.build();
    }

    private static Graph twoDiamonds(String name, int nPlats) {
        Builder b = new Builder(name, nPlats,
                Kind.SCAN, Kind.FILTER, Kind.MAP, Kind.JOIN,
                Kind.FILTER, Kind.MAP, Kind.JOIN, Kind.SINK);
        b.edge(0, 1).edge(0, 2).edge(1, 3).edge(2, 3);
        b.edge(3, 4).edge(3, 5).edge(4, 6).edge(5, 6).edge(6, 7);
        return b.build();
    }

    private static Graph bushy(String name, int nPlats) {
        Builder b = new Builder(name, nPlats,
                Kind.SCAN, Kind.SCAN, Kind.SCAN, Kind.SCAN,
                Kind.JOIN, Kind.JOIN, Kind.JOIN, Kind.SINK);
        b.edge(0, 4).edge(1, 4);
        b.edge(2, 5).edge(3, 5);
        b.edge(4, 6).edge(5, 6).edge(6, 7);
        return b.build();
    }

    private static Graph fanIn(String name, int sources, int nPlats) {
        Kind[] kinds = new Kind[sources + 2];
        Arrays.fill(kinds, Kind.SCAN);
        kinds[sources] = Kind.JOIN;
        kinds[sources + 1] = Kind.SINK;
        Builder b = new Builder(name, nPlats, kinds);
        for (int i = 0; i < sources; i++) {
            b.edge(i, sources);
        }
        b.edge(sources, sources + 1);
        return b.build();
    }

    private static Graph complexBranching(String name, int nPlats) {
        Builder b = new Builder(name, nPlats,
                Kind.SCAN, Kind.SCAN, Kind.SCAN,
                Kind.FILTER, Kind.MAP,
                Kind.JOIN, Kind.JOIN,
                Kind.AGG, Kind.MAP, Kind.SINK);
        b.edge(0, 3).edge(1, 4);
        b.edge(3, 5).edge(4, 5);
        b.edge(5, 6).edge(2, 6);
        b.edge(6, 7).edge(6, 8);
        b.edge(7, 9).edge(8, 9);
        return b.build();
    }

    private static double[] latency(Kind kind, int nPlats) {
        double base = baseLatency(kind);
        double[] out = new double[nPlats];
        if (nPlats == 1) {
            out[0] = base;
            return out;
        }
        for (int p = 0; p < nPlats; p++) {
            double t = (double) p / (nPlats - 1);
            out[p] = base * (0.45 + 1.3 * t);
        }
        return out;
    }

    private static double[] money(Kind kind, int nPlats) {
        double base = baseMoney(kind);
        double[] out = new double[nPlats];
        if (nPlats == 1) {
            out[0] = base;
            return out;
        }
        for (int p = 0; p < nPlats; p++) {
            double t = (double) p / (nPlats - 1);
            out[p] = base * (1.75 - 1.35 * t);
        }
        return out;
    }

    private static double baseLatency(Kind kind) {
        switch (kind) {
            case SCAN:
                return 20d;
            case FILTER:
                return 16d;
            case MAP:
                return 24d;
            case JOIN:
                return 80d;
            case AGG:
                return 36d;
            case SINK:
                return 12d;
            default:
                throw new IllegalArgumentException(kind.name());
        }
    }

    private static double baseMoney(Kind kind) {
        switch (kind) {
            case SCAN:
                return 8d;
            case FILTER:
                return 6d;
            case MAP:
                return 10d;
            case JOIN:
                return 22d;
            case AGG:
                return 12d;
            case SINK:
                return 5d;
            default:
                throw new IllegalArgumentException(kind.name());
        }
    }

    private static final class Builder {
        final String name;
        final int nPlats;
        final Kind[] ops;
        final List<int[]> edges = new ArrayList<>();

        Builder(String name, int nPlats, Kind... ops) {
            this.name = name;
            this.nPlats = nPlats;
            this.ops = ops;
        }

        Builder edge(int from, int to) {
            this.edges.add(new int[]{from, to});
            return this;
        }

        Graph build() {
            @SuppressWarnings("unchecked")
            List<Integer>[] preds = new List[this.ops.length];
            @SuppressWarnings("unchecked")
            List<Integer>[] succs = new List[this.ops.length];
            for (int i = 0; i < this.ops.length; i++) {
                preds[i] = new ArrayList<>();
                succs[i] = new ArrayList<>();
            }
            for (int[] e : this.edges) {
                succs[e[0]].add(e[1]);
                preds[e[1]].add(e[0]);
            }
            int[][] predArr = new int[this.ops.length][];
            int[][] succArr = new int[this.ops.length][];
            double[][] lat = new double[this.ops.length][];
            double[][] money = new double[this.ops.length][];
            for (int i = 0; i < this.ops.length; i++) {
                predArr[i] = preds[i].stream().mapToInt(Integer::intValue).toArray();
                succArr[i] = succs[i].stream().mapToInt(Integer::intValue).toArray();
                lat[i] = latency(this.ops[i], this.nPlats);
                money[i] = ExactVsAlphaPlanShapeTest.money(this.ops[i], this.nPlats);
            }
            return new Graph(this.name, this.nPlats, this.ops, predArr, succArr, lat, money);
        }
    }

    private static final class Graph {
        final String name;
        final int nPlats;
        final Kind[] ops;
        final int[][] preds;
        final int[][] succs;
        final double[][] lat;
        final double[][] money;

        Graph(String name, int nPlats, Kind[] ops, int[][] preds, int[][] succs, double[][] lat, double[][] money) {
            this.name = name;
            this.nPlats = nPlats;
            this.ops = ops;
            this.preds = preds;
            this.succs = succs;
            this.lat = lat;
            this.money = money;
        }

        long cartesian() {
            long n = 1L;
            for (int i = 0; i < this.ops.length; i++) {
                if (n > Long.MAX_VALUE / this.nPlats) {
                    return Long.MAX_VALUE;
                }
                n *= this.nPlats;
            }
            return n;
        }
    }

    private static final class Partial {
        final byte[] plat;
        final double lat;
        final double money;
        final String path;
        final VectorCost cost;

        Partial(byte[] plat, double lat, double money, String path) {
            this.plat = plat;
            this.lat = lat;
            this.money = money;
            this.path = path;
            this.cost = new VectorCost(lat, money);
        }

        VectorCost cost() {
            return this.cost;
        }
    }

    private static final class SearchResult {
        final List<Partial> front;
        final int generated;
        final int peakKept;
        final int keptSum;
        final int finalKept;

        SearchResult(List<Partial> front, int generated, int peakKept, int keptSum, int finalKept) {
            this.front = front;
            this.generated = generated;
            this.peakKept = peakKept;
            this.keptSum = keptSum;
            this.finalKept = finalKept;
        }
    }

    private static final class Row {
        final Graph graph;
        final SearchResult exact;
        final SearchResult alpha;
        final SearchResult brute;
        final long exactNs;
        final long alphaNs;
        final double budget;
        final Partial exactPick;
        final Partial alphaPick;
        final double rho;

        Row(Graph graph, SearchResult exact, SearchResult alpha, SearchResult brute,
            long exactNs, long alphaNs, double budget, Partial exactPick, Partial alphaPick, double rho) {
            this.graph = graph;
            this.exact = exact;
            this.alpha = alpha;
            this.brute = brute;
            this.exactNs = exactNs;
            this.alphaNs = alphaNs;
            this.budget = budget;
            this.exactPick = exactPick;
            this.alphaPick = alphaPick;
            this.rho = rho;
        }
    }
}
