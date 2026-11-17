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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Additive DAG enumerator used by the meeting-action tests: exact Pareto DP vs α-Pareto.
 */
final class PlanDagSimulator {

    static final Comparator<Partial> BY_PATH = Comparator.comparing(p -> p.path);

    enum Kind {SCAN, FILTER, MAP, JOIN, AGG, SINK}

    private PlanDagSimulator() {
    }

    static SearchResult optimize(Graph g, double epsilon) {
        return optimize(g, epsilon, 0);
    }

    /** Same DP, but each group keeps at most {@code maxFront} plans (see {@link ParetoFront#retainBounded});
     * {@code maxFront < 0} emulates original Wayang (one min-latency plan per group). */
    static SearchResult optimize(Graph g, double epsilon, int maxFront) {
        return optimize(g, epsilon, maxFront, false);
    }

    /**
     * @param crossGroup also drop plans that are dominated by a plan of <i>another</i> group once the cost of
     *                   switching that plan's open operators to this group's platforms is added (exact; see
     *                   {@link #crossGroupPrune})
     */
    static SearchResult optimize(Graph g, double epsilon, int maxFront, boolean crossGroup) {
        return optimize(g, epsilon, maxFront, crossGroup ? 0d : -1d);
    }

    /**
     * @param crossGroupEps {@code < 0}: no cross-group pruning; {@code 0}: exact cross-group pruning;
     *                      {@code > 0}: drop {@code p} if a switched-in vector is within {@code (1+ε)} of it
     */
    static SearchResult optimize(Graph g, double epsilon, int maxFront, double crossGroupEps) {
        return optimize(g, epsilon, maxFront, crossGroupEps, Double.POSITIVE_INFINITY);
    }

    /**
     * @param budget monetary bound. As in {@code ParetoPruningStrategy}: within a group, plans above the budget are
     *               dropped whenever the group still has a plan within it (costs only grow, so this is exact). Not
     *               applied in the original-Wayang mode ({@code maxFront < 0}), which has no notion of a budget.
     */
    static SearchResult optimize(Graph g, double epsilon, int maxFront, double crossGroupEps, double budget) {
        final boolean crossGroup = crossGroupEps >= 0d;
        List<Partial> front = new ArrayList<>();
        int generated = 0;
        int peak = 0;
        int keptSum = 0;
        for (int p = 0; p < g.nPlats; p++) {
            front.add(seed(g, p));
            generated++;
        }
        front = pruneByCut(g, front, 0, epsilon, maxFront, budget);
        if (crossGroup) front = crossGroupPrune(g, front, 0, crossGroupEps);
        peak = Math.max(peak, front.size());
        keptSum += front.size();
        for (int i = 1; i < g.nOps(); i++) {
            List<Partial> next = new ArrayList<>(front.size() * g.nPlats);
            for (Partial left : front) {
                for (int p = 0; p < g.nPlats; p++) {
                    next.add(extend(g, left, i, p));
                }
            }
            generated += next.size();
            front = pruneByCut(g, next, i, epsilon, maxFront, budget);
            if (crossGroup) front = crossGroupPrune(g, front, i, crossGroupEps);
            peak = Math.max(peak, front.size());
            keptSum += front.size();
        }
        return new SearchResult(front, generated, peak, keptSum, front.size());
    }

    static SearchResult exhaustive(Graph g) {
        List<Partial> all = new ArrayList<>();
        byte[] plat = new byte[g.nOps()];
        Arrays.fill(plat, (byte) -1);
        enumerate(g, 0, plat, all);
        List<Partial> front = ParetoFront.retain(all, Partial::cost, 0d, BY_PATH);
        return new SearchResult(front, all.size(), all.size(), all.size(), front.size());
    }

    static Partial pick(List<Partial> front, double budget) {
        return VectorPlanSelection.pick(front, Partial::cost, budget, 1d, 0d, BY_PATH);
    }

    static double budgetAt(List<Partial> front, double fraction) {
        double min = Double.POSITIVE_INFINITY;
        double max = 0d;
        for (Partial p : front) {
            min = Math.min(min, p.cost.getMonetary());
            max = Math.max(max, p.cost.getMonetary());
        }
        if (!Double.isFinite(min) || max <= min) {
            return Double.POSITIVE_INFINITY;
        }
        return min + fraction * (max - min);
    }

    static double rho(Partial approx, Partial exact) {
        return rho(approx, exact, Double.POSITIVE_INFINITY);
    }

    static double rho(Partial approx, Partial exact, double budget) {
        if (exact == null || approx == null || exact.cost.getLatency() <= 0d) {
            return 1d;
        }
        final boolean unconstrained = !Double.isFinite(budget) || budget == Double.POSITIVE_INFINITY;
        final boolean exactOk = unconstrained || exact.cost.getMonetary() <= budget;
        final boolean approxOk = unconstrained || approx.cost.getMonetary() <= budget;
        if (exactOk && !approxOk) {
            return Double.POSITIVE_INFINITY;
        }
        return approx.cost.getLatency() / exact.cost.getLatency();
    }

    static Graph chain(String name, int n, int nPlats) {
        Kind[] kinds = pipelineKinds(n);
        Builder b = new Builder(name, nPlats, kinds);
        for (int i = 1; i < n; i++) {
            b.edge(i - 1, i);
        }
        return b.build();
    }

    static Graph diamond(String name, int nPlats) {
        Builder b = new Builder(name, nPlats, Kind.SCAN, Kind.FILTER, Kind.MAP, Kind.JOIN, Kind.SINK);
        b.edge(0, 1).edge(0, 2).edge(1, 3).edge(2, 3).edge(3, 4);
        return b.build();
    }

    static Graph twoDiamonds(String name, int nPlats) {
        Builder b = new Builder(name, nPlats,
                Kind.SCAN, Kind.FILTER, Kind.MAP, Kind.JOIN,
                Kind.FILTER, Kind.MAP, Kind.JOIN, Kind.SINK);
        b.edge(0, 1).edge(0, 2).edge(1, 3).edge(2, 3);
        b.edge(3, 4).edge(3, 5).edge(4, 6).edge(5, 6).edge(6, 7);
        return b.build();
    }

    static Graph threeDiamonds(String name, int nPlats) {
        Builder b = new Builder(name, nPlats,
                Kind.SCAN,
                Kind.FILTER, Kind.MAP, Kind.JOIN,
                Kind.FILTER, Kind.MAP, Kind.JOIN,
                Kind.FILTER, Kind.MAP, Kind.JOIN,
                Kind.SINK);
        b.edge(0, 1).edge(0, 2).edge(1, 3).edge(2, 3);
        b.edge(3, 4).edge(3, 5).edge(4, 6).edge(5, 6);
        b.edge(6, 7).edge(6, 8).edge(7, 9).edge(8, 9).edge(9, 10);
        return b.build();
    }

    static Graph bushy(String name, int nPlats) {
        Builder b = new Builder(name, nPlats,
                Kind.SCAN, Kind.SCAN, Kind.SCAN, Kind.SCAN,
                Kind.JOIN, Kind.JOIN, Kind.JOIN, Kind.SINK);
        b.edge(0, 4).edge(1, 4);
        b.edge(2, 5).edge(3, 5);
        b.edge(4, 6).edge(5, 6).edge(6, 7);
        return b.build();
    }

    static Graph leftDeep(String name, int sources, int nPlats) {
        Kind[] kinds = new Kind[sources + sources - 1 + 1];
        Arrays.fill(kinds, Kind.SCAN);
        int join = sources;
        for (int i = 0; i < sources - 1; i++) {
            kinds[join + i] = Kind.JOIN;
        }
        kinds[kinds.length - 1] = Kind.SINK;
        Builder b = new Builder(name, nPlats, kinds);
        b.edge(0, sources);
        b.edge(1, sources);
        for (int i = 1; i < sources - 1; i++) {
            b.edge(sources + i - 1, sources + i);
            b.edge(i + 1, sources + i);
        }
        b.edge(sources + sources - 2, kinds.length - 1);
        return b.build();
    }

    static Graph fanIn(String name, int sources, int nPlats) {
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

    static Graph fanOut(String name, int nPlats) {
        Builder b = new Builder(name, nPlats,
                Kind.SCAN, Kind.FILTER, Kind.MAP, Kind.AGG, Kind.JOIN, Kind.SINK);
        b.edge(0, 1).edge(0, 2).edge(0, 3).edge(1, 4).edge(2, 4).edge(3, 4).edge(4, 5);
        return b.build();
    }

    static Graph skipEdges(String name, int n, int nPlats) {
        Kind[] kinds = pipelineKinds(n);
        Builder b = new Builder(name, nPlats, kinds);
        for (int i = 1; i < n; i++) {
            b.edge(i - 1, i);
        }
        if (n >= 4) {
            b.edge(0, n / 2);
        }
        if (n >= 6) {
            b.edge(1, n - 2);
        }
        return b.build();
    }

    static Graph complexBranching(String name, int nPlats) {
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

    /**
     * Left-deep join over {@code sources} tables in the order Wayang combines them: scan0, scan1, join1, scan2, join2,
     * ..., sink. Only the running join result and the next scan are open at any point ({@link #leftDeep} places all
     * scans first, which keeps every scan open at once and is not how Wayang enumerates a join tree).
     */
    static Graph leftDeepInterleaved(String name, int sources, int nPlats) {
        final int n = 2 * sources;                     // sources scans + (sources - 1) joins + sink
        final Kind[] kinds = new Kind[n];
        kinds[0] = Kind.SCAN;
        for (int t = 1; t < sources; t++) {
            kinds[2 * t - 1] = Kind.SCAN;
            kinds[2 * t] = Kind.JOIN;
        }
        kinds[n - 1] = Kind.SINK;
        final Builder b = new Builder(name, nPlats, kinds);
        int left = 0;                                   // running result
        for (int t = 1; t < sources; t++) {
            b.edge(left, 2 * t);
            b.edge(2 * t - 1, 2 * t);
            left = 2 * t;
        }
        b.edge(left, n - 1);
        return b.build();
    }

    static Graph randomConnected(String name, int n, int nPlats, long seed) {
        Kind[] kinds = pipelineKinds(n);
        Builder b = new Builder(name, nPlats, kinds);
        Random rng = new Random(seed);
        for (int i = 1; i < n; i++) {
            b.edge(rng.nextInt(i), i);
        }
        int extra = 1 + rng.nextInt(Math.max(1, n / 2));
        for (int e = 0; e < extra; e++) {
            int to = 2 + rng.nextInt(n - 2);
            int from = rng.nextInt(to);
            b.edge(from, to);
        }
        Graph g = b.build();
        jitterCosts(g, rng);
        return g;
    }

    private static void jitterCosts(Graph g, Random rng) {
        for (int i = 0; i < g.nOps(); i++) {
            for (int p = 0; p < g.nPlats; p++) {
                double j = 0.6 + rng.nextDouble() * 0.8;
                g.lat[i][p] *= j;
                g.money[i][p] *= 0.6 + rng.nextDouble() * 0.8;
            }
        }
    }

    private static Kind[] pipelineKinds(int n) {
        Kind[] kinds = new Kind[n];
        kinds[0] = Kind.SCAN;
        kinds[n - 1] = Kind.SINK;
        Kind[] mid = {Kind.FILTER, Kind.MAP, Kind.JOIN, Kind.AGG};
        for (int i = 1; i < n - 1; i++) {
            kinds[i] = mid[(i - 1) % mid.length];
        }
        return kinds;
    }

    private static List<Partial> pruneByCut(Graph g, List<Partial> items, int placedUpTo, double epsilon, int maxFront,
                                            double budget) {
        Map<String, List<Partial>> groups = new LinkedHashMap<>();
        for (Partial item : items) {
            groups.computeIfAbsent(cutKey(g, item, placedUpTo), k -> new ArrayList<>()).add(item);
        }
        List<Partial> kept = new ArrayList<>();
        for (List<Partial> group : groups.values()) {
            if (maxFront >= 0 && Double.isFinite(budget)) {
                final List<Partial> feasible = new ArrayList<>();
                for (Partial q : group) if (q.money <= budget) feasible.add(q);
                if (!feasible.isEmpty()) group = feasible;
            }
            if (maxFront < 0) {
                // Original Wayang: one plan per group, the one with the lowest scalar cost (here: latency).
                kept.add(group.stream().min(Comparator.<Partial>comparingDouble(q -> q.cost.getLatency())
                        .thenComparingDouble(q -> q.cost.getMonetary()).thenComparing(BY_PATH)).get());
            } else {
                kept.addAll(ParetoFront.retainBounded(group, Partial::cost, epsilon, maxFront, BY_PATH));
            }
        }
        return kept;
    }

    /**
     * Exact pruning across groups. Let {@code q} and {@code p} be partial plans whose open operators run on platforms
     * {@code a_j} and {@code b_j}. For every completion {@code s}, the conversions {@code s} needs from the open
     * operators cost {@code q} at most {@code Σ m_j·conv(a_j, b_j)} more than they cost {@code p}
     * ({@code m_j} = open out-edges of {@code j}; triangle inequality {@code conv(a,x) <= conv(a,b) + conv(b,x)}).
     * So if {@code c(q) + Σ m_j·conv(a_j, b_j)} dominates {@code c(p)}, then {@code q+s} dominates {@code p+s} for
     * every {@code s} and {@code p} can go, even though the two plans are in different groups.
     *
     * <p>Comparing all pairs of groups is quadratic. The penalty is a sum over open operators, so the best
     * "switched-in" vectors per group are computed one open operator at a time (a separable distance transform over
     * 2-D Pareto fronts): {@code #open ops × #groups × #platforms} front merges.</p>
     */
    static List<Partial> crossGroupPrune(Graph g, List<Partial> items, int placedUpTo) {
        return crossGroupPrune(g, items, placedUpTo, 0d);
    }

    /** With {@code eps > 0}, {@code p} also goes if a switched-in vector is within a factor {@code 1+eps} of it. */
    static List<Partial> crossGroupPrune(Graph g, List<Partial> items, int placedUpTo, double eps) {
        // Open operators and how many of their out-edges cross the cut.
        final List<Integer> open = new ArrayList<>();
        final List<Integer> mult = new ArrayList<>();
        for (int j = 0; j <= placedUpTo; j++) {
            int m = 0;
            for (int s : g.succs[j]) if (s > placedUpTo) m++;
            if (m > 0) {
                open.add(j);
                mult.add(m);
            }
        }
        if (open.isEmpty() || items.size() < 2) return items;
        final int w = open.size();
        final int base = g.nPlats;

        // Group code: platforms of the open operators in base nPlats.
        final Map<Long, List<Partial>> groups = new LinkedHashMap<>();
        for (Partial item : items) groups.computeIfAbsent(code(item, open, base), k -> new ArrayList<>()).add(item);

        // reach.get(G): Pareto front of c(q) + penalty(q -> G), over q seen so far.
        Map<Long, double[][]> reach = new java.util.HashMap<>(groups.size() * 2);
        for (Map.Entry<Long, List<Partial>> e : groups.entrySet()) {
            final double[][] pts = new double[e.getValue().size()][];
            for (int k = 0; k < pts.length; k++) pts[k] = new double[]{e.getValue().get(k).lat, e.getValue().get(k).money};
            reach.put(e.getKey(), front2d(pts));
        }
        long pow = 1;
        for (int j = 0; j < w; j++, pow *= base) {
            final Map<Long, double[][]> next = new java.util.HashMap<>(reach.size() * 2);
            for (Map.Entry<Long, double[][]> e : reach.entrySet()) {
                final long gcode = e.getKey();
                final int a = (int) ((gcode / pow) % base);
                final List<double[]> merged = new ArrayList<>(java.util.Arrays.asList(e.getValue()));
                for (int b = 0; b < base; b++) {
                    if (b == a) continue;
                    final double[][] src = reach.get(gcode + (long) (b - a) * pow);
                    if (src == null) continue;     // only existing groups: a smaller min, still safe
                    final VectorCost conv = conversion(b, a);
                    final double dl = mult.get(j) * conv.getLatency(), dm = mult.get(j) * conv.getMonetary();
                    for (double[] v : src) merged.add(new double[]{v[0] + dl, v[1] + dm});
                }
                next.put(gcode, front2d(merged.toArray(new double[0][])));
            }
            reach = next;
        }

        final List<Partial> kept = new ArrayList<>(items.size());
        for (Map.Entry<Long, List<Partial>> e : groups.entrySet()) {
            final double[][] r = reach.get(e.getKey());
            for (Partial p : e.getValue()) {
                final boolean drop = eps > 0d
                        ? approxDominatedByOther(r, p, eps)
                        : strictlyDominated(r, p.lat, p.money);
                if (!drop) kept.add(p);
            }
        }
        return kept;
    }

    private static long code(Partial item, List<Integer> open, int base) {
        long c = 0, pow = 1;
        for (int j : open) {
            c += item.plat[j] * pow;
            pow *= base;
        }
        return c;
    }

    /** Exact 2-D Pareto front of points (minimisation), sorted by first coordinate. */
    private static double[][] front2d(double[][] pts) {
        java.util.Arrays.sort(pts, (x, y) -> x[0] != y[0] ? Double.compare(x[0], y[0]) : Double.compare(x[1], y[1]));
        final List<double[]> out = new ArrayList<>();
        double best = Double.POSITIVE_INFINITY;
        for (double[] v : pts) {
            if (v[1] < best) {
                out.add(v);
                best = v[1];
            }
        }
        return out.toArray(new double[0][]);
    }

    /**
     * Approximate cross-group test on log-buckets of width {@code 1+eps}: {@code p} goes if some switched-in vector
     * {@code v} is in a bucket that is no worse on both axes and strictly better on one, or if {@code v} strictly
     * dominates {@code c(p)}. Both relations only ever move to lower buckets, so they cannot form a cycle and the
     * minimal plans always survive (no "both dropped" case). Per prune, a dropped plan is covered within
     * {@code 1+eps} by the plan that covers it; along a chain of drops the factor can multiply, so the overall gap is
     * measured, not guaranteed.
     */
    private static boolean approxDominatedByOther(double[][] front, Partial p, double eps) {
        final double logBase = Math.log1p(eps);
        final long bl = bucket(p.lat, logBase), bm = bucket(p.money, logBase);
        for (double[] v : front) {
            final long vl = bucket(v[0], logBase);
            if (vl > bl) break;                    // sorted by latency
            final long vm = bucket(v[1], logBase);
            if (vm > bm) continue;
            if (vl < bl || vm < bm) return true;
            if (v[0] <= p.lat && v[1] <= p.money && (v[0] < p.lat || v[1] < p.money)) return true;
        }
        return false;
    }

    private static long bucket(double x, double logBase) {
        return x <= 0d ? Long.MIN_VALUE : (long) Math.floor(Math.log(x) / logBase);
    }

    private static boolean strictlyDominated(double[][] front, double lat, double money) {
        for (double[] v : front) {
            if (v[0] > lat) break;                 // sorted by latency
            if (v[1] <= money && (v[0] < lat || v[1] < money)) return true;
        }
        return false;
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
        byte[] plat = new byte[g.nOps()];
        Arrays.fill(plat, (byte) -1);
        plat[0] = (byte) platform;
        return new Partial(plat, g.lat[0][platform], g.money[0][platform], "p" + platform);
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
        return new Partial(plat, lat, money, left.path + "-p" + platform);
    }

    private static void enumerate(Graph g, int op, byte[] plat, List<Partial> out) {
        if (op == g.nOps()) {
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
        for (int i = 0; i < g.nOps(); i++) {
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
            path.append('p').append(p);
        }
        return new Partial(plat.clone(), lat, money, path.toString());
    }

    private static VectorCost conversion(int from, int to) {
        if (from == to) {
            return new VectorCost(0, 0);
        }
        int dist = Math.abs(from - to);
        return new VectorCost(5d * dist, 4d * dist);
    }

    static final class Builder {
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
                lat[i] = scaledLatency(this.ops[i], this.nPlats);
                money[i] = scaledMoney(this.ops[i], this.nPlats);
            }
            return new Graph(this.name, this.nPlats, this.ops, predArr, succArr, lat, money);
        }
    }

    private static double[] scaledLatency(Kind kind, int nPlats) {
        double base = baseLatency(kind);
        double[] out = new double[nPlats];
        if (nPlats == 1) {
            out[0] = base;
            return out;
        }
        for (int p = 0; p < nPlats; p++) {
            out[p] = base * (0.45 + 1.3 * p / (nPlats - 1));
        }
        return out;
    }

    private static double[] scaledMoney(Kind kind, int nPlats) {
        double base = baseMoney(kind);
        double[] out = new double[nPlats];
        if (nPlats == 1) {
            out[0] = base;
            return out;
        }
        for (int p = 0; p < nPlats; p++) {
            out[p] = base * (1.75 - 1.35 * p / (nPlats - 1));
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

    static final class Graph {
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

        int nOps() {
            return this.ops.length;
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

    static final class Partial {
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

    static final class SearchResult {
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
}
