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
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Hard pipeline: five platforms, five operators, plus a two-branch join.
 * Ground truth is brute-force over every platform assignment. The vector
 * prune-after-concat path must recover the same hybrid that exhaustive search
 * finds under a monetary budget, while scalar greeds must not.
 */
class HybridMultiPlatformPipelineTest {

    private static final Platform[] PLATFORMS = Platform.values();
    private static final String[] STAGES = {"scan", "filter", "join", "agg", "sink"};
    /**
     * Tight enough that all-Spark / all-GPU miss it; loose enough that the
     * designed Java→Spark→Java hybrid fits.
     */
    private static final double BUDGET = 95d;
    private static final Comparator<Assignment> BY_PATH = Comparator.comparing(a -> a.path);

    /**
     * Rows: scan, filter, join, agg, sink.
     * Columns: JAVA, SPARK, FLINK, POSTGRES, GPU.
     * Join is cheap only on Spark/GPU; scan/sink are cheap only on Java/Postgres.
     */
    private static final double[][] LATENCY = {
            /* scan   */ {12, 55, 40, 18, 90},
            /* filter */ {10, 35, 22, 28, 60},
            /* join   */ {180, 20, 45, 110, 14},
            /* agg    */ {70, 18, 32, 16, 12},
            /* sink   */ {8, 50, 30, 25, 75}
    };
    private static final double[][] MONEY = {
            /* scan   */ {4, 38, 24, 9, 55},
            /* filter */ {3, 30, 18, 11, 48},
            /* join   */ {8, 42, 28, 14, 60},
            /* agg    */ {7, 36, 22, 12, 52},
            /* sink   */ {3, 34, 20, 10, 50}
    };

    @Test
    void fiveStagePipelinePicksHybridNotAUniformPlatform() {
        List<Assignment> universe = enumerateLinear();
        System.out.println("===== 5-stage × 5-platform linear pipeline =====");
        System.out.println("Operators: " + Arrays.toString(STAGES));
        System.out.println("Platforms: " + Arrays.toString(PLATFORMS));
        System.out.println("Budget B = " + BUDGET);
        System.out.println("|assignments| = " + universe.size());

        printUniform(universe);
        Assignment exhaustive = pickMinLatency(universe, BUDGET);
        Assignment greedyLat = greedyLinear(true);
        Assignment greedyMoney = greedyLinear(false);
        Assignment dp = pruneAfterConcatLinear(0d);

        System.out.println("\nScalar greedy (min latency each stage, + conversion): " + greedyLat);
        System.out.println("Scalar greedy (min money each stage, + conversion):   " + greedyMoney);
        System.out.println("Exhaustive min latency s.t. money <= B:               " + exhaustive);
        System.out.println("Prune-after-concat + pick (exact Pareto):             " + dp);

        assertTrue(exhaustive.distinctPlatforms() >= 3, "Optimum under B must be a hybrid, was " + exhaustive.path);
        assertTrue(exhaustive.path.contains("JAVA") && exhaustive.path.contains("SPARK"),
                "Designed hybrid uses Java for light ops and Spark for the heavy join: " + exhaustive.path);
        for (Platform p : PLATFORMS) {
            assertTrue(!isUniform(exhaustive, p), "Optimum must not be all-" + p);
        }
        assertTrue(greedyLat.cost.getMonetary() > BUDGET,
                "Latency-greedy should exceed the budget (it chases Spark/GPU), money="
                        + greedyLat.cost.getMonetary());
        assertTrue(greedyMoney.cost.getLatency() > exhaustive.cost.getLatency() + 1d,
                "Money-greedy should stay on cheap Java/Postgres and miss the fast join");
        assertEquals(exhaustive.path, dp.path);
        assertEquals(exhaustive.cost.getLatency(), dp.cost.getLatency(), 1e-9);
        assertEquals(exhaustive.cost.getMonetary(), dp.cost.getMonetary(), 1e-9);

        Assignment bestUniform = bestUniformUnderBudget(BUDGET);
        System.out.println("Best single-platform plan under B:                    " + bestUniform);
        assertTrue(bestUniform.cost.getLatency() > exhaustive.cost.getLatency(),
                "Hybrid must beat every uniform plan that still fits B");
    }

    @Test
    void twoBranchJoinPicksSplitPlacement() {
        System.out.println("\n===== diamond: scan ─► (filter JAVA-cheap || map SPARK-fast) ─► join ─► sink =====");
        List<Assignment> universe = enumerateDiamond();
        Assignment exhaustive = pickMinLatency(universe, BUDGET);
        Assignment dp = pruneAfterConcatDiamond(0d);
        System.out.println("|assignments| = " + universe.size());
        System.out.println("Exhaustive: " + exhaustive);
        System.out.println("DP prune:   " + dp);

        assertTrue(exhaustive.path.contains("filter=JAVA") || exhaustive.path.contains("filter=POSTGRES"),
                "Filter branch should stay on a cheap engine: " + exhaustive.path);
        assertTrue(exhaustive.path.contains("map=SPARK") || exhaustive.path.contains("map=GPU")
                        || exhaustive.path.contains("join=SPARK") || exhaustive.path.contains("join=GPU"),
                "Heavy map/join should use a fast engine: " + exhaustive.path);
        assertTrue(exhaustive.distinctPlatforms() >= 2, "Diamond optimum must be hybrid: " + exhaustive.path);
        assertEquals(exhaustive.path, dp.path);
        assertEquals(exhaustive.cost.getLatency(), dp.cost.getLatency(), 1e-9);
    }

    private static List<Assignment> enumerateLinear() {
        List<Assignment> out = new ArrayList<>();
        int n = PLATFORMS.length;
        int total = (int) Math.pow(n, STAGES.length);
        for (int code = 0; code < total; code++) {
            Platform[] choice = new Platform[STAGES.length];
            int rest = code;
            for (int s = STAGES.length - 1; s >= 0; s--) {
                choice[s] = PLATFORMS[rest % n];
                rest /= n;
            }
            out.add(scoreLinear(choice));
        }
        return out;
    }

    private static Assignment scoreLinear(Platform[] choice) {
        double lat = 0d;
        double money = 0d;
        for (int s = 0; s < choice.length; s++) {
            int p = choice[s].ordinal();
            lat += LATENCY[s][p];
            money += MONEY[s][p];
            if (s > 0) {
                VectorCost conv = conversion(choice[s - 1], choice[s]);
                lat += conv.getLatency();
                money += conv.getMonetary();
            }
        }
        return new Assignment(formatLinear(choice), new VectorCost(lat, money), EnumSet.copyOf(Arrays.asList(choice)));
    }

    private static Assignment pruneAfterConcatLinear(double epsilon) {
        List<Partial> front = new ArrayList<>();
        for (Platform p : PLATFORMS) {
            front.add(new Partial(p.name(), p, LATENCY[0][p.ordinal()], MONEY[0][p.ordinal()]));
        }
        front = retainPartials(front, epsilon);
        for (int s = 1; s < STAGES.length; s++) {
            List<Partial> next = new ArrayList<>();
            for (Partial left : front) {
                for (Platform p : PLATFORMS) {
                    VectorCost conv = conversion(left.last, p);
                    next.add(new Partial(
                            left.path + "-" + p,
                            p,
                            left.latency + LATENCY[s][p.ordinal()] + conv.getLatency(),
                            left.money + MONEY[s][p.ordinal()] + conv.getMonetary()
                    ));
                }
            }
            front = retainPartials(next, epsilon);
        }
        List<Assignment> plans = front.stream()
                .map(p -> new Assignment(p.path, new VectorCost(p.latency, p.money), EnumSet.of(p.last)))
                .collect(Collectors.toList());
        return VectorPlanSelection.pick(plans, a -> a.cost, BUDGET, 1d, 0d, BY_PATH);
    }

    private static Assignment greedyLinear(boolean byLatency) {
        Platform prev = null;
        Platform[] choice = new Platform[STAGES.length];
        for (int s = 0; s < STAGES.length; s++) {
            Platform best = null;
            double bestScore = Double.POSITIVE_INFINITY;
            for (Platform p : PLATFORMS) {
                VectorCost conv = prev == null ? new VectorCost(0, 0) : conversion(prev, p);
                double score = byLatency
                        ? LATENCY[s][p.ordinal()] + conv.getLatency()
                        : MONEY[s][p.ordinal()] + conv.getMonetary();
                if (score < bestScore) {
                    bestScore = score;
                    best = p;
                }
            }
            choice[s] = best;
            prev = best;
        }
        return scoreLinear(choice);
    }

    /**
     * Diamond: shared scan, then independent filter and map, then join, then sink.
     * Additive costs (meeting constraint): total = scan + filter + map + join + sink + conversions.
     */
    private static List<Assignment> enumerateDiamond() {
        List<Assignment> out = new ArrayList<>();
        for (Platform scan : PLATFORMS) {
            for (Platform filter : PLATFORMS) {
                for (Platform map : PLATFORMS) {
                    for (Platform join : PLATFORMS) {
                        for (Platform sink : PLATFORMS) {
                            out.add(scoreDiamond(scan, filter, map, join, sink));
                        }
                    }
                }
            }
        }
        return out;
    }

    private static Assignment scoreDiamond(Platform scan, Platform filter, Platform map, Platform join, Platform sink) {
        double lat = LATENCY[0][scan.ordinal()] + LATENCY[1][filter.ordinal()]
                + LATENCY[2][map.ordinal()] + LATENCY[3][join.ordinal()] + LATENCY[4][sink.ordinal()];
        double money = MONEY[0][scan.ordinal()] + MONEY[1][filter.ordinal()]
                + MONEY[2][map.ordinal()] + MONEY[3][join.ordinal()] + MONEY[4][sink.ordinal()];
        VectorCost c1 = conversion(scan, filter);
        VectorCost c2 = conversion(scan, map);
        VectorCost c3 = conversion(filter, join);
        VectorCost c4 = conversion(map, join);
        VectorCost c5 = conversion(join, sink);
        lat += c1.getLatency() + c2.getLatency() + c3.getLatency() + c4.getLatency() + c5.getLatency();
        money += c1.getMonetary() + c2.getMonetary() + c3.getMonetary() + c4.getMonetary() + c5.getMonetary();
        String path = String.format("scan=%s|filter=%s|map=%s|join=%s|sink=%s", scan, filter, map, join, sink);
        return new Assignment(path, new VectorCost(lat, money), EnumSet.of(scan, filter, map, join, sink));
    }

    private static Assignment pruneAfterConcatDiamond(double epsilon) {
        List<Partial> scans = new ArrayList<>();
        for (Platform p : PLATFORMS) {
            scans.add(new Partial("scan=" + p, p, LATENCY[0][p.ordinal()], MONEY[0][p.ordinal()]));
        }
        scans = retainPartials(scans, epsilon);

        List<Partial> filters = new ArrayList<>();
        for (Partial scan : scans) {
            for (Platform p : PLATFORMS) {
                VectorCost conv = conversion(scan.last, p);
                filters.add(new Partial(
                        scan.path + "|filter=" + p,
                        p,
                        scan.latency + LATENCY[1][p.ordinal()] + conv.getLatency(),
                        scan.money + MONEY[1][p.ordinal()] + conv.getMonetary(),
                        scan.last
                ));
            }
        }
        filters = retainPartials(filters, epsilon);

        List<Partial> maps = new ArrayList<>();
        for (Partial scan : scans) {
            for (Platform p : PLATFORMS) {
                VectorCost conv = conversion(scan.last, p);
                maps.add(new Partial(
                        "map=" + p + " from " + scan.last,
                        p,
                        scan.latency + LATENCY[2][p.ordinal()] + conv.getLatency(),
                        scan.money + MONEY[2][p.ordinal()] + conv.getMonetary(),
                        scan.last
                ));
            }
        }
        maps = retainPartials(maps, epsilon);

        List<Partial> joined = new ArrayList<>();
        for (Partial filter : filters) {
            for (Partial map : maps) {
                if (filter.scanPlatform != map.scanPlatform) {
                    continue;
                }
                for (Platform j : PLATFORMS) {
                    VectorCost c3 = conversion(filter.last, j);
                    VectorCost c4 = conversion(map.last, j);
                    double lat = filter.latency + (map.latency - LATENCY[0][map.scanPlatform.ordinal()])
                            + LATENCY[3][j.ordinal()] + c3.getLatency() + c4.getLatency();
                    double money = filter.money + (map.money - MONEY[0][map.scanPlatform.ordinal()])
                            + MONEY[3][j.ordinal()] + c3.getMonetary() + c4.getMonetary();
                    joined.add(new Partial(
                            filter.path + "|" + map.path + "|join=" + j,
                            j,
                            lat,
                            money,
                            filter.scanPlatform
                    ));
                }
            }
        }
        joined = retainPartials(joined, epsilon);

        List<Partial> sinks = new ArrayList<>();
        for (Partial j : joined) {
            for (Platform p : PLATFORMS) {
                VectorCost conv = conversion(j.last, p);
                sinks.add(new Partial(
                        j.path + "|sink=" + p,
                        p,
                        j.latency + LATENCY[4][p.ordinal()] + conv.getLatency(),
                        j.money + MONEY[4][p.ordinal()] + conv.getMonetary()
                ));
            }
        }
        sinks = retainPartials(sinks, epsilon);
        List<Assignment> plans = sinks.stream()
                .map(p -> new Assignment(p.path, new VectorCost(p.latency, p.money), EnumSet.of(p.last)))
                .collect(Collectors.toList());
        Assignment picked = VectorPlanSelection.pick(plans, a -> a.cost, BUDGET, 1d, 0d, BY_PATH);
        return matchExhaustiveFormat(picked);
    }

    private static Assignment matchExhaustiveFormat(Assignment dpPick) {
        Platform scan = extract(dpPick.path, "scan=");
        Platform filter = extract(dpPick.path, "filter=");
        Platform map = extract(dpPick.path, "map=");
        Platform join = extract(dpPick.path, "join=");
        Platform sink = extract(dpPick.path, "sink=");
        return scoreDiamond(scan, filter, map, join, sink);
    }

    private static Platform extract(String path, String key) {
        int i = path.indexOf(key);
        if (i < 0) {
            throw new IllegalArgumentException("missing " + key + " in " + path);
        }
        int start = i + key.length();
        int end = start;
        while (end < path.length() && Character.isLetter(path.charAt(end))) {
            end++;
        }
        return Platform.valueOf(path.substring(start, end));
    }

    private static List<Partial> retainPartials(List<Partial> items, double epsilon) {
        return ParetoFront.retain(items, Partial::cost, epsilon, Comparator.comparing(p -> p.path));
    }

    /**
     * Staying is free. Java↔Spark is a cheap Channel conversion. Anything else is expensive,
     * so the search is not “always switch to the locally fastest engine.”
     */
    private static VectorCost conversion(Platform from, Platform to) {
        if (from == to) {
            return new VectorCost(0, 0);
        }
        boolean javaSpark = (from == Platform.JAVA && to == Platform.SPARK)
                || (from == Platform.SPARK && to == Platform.JAVA);
        if (javaSpark) {
            return new VectorCost(6, 5);
        }
        return new VectorCost(22, 18);
    }

    private static Assignment pickMinLatency(List<Assignment> universe, double budget) {
        return VectorPlanSelection.pick(universe, a -> a.cost, budget, 1d, 0d, BY_PATH);
    }

    private static void printUniform(List<Assignment> universe) {
        System.out.println("\nUniform (single-platform) plans:");
        for (Platform p : PLATFORMS) {
            Platform[] choice = new Platform[STAGES.length];
            Arrays.fill(choice, p);
            Assignment a = scoreLinear(choice);
            System.out.printf(Locale.ROOT, "  all-%-8s %s%n", p, a);
        }
    }

    private static Assignment bestUniformUnderBudget(double budget) {
        Assignment best = null;
        for (Platform p : PLATFORMS) {
            Platform[] choice = new Platform[STAGES.length];
            Arrays.fill(choice, p);
            Assignment a = scoreLinear(choice);
            if (a.cost.getMonetary() > budget) {
                continue;
            }
            if (best == null || a.cost.getLatency() < best.cost.getLatency()) {
                best = a;
            }
        }
        if (best == null) {
            throw new AssertionError("No uniform plan fits B=" + budget);
        }
        return best;
    }

    private static boolean isUniform(Assignment a, Platform p) {
        return a.platforms.size() == 1 && a.platforms.contains(p);
    }

    private static String formatLinear(Platform[] choice) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < choice.length; i++) {
            if (i > 0) {
                sb.append('-');
            }
            sb.append(choice[i]);
        }
        return sb.toString();
    }

    private enum Platform {
        JAVA, SPARK, FLINK, POSTGRES, GPU
    }

    private static final class Assignment {
        final String path;
        final VectorCost cost;
        final Set<Platform> platforms;

        Assignment(String path, VectorCost cost, Set<Platform> platforms) {
            this.path = path;
            this.cost = cost;
            this.platforms = platforms;
        }

        int distinctPlatforms() {
            int n = 0;
            for (Platform p : PLATFORMS) {
                if (this.path.contains(p.name())) {
                    n++;
                }
            }
            return n;
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "%s  (lat=%.0f, money=%.0f, #engines=%d)",
                    this.path, this.cost.getLatency(), this.cost.getMonetary(), this.distinctPlatforms());
        }
    }

    private static final class Partial {
        final String path;
        final Platform last;
        final double latency;
        final double money;
        final Platform scanPlatform;

        Partial(String path, Platform last, double latency, double money) {
            this(path, last, latency, money, last);
        }

        Partial(String path, Platform last, double latency, double money, Platform scanPlatform) {
            this.path = path;
            this.last = last;
            this.latency = latency;
            this.money = money;
            this.scanPlatform = scanPlatform;
        }

        VectorCost cost() {
            return new VectorCost(this.latency, this.money);
        }
    }
}
