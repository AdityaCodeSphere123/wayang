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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Sweeps epsilon, budget, weights, and platform rates and prints how the pick changes.
 */
class CombinationSweepTest {

    private static final List<NamedPlan> CANDIDATES = Arrays.asList(
            new NamedPlan("java-only", 100, 20),
            new NamedPlan("spark-only", 30, 80),
            new NamedPlan("hybrid", 45, 50),
            new NamedPlan("dominated", 90, 90),
            new NamedPlan("cheap-slow", 200, 10),
            new NamedPlan("fast-expensive", 20, 120)
    );

    private static final Comparator<NamedPlan> BY_NAME = Comparator.comparing(p -> p.name);

    @Test
    void sweepEpsilonBudgetWeightsAndRates() {
        System.out.println("Candidates: " + format(CANDIDATES));

        System.out.println("\n=== A) epsilon x budget  (weights = min latency) ===");
        System.out.println(pad("eps\\B", 8) + " | "
                + Arrays.asList(10, 20, 50, 80, 120, "inf").stream()
                .map(b -> pad(String.valueOf(b), 16))
                .collect(Collectors.joining(" ")));
        Set<String> distinctPicks = new LinkedHashSet<>();
        for (double eps : Arrays.asList(0d, 0.1d, 0.5d, 1d, 2d)) {
            List<NamedPlan> front = ParetoFront.retain(CANDIDATES, NamedPlan::cost, eps);
            System.out.println("  front(eps=" + eps + "): " + names(front));
            StringBuilder row = new StringBuilder(pad(String.valueOf(eps), 8)).append(" | ");
            for (double budget : Arrays.asList(10d, 20d, 50d, 80d, 120d, Double.POSITIVE_INFINITY)) {
                NamedPlan pick = VectorPlanSelection.pick(front, NamedPlan::cost, budget, 1d, 0d, BY_NAME);
                distinctPicks.add(pick.name);
                row.append(pad(pick.name, 16)).append(" ");
            }
            System.out.println(row);
        }
        assertTrue(distinctPicks.size() >= 4, "Budget/epsilon sweep should change the picked plan");

        System.out.println("\n=== B) weights x budget  (exact Pareto, eps=0) ===");
        List<NamedPlan> exact = ParetoFront.retain(CANDIDATES, NamedPlan::cost, 0d);
        System.out.println("Exact front: " + format(exact));
        double[][] weights = {{1, 0}, {0, 1}, {1, 1}, {2, 1}, {1, 2}, {0.3, 0.7}};
        System.out.println(pad("wL,wM", 10) + " | "
                + Arrays.asList(10, 50, 120, "inf").stream()
                .map(b -> pad(String.valueOf(b), 16))
                .collect(Collectors.joining(" ")));
        Set<String> weightPicks = new LinkedHashSet<>();
        for (double[] w : weights) {
            StringBuilder row = new StringBuilder(pad(w[0] + "," + w[1], 10)).append(" | ");
            for (double budget : Arrays.asList(10d, 50d, 120d, Double.POSITIVE_INFINITY)) {
                NamedPlan pick = VectorPlanSelection.pick(exact, NamedPlan::cost, budget, w[0], w[1], BY_NAME);
                weightPicks.add(pick.name + "@" + budget);
                row.append(pad(pick.name, 16)).append(" ");
            }
            System.out.println(row);
        }
        assertTrue(weightPicks.size() > 3);

        System.out.println("\n=== C) fix times, vary Spark rate, pick with B=40 and B=inf ===");
        System.out.println("Times fixed: java=80ms, spark=25ms, postgres=60ms; java rate=0.2, pg rate=0.5");
        System.out.println(pad("spark$/ms", 12) + " | " + pad("Pareto set", 42) + " | "
                + pad("B=inf", 12) + " | " + pad("B=40", 12));
        List<String> picksAt40 = new ArrayList<>();
        for (double sparkRate : Arrays.asList(0.1d, 0.2d, 1d, 3d, 5d, 10d, 20d)) {
            List<NamedPlan> scaled = Arrays.asList(
                    new NamedPlan("java", 80, 80 * 0.2),
                    new NamedPlan("spark", 25, 25 * sparkRate),
                    new NamedPlan("postgres", 60, 60 * 0.5)
            );
            List<NamedPlan> front = ParetoFront.retain(scaled, NamedPlan::cost, 0d);
            NamedPlan unconstrained = VectorPlanSelection.pick(front, NamedPlan::cost, Double.POSITIVE_INFINITY, 1d, 0d, BY_NAME);
            NamedPlan budgeted = VectorPlanSelection.pick(front, NamedPlan::cost, 40d, 1d, 0d, BY_NAME);
            picksAt40.add(budgeted.name);
            System.out.println(pad(String.valueOf(sparkRate), 12) + " | "
                    + pad(names(front).toString(), 42) + " | "
                    + pad(unconstrained.name, 12) + " | "
                    + pad(budgeted.name, 12));
        }
        assertTrue(new LinkedHashSet<>(picksAt40).size() >= 2, "Spark rate should change the budgeted pick");

        System.out.println("\n=== D) epsilon vs front size (same candidates) ===");
        int previousSize = Integer.MAX_VALUE;
        for (double eps : Arrays.asList(0d, 0.05d, 0.1d, 0.25d, 0.5d, 1d, 2d, 5d)) {
            List<NamedPlan> front = ParetoFront.retain(CANDIDATES, NamedPlan::cost, eps);
            System.out.println("  eps=" + pad(String.valueOf(eps), 4)
                    + " | |S|=" + front.size()
                    + " | " + names(front));
            assertTrue(front.size() <= previousSize, "Coarser epsilon should not grow the front");
            previousSize = front.size();
        }
        System.out.println("\nSweeps complete.");
    }

    private static List<String> names(List<NamedPlan> plans) {
        return plans.stream().map(p -> p.name).collect(Collectors.toList());
    }

    private static String format(List<NamedPlan> plans) {
        return plans.stream().map(NamedPlan::toString).collect(Collectors.joining(", "));
    }

    private static String pad(String s, int n) {
        if (s.length() >= n) {
            return s;
        }
        return s + " ".repeat(n - s.length());
    }

    private static final class NamedPlan {
        final String name;
        final VectorCost cost;

        NamedPlan(String name, double latency, double monetary) {
            this.name = name;
            this.cost = new VectorCost(latency, monetary);
        }

        VectorCost cost() {
            return this.cost;
        }

        @Override
        public String toString() {
            return String.format("%s(lat=%.0f, money=%.0f)", this.name, this.cost.getLatency(), this.cost.getMonetary());
        }
    }
}
