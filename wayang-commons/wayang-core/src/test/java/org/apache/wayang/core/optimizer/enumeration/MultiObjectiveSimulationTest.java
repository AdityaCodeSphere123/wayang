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

import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Prints and checks multi-objective simulations on a fixed candidate set.
 */
class MultiObjectiveSimulationTest {

    private static final NamedPlan JAVA = new NamedPlan("java-only", 100, 20);
    private static final NamedPlan SPARK = new NamedPlan("spark-only", 30, 80);
    private static final NamedPlan HYBRID = new NamedPlan("hybrid", 45, 50);
    private static final NamedPlan DOMINATED = new NamedPlan("dominated", 90, 90);
    private static final NamedPlan CHEAP_SLOW = new NamedPlan("cheap-slow", 200, 10);
    private static final NamedPlan FAST_EXPENSIVE = new NamedPlan("fast-expensive", 20, 120);

    private static final List<NamedPlan> FIXED_CANDIDATES = Arrays.asList(
            JAVA, SPARK, HYBRID, DOMINATED, CHEAP_SLOW, FAST_EXPENSIVE
    );

    @Test
    void simulationsVaryWithEpsilonBudgetAndWeights() {
        System.out.println("========== FIXED CANDIDATE PLANS (cost vectors held constant) ==========");
        FIXED_CANDIDATES.forEach(plan -> System.out.println("  " + plan));

        System.out.println("\n========== 1) Exact vs approximate Pareto (vary epsilon, costs fixed) ==========");
        Map<Double, List<String>> frontsByEpsilon = new LinkedHashMap<>();
        for (double epsilon : Arrays.asList(0d, 0.1d, 0.5d, 1.0d)) {
            System.out.println("\n--- epsilon = " + epsilon + " (alpha = " + (1d + epsilon) + ") ---");
            if (epsilon > 0d) {
                System.out.println("  Intermediate log-buckets:");
                ParetoFront.groupByBuckets(FIXED_CANDIDATES, NamedPlan::cost, epsilon).forEach((bucket, members) ->
                        System.out.println("    bucket " + bucket + " -> " + names(members)));
            }
            List<NamedPlan> front = ParetoFront.retain(FIXED_CANDIDATES, NamedPlan::cost, epsilon);
            frontsByEpsilon.put(epsilon, names(front));
            System.out.println("  Pareto set: " + format(front));
            assertTrue(front.stream().noneMatch(plan -> plan.name.equals("dominated")),
                    "Dominated plan must never survive");
        }
        assertNotEquals(frontsByEpsilon.get(0d), frontsByEpsilon.get(1.0d),
                "Coarse epsilon should change which plans survive");

        System.out.println("\n========== 2) Budget sweep on the exact Pareto set (costs fixed) ==========");
        List<NamedPlan> exactFront = ParetoFront.retain(FIXED_CANDIDATES, NamedPlan::cost, 0d);
        System.out.println("  Intermediate exact Pareto set: " + format(exactFront));
        for (double budget : Arrays.asList(5d, 10d, 20d, 50d, 80d, 120d, Double.POSITIVE_INFINITY)) {
            NamedPlan picked = VectorPlanSelection.pick(
                    exactFront, NamedPlan::cost, budget, 1d, 0d, Comparator.comparing(p -> p.name));
            System.out.println(String.format("  budget=%s -> pick %s  (latency=%.0f, money=%.0f)",
                    Double.isInfinite(budget) ? "inf" : String.valueOf((int) budget),
                    picked.name, picked.cost.getLatency(), picked.cost.getMonetary()));
            if (budget >= 10d && !Double.isInfinite(budget)) {
                assertTrue(picked.cost.getMonetary() <= budget,
                        "Feasible budget must not pick a plan that exceeds it");
            }
        }
        NamedPlan at10 = VectorPlanSelection.pick(exactFront, NamedPlan::cost, 10d, 1d, 0d, Comparator.comparing(p -> p.name));
        NamedPlan at80 = VectorPlanSelection.pick(exactFront, NamedPlan::cost, 80d, 1d, 0d, Comparator.comparing(p -> p.name));
        NamedPlan atInf = VectorPlanSelection.pick(exactFront, NamedPlan::cost, Double.POSITIVE_INFINITY, 1d, 0d, Comparator.comparing(p -> p.name));
        assertEquals("cheap-slow", at10.name);
        assertEquals("spark-only", at80.name);
        assertEquals("fast-expensive", atInf.name);
        assertNotEquals(at10.name, at80.name);
        assertNotEquals(at80.name, atInf.name);

        System.out.println("\n========== 3) Weight sweep, no budget (costs and Pareto set fixed) ==========");
        System.out.println("  Intermediate Pareto set: " + format(exactFront));
        NamedPlan minLatency = VectorPlanSelection.pick(exactFront, NamedPlan::cost, Double.POSITIVE_INFINITY, 1d, 0d, Comparator.comparing(p -> p.name));
        NamedPlan minMoney = VectorPlanSelection.pick(exactFront, NamedPlan::cost, Double.POSITIVE_INFINITY, 0d, 1d, Comparator.comparing(p -> p.name));
        NamedPlan balanced = VectorPlanSelection.pick(exactFront, NamedPlan::cost, Double.POSITIVE_INFINITY, 1d, 1d, Comparator.comparing(p -> p.name));
        System.out.println("  weights (1,0) min-latency -> " + minLatency);
        System.out.println("  weights (0,1) min-money   -> " + minMoney);
        System.out.println("  weights (1,1) sum         -> " + balanced);
        assertEquals("fast-expensive", minLatency.name);
        assertEquals("cheap-slow", minMoney.name);
        assertNotEquals(minLatency.name, minMoney.name);

        System.out.println("\n========== 4) Same latencies, vary only monetary rates ==========");
        System.out.println("  Base times fixed: java=80ms, spark=25ms, postgres=60ms");
        double[] sparkRates = {0.2, 1.0, 5.0, 20.0};
        for (double sparkRate : sparkRates) {
            List<NamedPlan> scaled = Arrays.asList(
                    new NamedPlan("java", 80, 80 * 0.2),
                    new NamedPlan("spark", 25, 25 * sparkRate),
                    new NamedPlan("postgres", 60, 60 * 0.5)
            );
            System.out.println("\n  spark $/ms = " + sparkRate);
            System.out.println("    candidates: " + format(scaled));
            List<NamedPlan> front = ParetoFront.retain(scaled, NamedPlan::cost, 0d);
            System.out.println("    intermediate Pareto set: " + format(front));
            NamedPlan picked = VectorPlanSelection.pick(
                    front, NamedPlan::cost, Double.POSITIVE_INFINITY, 1d, 0d, Comparator.comparing(p -> p.name));
            NamedPlan underBudget40 = VectorPlanSelection.pick(
                    front, NamedPlan::cost, 40d, 1d, 0d, Comparator.comparing(p -> p.name));
            System.out.println("    pick min-latency: " + picked);
            System.out.println("    pick min-latency s.t. money<=40: " + underBudget40);
        }
        List<NamedPlan> cheapSpark = Arrays.asList(
                new NamedPlan("java", 80, 16),
                new NamedPlan("spark", 25, 25 * 0.2),
                new NamedPlan("postgres", 60, 30)
        );
        List<NamedPlan> dearSpark = Arrays.asList(
                new NamedPlan("java", 80, 16),
                new NamedPlan("spark", 25, 25 * 20),
                new NamedPlan("postgres", 60, 30)
        );
        NamedPlan cheapSparkBudget = VectorPlanSelection.pick(
                ParetoFront.retain(cheapSpark, NamedPlan::cost, 0d),
                NamedPlan::cost, 40d, 1d, 0d, Comparator.comparing(p -> p.name));
        NamedPlan dearSparkBudget = VectorPlanSelection.pick(
                ParetoFront.retain(dearSpark, NamedPlan::cost, 0d),
                NamedPlan::cost, 40d, 1d, 0d, Comparator.comparing(p -> p.name));
        System.out.println("\n  CHECK: money<=40 with spark rate 0.2 -> " + cheapSparkBudget.name);
        System.out.println("  CHECK: money<=40 with spark rate 20  -> " + dearSparkBudget.name);
        assertEquals("spark", cheapSparkBudget.name);
        assertEquals("postgres", dearSparkBudget.name);

        System.out.println("\n========== 5) Infeasible budget falls back to cheapest ==========");
        NamedPlan fallback = VectorPlanSelection.pick(exactFront, NamedPlan::cost, 5d, 1d, 0d, Comparator.comparing(p -> p.name));
        System.out.println("  budget=5 (none feasible) -> " + fallback);
        assertEquals("cheap-slow", fallback.name);

        System.out.println("\n========== 6) Intermediate sets while composing two stages ==========");
        List<NamedPlan> stage1 = Arrays.asList(
                new NamedPlan("s1-java", 40, 8),
                new NamedPlan("s1-spark", 10, 40),
                new NamedPlan("s1-waste", 50, 50)
        );
        List<NamedPlan> stage1Front = ParetoFront.retain(stage1, NamedPlan::cost, 0d);
        System.out.println("  Stage-1 candidates: " + format(stage1));
        System.out.println("  Stage-1 Pareto set: " + format(stage1Front));
        List<NamedPlan> composed = Arrays.asList(
                new NamedPlan("java+java", 40 + 50, 8 + 10),
                new NamedPlan("java+spark", 40 + 12, 8 + 45),
                new NamedPlan("spark+java", 10 + 50, 40 + 10),
                new NamedPlan("spark+spark", 10 + 12, 40 + 45)
        );
        System.out.println("  Stage-2 composed candidates: " + format(composed));
        List<NamedPlan> composedFront = ParetoFront.retain(composed, NamedPlan::cost, 0d);
        System.out.println("  Stage-2 Pareto set: " + format(composedFront));
        NamedPlan tight = VectorPlanSelection.pick(composedFront, NamedPlan::cost, 20d, 1d, 0d, Comparator.comparing(p -> p.name));
        NamedPlan loose = VectorPlanSelection.pick(composedFront, NamedPlan::cost, 100d, 1d, 0d, Comparator.comparing(p -> p.name));
        System.out.println("  pick budget=20  -> " + tight);
        System.out.println("  pick budget=100 -> " + loose);
        assertEquals("java+java", tight.name);
        assertEquals("spark+spark", loose.name);
        assertNotEquals(tight.name, loose.name);

        System.out.println("\n========== simulations complete ==========");
    }

    private static List<String> names(List<NamedPlan> plans) {
        return plans.stream().map(p -> p.name).collect(Collectors.toList());
    }

    private static String format(List<NamedPlan> plans) {
        return plans.stream().map(NamedPlan::toString).collect(Collectors.joining(", "));
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
            return String.format("%s(lat=%.1f, money=%.1f)", this.name, this.cost.getLatency(), this.cost.getMonetary());
        }
    }
}
