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
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Correctness checks for the VectorPlanSelection class.
 */
class TrummerKochCorrectnessTest {

    private static final Comparator<Named> BY_NAME = Comparator.comparing(p -> p.name);
    private static final double EPS = 1e-9;

    @Test
    void exactParetoIsMinimalAndComplete() {
        List<Named> plans = Arrays.asList(
                named("a", 1, 10),
                named("b", 2, 8),
                named("c", 3, 9),
                named("d", 4, 1),
                named("dup", 2, 8),
                named("dom", 5, 12)
        );
        List<Named> front = ParetoFront.retain(plans, Named::cost, 0d);
        assertNoStrictlyDominatedPair(front);
        assertAlphaApproximateCoverage(plans, front, 1d);
        assertFalse(names(front).contains("c"));
        assertFalse(names(front).contains("dom"));
        assertTrue(names(front).contains("a"));
        assertTrue(names(front).contains("b") || names(front).contains("dup"));
        assertTrue(names(front).contains("d"));
    }

    @Test
    void sameLogBucketImpliesMutualAlphaDominance() {
        final double epsilon = 0.25d;
        final double alpha = 1d + epsilon;
        Random rng = new Random(7);
        for (int i = 0; i < 400; i++) {
            VectorCost x = randomCost(rng);
            VectorCost y = randomCost(rng);
            if (!Arrays.equals(x.logBuckets(epsilon), y.logBuckets(epsilon))) {
                continue;
            }
            assertTrue(x.approximatelyDominates(y, alpha + EPS),
                    x + " should α-dominate " + y);
            assertTrue(y.approximatelyDominates(x, alpha + EPS),
                    y + " should α-dominate " + x);
        }
    }

    @Test
    void approximateFrontIsAlphaApproximateParetoSet() {
        Random rng = new Random(11);
        for (double epsilon : Arrays.asList(0d, 0.1d, 0.25d, 0.5d, 1d)) {
            double alpha = 1d + epsilon;
            for (int trial = 0; trial < 40; trial++) {
                List<Named> plans = randomPlans(rng, 30 + trial);
                List<Named> front = ParetoFront.retain(plans, Named::cost, epsilon, BY_NAME);
                assertNoStrictlyDominatedPair(front);
                assertAlphaApproximateCoverage(plans, front, alpha);
                assertAtMostOnePlanPerBucket(front, epsilon);
            }
        }
    }

    @Test
    void twoDimensionalFrontSizeIsLogBounded() {
        List<Named> dense = new ArrayList<>();
        for (int i = 1; i <= 200; i++) {
            dense.add(named("p" + i, i, 201 - i));
        }
        final double epsilon = 0.2d;
        List<Named> approx = ParetoFront.retain(dense, Named::cost, epsilon);
        assertEquals(200, ParetoFront.retain(dense, Named::cost, 0d).size());
        double cMax = 200d;
        long bucketsPerAxis = 1L + (long) Math.ceil(Math.log(cMax) / Math.log(1d + epsilon));
        assertTrue(approx.size() < dense.size());
        assertTrue(approx.size() <= 4 * bucketsPerAxis,
                "2-D RTA front is O(log_{1+ε} C) after the Pareto staircase, was "
                        + approx.size() + " vs 4·" + bucketsPerAxis);
    }

    @Test
    void weightedSelectBestOnExactFrontMatchesExhaustiveOptimum() {
        Random rng = new Random(23);
        for (int trial = 0; trial < 50; trial++) {
            List<Named> plans = randomPlans(rng, 25);
            List<Named> exact = ParetoFront.retain(plans, Named::cost, 0d, BY_NAME);
            double[][] weights = {{1, 0}, {0, 1}, {1, 1}, {2, 1}, {0.3, 0.7}};
            for (double[] w : weights) {
                Named fromFront = VectorPlanSelection.selectBest(
                        exact, Named::cost, Double.POSITIVE_INFINITY, w[0], w[1], BY_NAME);
                Named fromAll = VectorPlanSelection.selectBest(
                        plans, Named::cost, Double.POSITIVE_INFINITY, w[0], w[1], BY_NAME);
                assertEquals(weighted(fromAll.cost, w[0], w[1]), weighted(fromFront.cost, w[0], w[1]), EPS);
            }
        }
    }

    @Test
    void rtaWeightedPickIsAlphaApproximate() {
        Random rng = new Random(29);
        for (double epsilon : Arrays.asList(0.1d, 0.5d, 1d)) {
            double alpha = 1d + epsilon;
            for (int trial = 0; trial < 40; trial++) {
                List<Named> plans = randomPlans(rng, 40);
                List<Named> approx = ParetoFront.retain(plans, Named::cost, epsilon, BY_NAME);
                for (double[] w : Arrays.asList(new double[]{1, 0}, new double[]{0, 1}, new double[]{1, 1})) {
                    Named picked = VectorPlanSelection.selectBest(
                            approx, Named::cost, Double.POSITIVE_INFINITY, w[0], w[1], BY_NAME);
                    Named optimal = VectorPlanSelection.selectBest(
                            plans, Named::cost, Double.POSITIVE_INFINITY, w[0], w[1], BY_NAME);
                    double rho = weighted(picked.cost, w[0], w[1]) / weighted(optimal.cost, w[0], w[1]);
                    assertTrue(rho <= alpha + 1e-6,
                            "RTA weighted guarantee: ρ=" + rho + " α=" + alpha + " w=" + Arrays.toString(w));
                }
            }
        }
    }

    @Test
    void additiveDpPruneAfterEachConcatPreservesWeightedAlphaGuarantee() {
        Random rng = new Random(31);
        final double epsilon = 0.15d;
        final double alpha = 1d + epsilon;
        for (int trial = 0; trial < 25; trial++) {
            List<List<Named>> stages = new ArrayList<>();
            for (int s = 0; s < 4; s++) {
                List<Named> choices = new ArrayList<>();
                for (int k = 0; k < 5; k++) {
                    choices.add(named("s" + s + "p" + k, 1 + rng.nextInt(40), 1 + rng.nextInt(40)));
                }
                stages.add(choices);
            }

            List<Named> exhaustive = cartesianSum(stages);
            List<Named> dpFront = additiveDp(stages, epsilon);
            assertAlphaApproximateCoverage(exhaustive, dpFront, alpha);

            Named picked = VectorPlanSelection.selectBest(
                    dpFront, Named::cost, Double.POSITIVE_INFINITY, 1d, 1d, BY_NAME);
            Named optimal = VectorPlanSelection.selectBest(
                    exhaustive, Named::cost, Double.POSITIVE_INFINITY, 1d, 1d, BY_NAME);
            double rho = weighted(picked.cost, 1d, 1d) / weighted(optimal.cost, 1d, 1d);
            assertTrue(rho <= alpha + 1e-6, "additive DP ρ=" + rho + " α=" + alpha);
        }
    }

    @Test
    void selectBestRespectsBudgetWhenFeasibleElseIgnoresBounds() {
        List<Named> front = ParetoFront.retain(Arrays.asList(
                named("cheap-slow", 200, 10),
                named("java", 100, 20),
                named("hybrid", 45, 50),
                named("fast", 20, 120)
        ), Named::cost, 0d);

        Named under50 = VectorPlanSelection.selectBest(front, Named::cost, 50d, 1d, 0d, BY_NAME);
        assertEquals("hybrid", under50.name);
        assertTrue(under50.cost.getMonetary() <= 50d);

        Named infeasible = VectorPlanSelection.selectBest(front, Named::cost, 5d, 1d, 0d, BY_NAME);
        assertEquals("fast", infeasible.name);

        Named infeasibleMoney = VectorPlanSelection.selectBest(front, Named::cost, 5d, 0d, 1d, BY_NAME);
        assertEquals("cheap-slow", infeasibleMoney.name);
    }

    @Test
    void boundedMoqoAlphaFrontNeedNotContainNearOptimalPlan() {
        final double epsilon = 1d;
        final double alpha = 1d + epsilon;
        final double budget = 10d;
        Named justFeasible = named("optimal-feasible", 11, 10);
        Named overBudgetTwin = named("over-budget-twin", 10, 15);
        Named feasibleSlow = named("feasible-slow", 80, 9);
        assertEquals(
                Arrays.toString(justFeasible.cost.logBuckets(epsilon)),
                Arrays.toString(overBudgetTwin.cost.logBuckets(epsilon)),
                "Fig. 8 needs the feasible optimum and its twin in one coarsened cell");
        assertTrue(overBudgetTwin.cost.getLatency() < justFeasible.cost.getLatency());

        List<Named> plans = Arrays.asList(justFeasible, overBudgetTwin, feasibleSlow);
        List<Named> approx = ParetoFront.retain(plans, Named::cost, epsilon, BY_NAME);
        assertFalse(names(approx).contains("optimal-feasible"));
        assertTrue(names(approx).contains("over-budget-twin"));

        Named boundedPick = VectorPlanSelection.selectBest(approx, Named::cost, budget, 1d, 0d, BY_NAME);
        Named trueOpt = VectorPlanSelection.selectBest(plans, Named::cost, budget, 1d, 0d, BY_NAME);
        assertEquals("optimal-feasible", trueOpt.name);
        assertEquals("feasible-slow", boundedPick.name);
        double rho = weighted(boundedPick.cost, 1d, 0d) / weighted(trueOpt.cost, 1d, 0d);
        assertTrue(rho > alpha,
                "Fig. 8: bounded RTA relative cost " + rho + " exceeds α=" + alpha
                        + "; IRA is not implemented");
    }

    @Test
    void wayangPickFallsBackToCheapestMoneyWhenBudgetImpossible() {
        List<Named> front = Arrays.asList(named("cheap-slow", 200, 10), named("fast", 20, 120));
        Named wayang = VectorPlanSelection.pick(front, Named::cost, 5d, 1d, 0d, BY_NAME);
        Named paper = VectorPlanSelection.selectBest(front, Named::cost, 5d, 1d, 0d, BY_NAME);
        assertEquals("cheap-slow", wayang.name);
        assertEquals("fast", paper.name);
        assertNotEquals(wayang.name, paper.name);
    }

    private static void assertAlphaApproximateCoverage(List<Named> universe, List<Named> front, double alpha) {
        for (Named original : universe) {
            boolean covered = front.stream().anyMatch(rep ->
                    rep.cost.approximatelyDominates(original.cost, alpha + 1e-6));
            assertTrue(covered, "No α-representative for " + original + " in " + front + " α=" + alpha);
        }
    }

    private static void assertNoStrictlyDominatedPair(List<Named> front) {
        for (Named a : front) {
            for (Named b : front) {
                if (a == b) {
                    continue;
                }
                assertFalse(a.cost.dominates(b.cost), a + " dominates " + b + " on the front");
            }
        }
    }

    private static void assertAtMostOnePlanPerBucket(List<Named> front, double epsilon) {
        if (epsilon <= 0d) {
            return;
        }
        Set<String> seen = new HashSet<>();
        for (Named plan : front) {
            long[] b = plan.cost.logBuckets(epsilon);
            String key = b[0] + "," + b[1];
            assertTrue(seen.add(key), "two front members share bucket " + key);
        }
    }

    private static List<Named> additiveDp(List<List<Named>> stages, double epsilon) {
        List<Named> front = new ArrayList<>(stages.get(0));
        front = ParetoFront.retain(front, Named::cost, epsilon, BY_NAME);
        for (int s = 1; s < stages.size(); s++) {
            List<Named> combined = new ArrayList<>();
            int i = 0;
            for (Named left : front) {
                for (Named right : stages.get(s)) {
                    combined.add(new Named(
                            left.name + "+" + right.name + "#" + i++,
                            left.cost.getLatency() + right.cost.getLatency(),
                            left.cost.getMonetary() + right.cost.getMonetary()
                    ));
                }
            }
            front = ParetoFront.retain(combined, Named::cost, epsilon, BY_NAME);
        }
        return front;
    }

    private static List<Named> cartesianSum(List<List<Named>> stages) {
        List<Named> acc = new ArrayList<>(stages.get(0));
        for (int s = 1; s < stages.size(); s++) {
            List<Named> next = new ArrayList<>();
            int i = 0;
            for (Named left : acc) {
                for (Named right : stages.get(s)) {
                    next.add(new Named(
                            left.name + "+" + right.name + "#" + i++,
                            left.cost.getLatency() + right.cost.getLatency(),
                            left.cost.getMonetary() + right.cost.getMonetary()
                    ));
                }
            }
            acc = next;
        }
        return acc;
    }

    private static List<Named> randomPlans(Random rng, int n) {
        List<Named> plans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            plans.add(named("p" + i, 1 + rng.nextInt(80), 1 + rng.nextInt(80)));
        }
        return plans;
    }

    private static VectorCost randomCost(Random rng) {
        return new VectorCost(0.5 + rng.nextDouble() * 200, 0.5 + rng.nextDouble() * 200);
    }

    private static double weighted(VectorCost cost, double wL, double wM) {
        return wL * cost.getLatency() + wM * cost.getMonetary();
    }

    private static List<String> names(List<Named> plans) {
        return plans.stream().map(p -> p.name).collect(Collectors.toList());
    }

    private static Named named(String name, double latency, double monetary) {
        return new Named(name, latency, monetary);
    }

    private static final class Named {
        final String name;
        final VectorCost cost;

        Named(String name, double latency, double monetary) {
            this.name = name;
            this.cost = new VectorCost(latency, monetary);
        }

        VectorCost cost() {
            return this.cost;
        }

        @Override
        public String toString() {
            return String.format("%s(%.3f,%.3f)", this.name, this.cost.getLatency(), this.cost.getMonetary());
        }
    }
}
