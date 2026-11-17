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

import org.apache.wayang.core.api.Configuration;
import org.apache.wayang.core.optimizer.ProbabilisticDoubleInterval;
import org.apache.wayang.core.optimizer.costs.ObjectiveSettings;
import org.apache.wayang.core.optimizer.costs.VectorCost;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks the properties the optimizer's correctness argument relies on: additive costs, the global ε guarantee,
 * and the final pick under budget/deadline.
 */
class MultiObjectiveSoundnessTest {

    // ---------------------------------------------------------------- additivity

    /**
     * The scalar path squashes intervals with the geometric mean. That is not additive: appending the same suffix to
     * two prefixes can flip which one is better, so pruning with it could drop the prefix of the optimal plan.
     * The expected value (used by VectorEstimatableCost) is linear and keeps the order.
     */
    @Test
    void geometricMeanCanFlipOrderButMeanCannot() {
        final ProbabilisticDoubleInterval p = new ProbabilisticDoubleInterval(1, 100, 1d);
        final ProbabilisticDoubleInterval q = new ProbabilisticDoubleInterval(20, 20, 1d);
        final ProbabilisticDoubleInterval suffix = new ProbabilisticDoubleInterval(20, 20, 1d);

        // Geometric mean: p (10) beats q (20) alone, but loses after the common suffix (≈50.2 vs 40).
        assertTrue(p.getGeometricMeanEstimate() < q.getGeometricMeanEstimate());
        assertTrue(p.plus(suffix).getGeometricMeanEstimate() > q.plus(suffix).getGeometricMeanEstimate());

        // Mean: order is kept, and the cost of the whole is the sum of the parts.
        final boolean before = p.getAverageEstimate() < q.getAverageEstimate();
        final boolean after = p.plus(suffix).getAverageEstimate() < q.plus(suffix).getAverageEstimate();
        assertEquals(before, after);
        assertEquals(p.getAverageEstimate() + suffix.getAverageEstimate(), p.plus(suffix).getAverageEstimate(), 1e-9);
    }

    // ---------------------------------------------------------------- ε splitting

    @Test
    void globalEpsilonIsSplitOverSteps() {
        final ObjectiveSettings global = settings("0.1", "global");
        final double step = global.stepEpsilon(20);
        assertEquals(1.1, Math.pow(1 + step, 20), 1e-9);
        assertTrue(step < 0.01);

        final ObjectiveSettings perStep = settings("0.1", "per-step");
        assertEquals(0.1, perStep.stepEpsilon(20), 1e-12);

        assertEquals(0d, settings("0", "global").stepEpsilon(20), 0d);
        assertEquals(0.1, global.stepEpsilon(1), 1e-12);
    }

    @Test
    void epsilonStepsOverrideWins() {
        final Configuration configuration = new Configuration();
        configuration.setProperty(ObjectiveSettings.EPSILON, "0.1");
        configuration.setProperty(ObjectiveSettings.EPSILON_STEPS, "5");
        final double step = ObjectiveSettings.of(configuration).stepEpsilon(100);
        assertEquals(1.1, Math.pow(1 + step, 5), 1e-9);
    }

    /**
     * Additive chain DP: at every stage each prefix is extended with every option, then pruned. With the global
     * mode the final front covers the exact front within 1 + ε on both axes; per-step ε is allowed to compound.
     */
    @Test
    void globalModeKeepsTheFinalFrontWithinOnePlusEpsilon() {
        final double epsilon = 0.1;
        final int stages = 12;
        double worstGlobal = 1d, worstPerStep = 1d;
        for (int seed = 0; seed < 60; seed++) {
            final List<List<VectorCost>> options = randomChain(seed, stages, 4);
            final List<VectorCost> exact = chainFront(options, 0d);
            final List<VectorCost> global = chainFront(options, settings("0.1", "global").stepEpsilon(stages));
            final List<VectorCost> perStep = chainFront(options, epsilon);
            worstGlobal = Math.max(worstGlobal, coverFactor(exact, global));
            worstPerStep = Math.max(worstPerStep, coverFactor(exact, perStep));
            assertTrue(global.size() <= exact.size());
        }
        assertTrue(worstGlobal <= 1 + epsilon + 1e-9, "global mode broke the guarantee: " + worstGlobal);
        assertTrue(worstPerStep >= worstGlobal, "per-step should not be more precise than global");
    }

    private static List<List<VectorCost>> randomChain(int seed, int stages, int optionsPerStage) {
        final Random random = new Random(seed);
        final List<List<VectorCost>> chain = new ArrayList<>();
        for (int s = 0; s < stages; s++) {
            final List<VectorCost> stage = new ArrayList<>();
            for (int o = 0; o < optionsPerStage; o++) {
                // Fast-and-dear vs slow-and-cheap, with noise.
                final double speed = 1 + random.nextInt(8);
                stage.add(new VectorCost(100d / speed + random.nextDouble() * 10, speed * 10 + random.nextDouble() * 10));
            }
            chain.add(stage);
        }
        return chain;
    }

    private static List<VectorCost> chainFront(List<List<VectorCost>> chain, double stepEpsilon) {
        List<VectorCost> front = ParetoFront.retain(chain.get(0), Function.identity(), stepEpsilon);
        for (int s = 1; s < chain.size(); s++) {
            final List<VectorCost> next = new ArrayList<>();
            for (VectorCost prefix : front) {
                for (VectorCost option : chain.get(s)) {
                    next.add(new VectorCost(prefix.getLatency() + option.getLatency(),
                            prefix.getMonetary() + option.getMonetary()));
                }
            }
            front = ParetoFront.retain(next, Function.identity(), stepEpsilon);
        }
        return front;
    }

    /** Smallest α such that every exact point is α-dominated by some approximate point. */
    private static double coverFactor(List<VectorCost> exact, List<VectorCost> approx) {
        double worst = 1d;
        for (VectorCost e : exact) {
            double best = Double.POSITIVE_INFINITY;
            for (VectorCost a : approx) {
                best = Math.min(best, Math.max(a.getLatency() / e.getLatency(), a.getMonetary() / e.getMonetary()));
            }
            worst = Math.max(worst, best);
        }
        return worst;
    }

    private static ObjectiveSettings settings(String epsilon, String mode) {
        final Configuration configuration = new Configuration();
        configuration.setProperty(ObjectiveSettings.EPSILON, epsilon);
        configuration.setProperty(ObjectiveSettings.EPSILON_MODE, mode);
        return ObjectiveSettings.of(configuration);
    }

    // ---------------------------------------------------------------- final pick

    private static final VectorCost FAST_DEAR = new VectorCost(10, 100);
    private static final VectorCost MID = new VectorCost(20, 40);
    private static final VectorCost SLOW_CHEAP = new VectorCost(50, 10);
    private static final List<VectorCost> FRONT = Arrays.asList(FAST_DEAR, MID, SLOW_CHEAP);

    private static VectorCost pick(List<VectorCost> plans, double budget, double deadline,
                                   double wL, double wM, boolean normalize) {
        return VectorPlanSelection.pick(plans, Function.identity(), budget, deadline, wL, wM, normalize, null);
    }

    @Test
    void deadlinePicksCheapestPlanThatIsFastEnough() {
        assertSame(MID, pick(FRONT, Double.POSITIVE_INFINITY, 25, 0, 1, true));
        assertSame(SLOW_CHEAP, pick(FRONT, Double.POSITIVE_INFINITY, 60, 0, 1, true));
    }

    @Test
    void budgetAndDeadlineTogether() {
        // Feasible: latency <= 30 and money <= 50 -> only MID.
        assertSame(MID, pick(FRONT, 50, 30, 1, 0, true));
    }

    @Test
    void infeasibleBoundsFallBackToLeastViolation() {
        // Only a deadline that nothing meets: fastest plan.
        assertSame(FAST_DEAR, pick(FRONT, Double.POSITIVE_INFINITY, 5, 0, 1, true));
        // Only a budget that nothing meets: cheapest plan.
        assertSame(SLOW_CHEAP, pick(FRONT, 5, Double.POSITIVE_INFINITY, 1, 0, true));
        // Both: FAST_DEAR violates money by 19x, MID by 7x and latency 3x, SLOW_CHEAP by latency 9x -> MID.
        assertSame(MID, pick(FRONT, 5, 5, 1, 0, true));
    }

    @Test
    void latencyTiesGoToTheCheaperPlan() {
        final VectorCost dear = new VectorCost(10, 100);
        final VectorCost cheap = new VectorCost(10, 50);
        assertSame(cheap, pick(Arrays.asList(dear, cheap), Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 1, 0, true));
    }

    @Test
    void normalizedWeightsDoNotDependOnUnits() {
        // Same front with money in cents instead of dollars.
        final List<VectorCost> cents = Arrays.asList(
                new VectorCost(10, 10000), new VectorCost(20, 4000), new VectorCost(50, 1000));
        final VectorCost inDollars = pick(FRONT, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 1, 1, true);
        final VectorCost inCents = pick(cents, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 1, 1, true);
        assertEquals(FRONT.indexOf(inDollars), cents.indexOf(inCents));

        // Raw weights flip with the unit, which is why normalisation is the default.
        final VectorCost rawDollars = pick(FRONT, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 1, 1, false);
        final VectorCost rawCents = pick(cents, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 1, 1, false);
        assertNotEquals(FRONT.indexOf(rawDollars), cents.indexOf(rawCents));
    }

    @Test
    void oldEntryPointBehavesAsBefore() {
        assertSame(MID, VectorPlanSelection.pick(FRONT, Function.identity(), 50, 1, 0, null));
        assertSame(SLOW_CHEAP, VectorPlanSelection.pick(FRONT, Function.identity(), 1, 1, 0, null));
    }

    // ---------------------------------------------------------------- configuration

    @Test
    void settingCostModelPropertySwitchesTheModel() {
        final Configuration configuration = new Configuration();
        configuration.setProperty("wayang.core.optimizer.cost.model", "default");
        assertEquals("DefaultEstimatableCost", configuration.getCostModel().getClass().getSimpleName());
        configuration.setProperty("wayang.core.optimizer.cost.model", "vector");
        assertEquals("VectorEstimatableCost", configuration.getCostModel().getClass().getSimpleName());
    }
}
