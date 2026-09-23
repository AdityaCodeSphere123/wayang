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
import java.util.List;
import java.util.Random;

import static org.apache.wayang.core.optimizer.enumeration.PlanDagSimulator.Graph;
import static org.apache.wayang.core.optimizer.enumeration.PlanDagSimulator.Partial;
import static org.apache.wayang.core.optimizer.enumeration.PlanDagSimulator.SearchResult;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Gaps that the first test rounds missed: null bucketing, one platform, correlated
 * objectives, antichain fronts, weighted pick, and α-coverage of exhaustive DP.
 */
class OptimizerMissedCasesTest {

    private static final Comparator<Named> BY_NAME = Comparator.comparing(p -> p.name);

    @Test
    void groupByBucketsSkipsNulls() {
        List<Named> plans = Arrays.asList(null, named("a", 10, 10), named("b", 11, 11));
        assertTrue(ParetoFront.groupByBuckets(null, Named::cost, 0.2d).isEmpty());
        assertEquals(1, ParetoFront.groupByBuckets(plans, Named::cost, 0.5d).size());
    }

    @Test
    void retainNeverReturnsEmptyWhenInputHasAValidCost() {
        List<Named> plans = Arrays.asList(named("only", 3, 4));
        assertEquals(1, ParetoFront.retain(plans, Named::cost, 0.1d).size());
        List<Named> mixed = Arrays.asList(named("ok", 3, 4), named("null-cost", Double.NaN, Double.NaN));
        List<Named> front = ParetoFront.retain(mixed, Named::cost, 0d);
        assertEquals(1, front.size());
        assertEquals("ok", front.get(0).name);
    }

    @Test
    void singlePlatformExactAndAlphaPickTheSamePlan() {
        Graph g = PlanDagSimulator.chain("one-plat", 8, 1);
        SearchResult exact = PlanDagSimulator.optimize(g, 0d);
        SearchResult alpha = PlanDagSimulator.optimize(g, 0.25d);
        assertEquals(1, exact.front.size());
        assertEquals(1, alpha.front.size());
        assertEquals(exact.front.get(0).cost.getLatency(), alpha.front.get(0).cost.getLatency(), 1e-9);
        assertEquals(exact.generated, alpha.generated);
    }

    @Test
    void correlatedObjectivesCollapseToASingleWinnerPerCut() {
        Graph g = PlanDagSimulator.chain("corr", 6, 4);
        for (int i = 0; i < g.nOps(); i++) {
            for (int p = 0; p < g.nPlats; p++) {
                g.money[i][p] = g.lat[i][p];
            }
        }
        SearchResult exact = PlanDagSimulator.optimize(g, 0d);
        assertTrue(exact.finalKept <= g.nPlats,
                "diagonal costs should not grow a 2-D front, kept=" + exact.finalKept);
        Partial pick = PlanDagSimulator.pick(exact.front, Double.POSITIVE_INFINITY);
        for (Partial other : exact.front) {
            assertTrue(pick.cost.getLatency() <= other.cost.getLatency() + 1e-9);
        }
    }

    @Test
    void exactFrontIsAnAntichainOnRandomClouds() {
        Random rng = new Random(99);
        for (int trial = 0; trial < 50; trial++) {
            List<Named> cloud = new ArrayList<>();
            for (int i = 0; i < 40; i++) {
                cloud.add(named("p" + i, 1 + rng.nextInt(80), 1 + rng.nextInt(80)));
            }
            List<Named> front = ParetoFront.retain(cloud, Named::cost, 0d, BY_NAME);
            for (Named a : front) {
                for (Named b : front) {
                    if (a != b) {
                        assertFalse(a.cost.dominates(b.cost), a.name + " vs " + b.name);
                    }
                }
            }
            assertFalse(front.isEmpty());
        }
    }

    @Test
    void additiveAlphaFrontCoversExhaustiveOnSmallRandomStages() {
        Random rng = new Random(5);
        for (int trial = 0; trial < 20; trial++) {
            int n = 4;
            int p = 3;
            Graph g = PlanDagSimulator.randomConnected("cov-" + trial, n, p, rng.nextLong());
            if (g.cartesian() > 20_000) {
                continue;
            }
            SearchResult brute = PlanDagSimulator.exhaustive(g);
            SearchResult alpha = PlanDagSimulator.optimize(g, 0.15d);
            for (Partial orig : brute.front) {
                boolean covered = alpha.front.stream().anyMatch(rep ->
                        rep.cost.approximatelyDominates(orig.cost, 1.15d + 1e-6)
                                || rep.cost.approximatelyDominates(orig.cost, Math.pow(1.15d, n) + 1e-6));
                assertTrue(covered, g.name + " missing cover for " + orig.path);
            }
        }
    }

    @Test
    void weightedPickMinMoneyAndBalancedDifferFromMinLatency() {
        List<Named> front = ParetoFront.retain(Arrays.asList(
                named("fast", 10, 90),
                named("mid", 40, 40),
                named("cheap", 90, 10)
        ), Named::cost, 0d, BY_NAME);
        Named minL = VectorPlanSelection.pick(front, Named::cost, Double.POSITIVE_INFINITY, 1d, 0d, BY_NAME);
        Named minM = VectorPlanSelection.pick(front, Named::cost, Double.POSITIVE_INFINITY, 0d, 1d, BY_NAME);
        Named sum = VectorPlanSelection.pick(front, Named::cost, Double.POSITIVE_INFINITY, 1d, 1d, BY_NAME);
        assertEquals("fast", minL.name);
        assertEquals("cheap", minM.name);
        assertEquals("mid", sum.name);
    }

    @Test
    void epsilonZeroMatchesExactOnConversionHeavyDiamond() {
        Graph g = PlanDagSimulator.diamond("conv-diamond", 4);
        SearchResult a = PlanDagSimulator.optimize(g, 0d);
        SearchResult b = PlanDagSimulator.optimize(g, 0d);
        Partial pa = PlanDagSimulator.pick(a.front, Double.POSITIVE_INFINITY);
        Partial pb = PlanDagSimulator.pick(b.front, Double.POSITIVE_INFINITY);
        assertEquals(pa.path, pb.path);
        SearchResult brute = PlanDagSimulator.exhaustive(g);
        Partial pe = PlanDagSimulator.pick(brute.front, Double.POSITIVE_INFINITY);
        assertEquals(pe.cost.getLatency(), pa.cost.getLatency(), 1e-9);
    }

    @Test
    void coarseEpsilonDoesNotReturnAnEmptyFront() {
        Graph g = PlanDagSimulator.chain("coarse", 10, 8);
        for (double eps : Arrays.asList(0d, 0.1d, 1d, 5d, 50d)) {
            SearchResult r = PlanDagSimulator.optimize(g, eps);
            assertFalse(r.front.isEmpty(), "ε=" + eps);
            assertTrue(r.finalKept >= 1);
        }
    }

    @Test
    void defaultScalarVectorEstimateLiesOnTheDiagonal() {
        VectorCost mapped = new VectorCost(7, 7);
        assertEquals(mapped.getLatency(), mapped.getMonetary());
        List<Named> plans = Arrays.asList(
                named("a", 10, 10),
                named("b", 20, 20),
                named("c", 15, 15)
        );
        List<Named> front = ParetoFront.retain(plans, Named::cost, 0d, BY_NAME);
        assertEquals(1, front.size());
        assertEquals("a", front.get(0).name);
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
    }
}
