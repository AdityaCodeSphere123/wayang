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
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Corner cases for the final pick and for α-Pareto when costs are degenerate.
 */
class MultiObjectiveCornerCaseTest {

    private static final Comparator<Named> BY_NAME = Comparator.comparing(p -> p.name);

    @Test
    void emptyAndNullCollectionsDoNotThrow() {
        assertTrue(ParetoFront.retain(null, Named::cost, 0.1d).isEmpty());
        assertTrue(ParetoFront.retain(Collections.emptyList(), Named::cost, 0.1d).isEmpty());
        assertNull(VectorPlanSelection.pick(null, Named::cost, 10d, 1d, 0d, BY_NAME));
        assertNull(VectorPlanSelection.pick(Collections.emptyList(), Named::cost, 10d, 1d, 0d, BY_NAME));
    }

    @Test
    void negativeBudgetOnlyKeepsFreePlansThenFallsBack() {
        List<Named> plans = Arrays.asList(named("paid", 10, 5), named("free", 40, 0));
        Named pick = VectorPlanSelection.pick(plans, Named::cost, -3d, 1d, 0d, BY_NAME);
        assertEquals("free", pick.name);
        Named noneFree = VectorPlanSelection.pick(
                Collections.singletonList(named("paid", 10, 5)), Named::cost, -3d, 1d, 0d, BY_NAME);
        assertEquals("paid", noneFree.name);
    }

    @Test
    void zeroBudgetAcceptsOnlyZeroMoney() {
        List<Named> plans = Arrays.asList(named("free", 90, 0), named("cheap", 20, 1), named("fast", 5, 40));
        Named pick = VectorPlanSelection.pick(plans, Named::cost, 0d, 1d, 0d, BY_NAME);
        assertEquals("free", pick.name);
    }

    @Test
    void brokenWeightsFallBackToMinLatency() {
        List<Named> plans = Arrays.asList(named("slow", 80, 1), named("fast", 10, 9));
        Named pick = VectorPlanSelection.pick(
                plans, Named::cost, Double.POSITIVE_INFINITY, Double.NaN, -2d, BY_NAME);
        assertEquals("fast", pick.name);
    }

    @Test
    void nanEstimateCannotDominateAFinitePlan() {
        Named broken = named("broken", Double.NaN, Double.POSITIVE_INFINITY);
        Named ok = named("ok", 12, 8);
        List<Named> front = ParetoFront.retain(Arrays.asList(broken, ok), Named::cost, 0d);
        assertEquals(1, front.size());
        assertEquals("ok", front.get(0).name);
    }

    @Test
    void duplicateAndWeaklyDominatedPrefixesCollapse() {
        List<Named> plans = Arrays.asList(
                named("a", 4, 7),
                named("a-dup", 4, 7),
                named("weaker-same-money", 9, 7),
                named("weaker-same-lat", 4, 12)
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
