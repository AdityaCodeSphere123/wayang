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
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ParetoFrontTest {

    @Test
    void exactFrontDropsDominatedPoints() {
        List<VectorCost> costs = Arrays.asList(
                new VectorCost(1, 10),
                new VectorCost(2, 8),
                new VectorCost(3, 9),
                new VectorCost(4, 1)
        );
        List<VectorCost> front = ParetoFront.retain(costs, c -> c, 0d);
        assertEquals(3, front.size());
        assertTrue(front.contains(costs.get(0)));
        assertTrue(front.contains(costs.get(1)));
        assertTrue(front.contains(costs.get(3)));
    }

    @Test
    void bucketingYieldsPolynomialApproximateSet() {
        List<VectorCost> costs = Arrays.asList(
                new VectorCost(10, 100),
                new VectorCost(11, 99),
                new VectorCost(12, 98),
                new VectorCost(100, 10)
        );
        List<VectorCost> approx = ParetoFront.retain(costs, c -> c, 0.2d);
        assertTrue(approx.size() <= 3);
        assertTrue(approx.stream().anyMatch(c -> c.getLatency() >= 90));
    }

    @Test
    void emptyOrSingletonCollectionsAreUnchanged() {
        assertTrue(ParetoFront.retain(java.util.Collections.<VectorCost>emptyList(), c -> c, 0d).isEmpty());
        List<VectorCost> single = Arrays.asList(new VectorCost(1, 1));
        assertEquals(1, ParetoFront.retain(single, c -> c, 0d).size());
    }

    @Test
    void equivalentCostsKeepASingleRepresentative() {
        VectorCost a = new VectorCost(5, 5);
        VectorCost b = new VectorCost(5, 5);
        List<VectorCost> front = ParetoFront.retain(Arrays.asList(a, b), c -> c, 0d);
        assertEquals(1, front.size());
    }

    @Test
    void skipsNullsAndDropsWeaklyDominatedEquals() {
        VectorCost keep = new VectorCost(2, 4);
        VectorCost sameMoneySlower = new VectorCost(5, 4);
        VectorCost sameLatencyDearer = new VectorCost(2, 9);
        List<VectorCost> front = ParetoFront.retain(
                Arrays.asList(null, keep, sameMoneySlower, sameLatencyDearer, null),
                c -> c,
                0d);
        assertEquals(1, front.size());
        assertEquals(keep, front.get(0));
    }

    @Test
    void tinyEpsilonMatchesExactFront() {
        List<VectorCost> costs = Arrays.asList(
                new VectorCost(1, 10),
                new VectorCost(2, 8),
                new VectorCost(4, 1)
        );
        assertEquals(
                ParetoFront.retain(costs, c -> c, 0d),
                ParetoFront.retain(costs, c -> c, 1e-18d)
        );
    }

    @Test
    void zeroCostAndInfiniteCostDoNotCrash() {
        List<VectorCost> costs = Arrays.asList(
                new VectorCost(0, 0),
                new VectorCost(0, 12),
                new VectorCost(Double.POSITIVE_INFINITY, Double.NaN),
                new VectorCost(3, 3)
        );
        List<VectorCost> front = ParetoFront.retain(costs, c -> c, 0.1d);
        assertEquals(1, front.size());
        assertEquals(0d, front.get(0).getLatency());
        assertEquals(0d, front.get(0).getMonetary());
    }
}
