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

package org.apache.wayang.core.optimizer.costs;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

class VectorCostTest {

    @Test
    void dominanceIsPareto() {
        VectorCost fastExpensive = new VectorCost(10, 100);
        VectorCost slowCheap = new VectorCost(50, 10);
        VectorCost slowExpensive = new VectorCost(50, 100);

        assertTrue(fastExpensive.dominates(slowExpensive));
        assertTrue(slowCheap.dominates(slowExpensive));
        assertFalse(fastExpensive.dominates(slowCheap));
        assertFalse(slowCheap.dominates(fastExpensive));
    }

    @Test
    void alphaDominanceRelaxesExactPareto() {
        VectorCost better = new VectorCost(10, 10);
        VectorCost slightlyWorse = new VectorCost(11, 11);
        VectorCost muchWorse = new VectorCost(15, 15);
        assertTrue(better.dominates(slightlyWorse));
        assertTrue(slightlyWorse.approximatelyDominates(better, 1.2d));
        assertFalse(muchWorse.approximatelyDominates(better, 1.2d));
    }

    @Test
    void logBucketsGroupSimilarTradeoffs() {
        VectorCost a = new VectorCost(10, 10);
        VectorCost b = new VectorCost(11, 11);
        long[] bucketA = a.logBuckets(0.5d);
        long[] bucketB = b.logBuckets(0.5d);
        assertEquals(bucketA[0], bucketB[0]);
        assertEquals(bucketA[1], bucketB[1]);
    }

    @Test
    void sanitizesNonFiniteValues() {
        VectorCost nan = new VectorCost(Double.NaN, Double.NEGATIVE_INFINITY);
        assertEquals(Double.POSITIVE_INFINITY, nan.getLatency());
        assertEquals(Double.POSITIVE_INFINITY, nan.getMonetary());
        assertFalse(nan.isFinite());
        assertFalse(nan.dominates(null));
        VectorCost real = new VectorCost(10, 10);
        assertFalse(nan.dominates(real), "a broken estimate must not look free and win");
        assertTrue(real.dominates(nan));
        VectorCost clipped = new VectorCost(-3, -0.5);
        assertEquals(0d, clipped.getLatency());
        assertEquals(0d, clipped.getMonetary());
    }

    @Test
    void tinyEpsilonIsTreatedAsExact() {
        assertEquals(0d, VectorCost.finiteEpsilon(-1d));
        assertEquals(0d, VectorCost.finiteEpsilon(Double.NaN));
        assertEquals(0d, VectorCost.finiteEpsilon(1e-18d));
        assertEquals(0.1d, VectorCost.finiteEpsilon(0.1d), 0d);
        VectorCost a = new VectorCost(10, 20);
        long[] exact = a.logBuckets(0d);
        long[] tiny = a.logBuckets(1e-18d);
        assertEquals(exact[0], tiny[0]);
        assertEquals(exact[1], tiny[1]);
    }

    @Test
    void alphaDominanceIsNullSafeAndIgnoresBadAlpha() {
        VectorCost a = new VectorCost(10, 10);
        assertFalse(a.approximatelyDominates(null, 1.1d));
        assertTrue(a.approximatelyDominates(a, Double.NaN));
        assertTrue(a.approximatelyDominates(new VectorCost(11, 11), 1.2d));
    }
}
