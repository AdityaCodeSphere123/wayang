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

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cross-group pruning must be exact: the final Pareto front is the same with and without it (and the same as brute
 * force), while fewer partial plans are kept.
 */
class CrossGroupPruningTest {

    @Test
    void sameFrontAsBruteForceOnSmallDags() {
        for (long seed = 1; seed <= 25; seed++) {
            final PlanDagSimulator.Graph g = PlanDagSimulator.randomConnected("dag", 8, 3, seed);
            final PlanDagSimulator.SearchResult brute = PlanDagSimulator.exhaustive(g);
            final PlanDagSimulator.SearchResult cross = PlanDagSimulator.optimize(g, 0d, 0, true);
            assertEquals(points(brute.front), points(cross.front), "seed " + seed);
        }
    }

    @Test
    void sameFrontWithFewerStatesOnBranchyDags() {
        final List<String> shapes = new ArrayList<>();
        long plain = 0, pruned = 0;
        for (int n : new int[]{12, 14, 16}) {
            for (long seed = 1; seed <= 5; seed++) {
                final PlanDagSimulator.Graph g = PlanDagSimulator.randomConnected("dag", n, 4, seed);
                final PlanDagSimulator.SearchResult exact = PlanDagSimulator.optimize(g, 0d, 0, false);
                final PlanDagSimulator.SearchResult cross = PlanDagSimulator.optimize(g, 0d, 0, true);
                assertEquals(points(exact.front), points(cross.front), "n=" + n + " seed=" + seed);
                assertTrue(cross.keptSum <= exact.keptSum);
                plain += exact.keptSum;
                pruned += cross.keptSum;
                shapes.add(n + "/" + seed);
            }
        }
        assertTrue(pruned < plain, "cross-group pruning should drop some states");
    }

    @Test
    void leftDeepJoinsAndBushyPlansKeepTheirFront() {
        for (PlanDagSimulator.Graph g : new PlanDagSimulator.Graph[]{
                PlanDagSimulator.leftDeep("ld", 8, 4),
                PlanDagSimulator.bushy("bushy", 4),
                PlanDagSimulator.complexBranching("branch", 4),
                PlanDagSimulator.threeDiamonds("3d", 4)}) {
            final PlanDagSimulator.SearchResult exact = PlanDagSimulator.optimize(g, 0d, 0, false);
            final PlanDagSimulator.SearchResult cross = PlanDagSimulator.optimize(g, 0d, 0, true);
            assertEquals(points(exact.front), points(cross.front), g.name);
        }
    }

    /** Cost vectors of a front, rounded so that summation order does not matter. */
    private static TreeSet<String> points(List<PlanDagSimulator.Partial> front) {
        final TreeSet<String> out = new TreeSet<>();
        for (PlanDagSimulator.Partial p : front) out.add(String.format("%.6f/%.6f", p.lat, p.money));
        return out;
    }
}
