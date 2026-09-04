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

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Picks one plan from a Pareto set: min latency subject to a monetary budget, or a weighted sum.
 */
public final class VectorPlanSelection {

    private VectorPlanSelection() {
    }

    public static <T> T pick(Collection<T> plans,
                             Function<T, VectorCost> costs,
                             double budget,
                             double latencyWeight,
                             double monetaryWeight,
                             Comparator<T> tieBreaker) {
        if (plans == null || plans.isEmpty()) {
            return null;
        }
        final boolean unconstrained = Double.isNaN(budget) || budget == Double.POSITIVE_INFINITY;
        List<T> feasible = plans.stream()
                .filter(plan -> unconstrained || costs.apply(plan).getMonetary() <= budget)
                .collect(Collectors.toList());
        final boolean budgetInfeasible = feasible.isEmpty();
        if (budgetInfeasible) {
            feasible = new ArrayList<>(plans);
        }
        final Comparator<T> order = Comparator
                .comparingDouble((T plan) -> score(costs.apply(plan), latencyWeight, monetaryWeight, budgetInfeasible))
                .thenComparing(tieBreaker == null ? (a, b) -> 0 : tieBreaker);
        return feasible.stream().min(order).orElse(null);
    }

    /**
     *minimize weighted cost among plans that
     * respect the monetary bound; if none do, minimize weighted cost over the whole set.
     */
    public static <T> T selectBest(Collection<T> plans,
                                   Function<T, VectorCost> costs,
                                   double budget,
                                   double latencyWeight,
                                   double monetaryWeight,
                                   Comparator<T> tieBreaker) {
        if (plans == null || plans.isEmpty()) {
            return null;
        }
        final boolean unconstrained = Double.isNaN(budget) || budget == Double.POSITIVE_INFINITY;
        List<T> feasible = plans.stream()
                .filter(plan -> unconstrained || costs.apply(plan).getMonetary() <= budget)
                .collect(Collectors.toList());
        if (feasible.isEmpty()) {
            feasible = new ArrayList<>(plans);
        }
        final Comparator<T> order = Comparator
                .comparingDouble((T plan) -> weightedCost(costs.apply(plan), latencyWeight, monetaryWeight))
                .thenComparing(tieBreaker == null ? (a, b) -> 0 : tieBreaker);
        return feasible.stream().min(order).orElse(null);
    }

    private static double weightedCost(VectorCost cost, double latencyWeight, double monetaryWeight) {
        return latencyWeight * cost.getLatency() + monetaryWeight * cost.getMonetary();
    }

    private static double score(VectorCost cost, double latencyWeight, double monetaryWeight, boolean preferCheapest) {
        if (preferCheapest) {
            return cost.getMonetary();
        }
        if (monetaryWeight == 0d) {
            return cost.getLatency();
        }
        if (latencyWeight == 0d) {
            return cost.getMonetary();
        }
        return latencyWeight * cost.getLatency() + monetaryWeight * cost.getMonetary();
    }
}
