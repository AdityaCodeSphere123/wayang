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
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Picks one plan from a Pareto set.
 *
 * <ol>
 *   <li>Feasible plans: {@code monetary <= budget} and {@code latency <= deadline}.</li>
 *   <li>Among them, minimise the objective:
 *     <ul>
 *       <li>only latency weighted (default {@code (1, 0)}): fastest, ties go to the cheaper plan;</li>
 *       <li>only money weighted: cheapest, ties go to the faster plan;</li>
 *       <li>both weighted: {@code wL·latency/L* + wM·money/M*}. With normalisation, {@code L*} and {@code M*}
 *           are the best latency and money among the candidates, so the weights are unit-free
 *           (milliseconds and dollars are not comparable otherwise). Without it, {@code L* = M* = 1}.</li>
 *     </ul></li>
 *   <li>If no plan is feasible, take the plan with the smallest relative violation
 *       {@code max((money-B)/B, (latency-T)/T)}. With only a budget this is the cheapest plan; with only a
 *       deadline it is the fastest one.</li>
 * </ol>
 */
public final class VectorPlanSelection {

    private static final double TINY = 1e-12;

    private VectorPlanSelection() {
    }

    /**
     * Backwards-compatible entry point: budget only, raw (unnormalised) weights.
     */
    public static <T> T pick(Collection<T> plans,
                             Function<T, VectorCost> costs,
                             double budget,
                             double latencyWeight,
                             double monetaryWeight,
                             Comparator<T> tieBreaker) {
        return pick(plans, costs, budget, Double.POSITIVE_INFINITY, latencyWeight, monetaryWeight, false, tieBreaker);
    }

    public static <T> T pick(Collection<T> plans,
                             Function<T, VectorCost> costs,
                             double budget,
                             double deadline,
                             double latencyWeight,
                             double monetaryWeight,
                             boolean normalize,
                             Comparator<T> tieBreaker) {
        if (plans == null || plans.isEmpty()) {
            return null;
        }
        final double maxMoney = sanitizeBound(budget);
        final double maxLatency = sanitizeBound(deadline);
        final Comparator<T> ties = tieBreaker == null ? (a, b) -> 0 : tieBreaker;

        // Evaluate each cost once; keep the input order so full ties stay deterministic.
        final Map<T, VectorCost> cost = new IdentityHashMap<>();
        final List<T> candidates = new ArrayList<>(plans.size());
        for (T plan : plans) {
            if (plan == null || cost.containsKey(plan)) continue;
            final VectorCost c = costs.apply(plan);
            cost.put(plan, c == null ? new VectorCost(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY) : c);
            candidates.add(plan);
        }
        if (candidates.isEmpty()) {
            return null;
        }

        final List<T> feasible = candidates.stream()
                .filter(p -> cost.get(p).getMonetary() <= maxMoney && cost.get(p).getLatency() <= maxLatency)
                .collect(Collectors.toList());

        if (feasible.isEmpty()) {
            final Comparator<T> byViolation = Comparator
                    .comparingDouble((T p) -> violation(cost.get(p), maxMoney, maxLatency))
                    .thenComparingDouble(p -> cost.get(p).getMonetary())
                    .thenComparingDouble(p -> cost.get(p).getLatency())
                    .thenComparing(ties);
            return candidates.stream().min(byViolation).orElse(null);
        }

        final double wL = sanitizeWeight(latencyWeight);
        final double wM = sanitizeWeight(monetaryWeight);
        final Comparator<T> latencyFirst = Comparator
                .comparingDouble((T p) -> cost.get(p).getLatency())
                .thenComparingDouble(p -> cost.get(p).getMonetary());
        final Comparator<T> moneyFirst = Comparator
                .comparingDouble((T p) -> cost.get(p).getMonetary())
                .thenComparingDouble(p -> cost.get(p).getLatency());

        final Comparator<T> order;
        if (wM == 0d) {
            order = latencyFirst.thenComparing(ties);
        } else if (wL == 0d) {
            order = moneyFirst.thenComparing(ties);
        } else {
            final double refL = normalize ? reference(feasible, p -> cost.get(p).getLatency()) : 1d;
            final double refM = normalize ? reference(feasible, p -> cost.get(p).getMonetary()) : 1d;
            order = Comparator
                    .comparingDouble((T p) -> wL * cost.get(p).getLatency() / refL
                            + wM * cost.get(p).getMonetary() / refM)
                    .thenComparing(latencyFirst)
                    .thenComparing(ties);
        }
        return feasible.stream().min(order).orElse(null);
    }

    /**
     * Minimise weighted cost among plans that respect the monetary bound; if none do, minimise weighted cost
     * over the whole set (SelectBest in Trummer and Koch).
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
        final double budgetCap = sanitizeBound(budget);
        List<T> feasible = plans.stream()
                .filter(plan -> money(costs, plan) <= budgetCap)
                .collect(Collectors.toList());
        if (feasible.isEmpty()) {
            feasible = new ArrayList<>(plans);
        }
        final double wL = sanitizeWeight(latencyWeight);
        final double wM = sanitizeWeight(monetaryWeight);
        final Comparator<T> order = Comparator
                .comparingDouble((T plan) -> weightedCost(costs.apply(plan), wL, wM))
                .thenComparing(tieBreaker == null ? (a, b) -> 0 : tieBreaker);
        return feasible.stream().min(order).orElse(null);
    }

    /** Relative amount by which a plan breaks the bounds; {@code 0} if it respects both. */
    static double violation(VectorCost cost, double maxMoney, double maxLatency) {
        double v = 0d;
        if (Double.isFinite(maxMoney)) {
            v = Math.max(v, Math.max(0d, cost.getMonetary() - maxMoney) / Math.max(maxMoney, TINY));
        }
        if (Double.isFinite(maxLatency)) {
            v = Math.max(v, Math.max(0d, cost.getLatency() - maxLatency) / Math.max(maxLatency, TINY));
        }
        return v;
    }

    private static <T> double reference(List<T> plans, Function<T, Double> axis) {
        final double best = plans.stream().mapToDouble(axis::apply).filter(Double::isFinite).min().orElse(1d);
        return best > TINY ? best : 1d;
    }

    private static <T> double money(Function<T, VectorCost> costs, T plan) {
        final VectorCost cost = costs.apply(plan);
        return cost == null ? Double.POSITIVE_INFINITY : cost.getMonetary();
    }

    /** {@code +∞} and NaN mean "no bound"; a negative bound is treated as {@code 0}. */
    private static double sanitizeBound(double bound) {
        if (Double.isNaN(bound) || bound == Double.POSITIVE_INFINITY) {
            return Double.POSITIVE_INFINITY;
        }
        if (!Double.isFinite(bound) || bound < 0d) {
            return 0d;
        }
        return bound;
    }

    private static double sanitizeWeight(double weight) {
        if (!Double.isFinite(weight) || weight < 0d) {
            return 0d;
        }
        return weight;
    }

    private static double weightedCost(VectorCost cost, double latencyWeight, double monetaryWeight) {
        if (cost == null) {
            return Double.POSITIVE_INFINITY;
        }
        if (latencyWeight == 0d && monetaryWeight == 0d) {
            return cost.getLatency();
        }
        return latencyWeight * cost.getLatency() + monetaryWeight * cost.getMonetary();
    }
}
