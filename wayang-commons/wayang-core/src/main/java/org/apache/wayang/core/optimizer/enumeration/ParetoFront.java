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
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Exact and {@code (1 + ε)}-approximate Pareto filtering for two-objective plan costs.
 * The approximate variant is the logarithmic-bucketing scheme from Trummer and Koch (SIGMOD 2014).
 */
public final class ParetoFront {

    private ParetoFront() {
    }

    public static <T> List<T> retain(Collection<T> items, Function<T, VectorCost> costs, double epsilon) {
        return retain(items, costs, epsilon, null);
    }

    public static <T> List<T> retain(Collection<T> items,
                                     Function<T, VectorCost> costs,
                                     double epsilon,
                                     Comparator<T> tieBreaker) {
        if (items == null || items.isEmpty()) {
            return new ArrayList<>();
        }
        // Evaluate every cost exactly once: sorting and bucketing look costs up many times.
        final Map<T, VectorCost> cache = new IdentityHashMap<>(items.size() * 2);
        List<T> cleaned = new ArrayList<>(items.size());
        for (T item : items) {
            if (item == null || cache.containsKey(item)) {
                continue;
            }
            final VectorCost cost = costs.apply(item);
            if (cost == null) {
                continue;
            }
            cache.put(item, cost);
            cleaned.add(item);
        }
        costs = cache::get;
        if (cleaned.size() <= 1) {
            return cleaned;
        }
        final double eps = VectorCost.finiteEpsilon(epsilon);
        Collection<T> representatives = cleaned;
        if (eps > 0d) {
            // One plan per log-bucket, plus the exact fastest and the exact cheapest plan. Keeping the two extremes
            // costs at most two plans, and it makes "fastest" and "cheapest" exact for any ε: with additive costs the
            // fastest (cheapest) complete plan always extends the fastest (cheapest) prefix of its group.
            final java.util.LinkedHashSet<T> kept = new java.util.LinkedHashSet<>(
                    bucketRepresentatives(cleaned, costs, eps, tieBreaker));
            kept.add(extreme(cleaned, costs, true, tieBreaker));
            kept.add(extreme(cleaned, costs, false, tieBreaker));
            representatives = kept;
        }
        return exactPareto(representatives, costs, tieBreaker);
    }

    /**
     * Groups items by logarithmic cost buckets. Used for traces; {@link #retain} keeps one plan per group.
     */
    public static <T> Map<List<Long>, List<T>> groupByBuckets(Collection<T> items,
                                                              Function<T, VectorCost> costs,
                                                              double epsilon) {
        Map<List<Long>, List<T>> grouped = new LinkedHashMap<>();
        if (items == null) {
            return grouped;
        }
        for (T item : items) {
            if (item == null) {
                continue;
            }
            final VectorCost cost = costs.apply(item);
            if (cost == null) {
                continue;
            }
            final long[] buckets = cost.logBuckets(epsilon);
            grouped.computeIfAbsent(Arrays.asList(buckets[0], buckets[1]), key -> new ArrayList<>()).add(item);
        }
        return grouped;
    }

    private static <T> Collection<T> bucketRepresentatives(Collection<T> items,
                                                           Function<T, VectorCost> costs,
                                                           double epsilon,
                                                           Comparator<T> tieBreaker) {
        Map<List<Long>, T> byBucket = new LinkedHashMap<>();
        for (T item : items) {
            final long[] buckets = costs.apply(item).logBuckets(epsilon);
            final List<Long> key = Arrays.asList(buckets[0], buckets[1]);
            byBucket.merge(key, item, (existing, candidate) -> prefer(existing, candidate, costs, tieBreaker));
        }
        return byBucket.values();
    }

    private static <T> List<T> exactPareto(Collection<T> items,
                                           Function<T, VectorCost> costs,
                                           Comparator<T> tieBreaker) {
        List<T> ordered = new ArrayList<>(items);
        ordered.sort((a, b) -> {
            final VectorCost ca = costs.apply(a);
            final VectorCost cb = costs.apply(b);
            int cmp = Double.compare(ca.getLatency(), cb.getLatency());
            if (cmp != 0) {
                return cmp;
            }
            cmp = Double.compare(ca.getMonetary(), cb.getMonetary());
            if (cmp != 0) {
                return cmp;
            }
            return tieBreaker == null ? 0 : tieBreaker.compare(a, b);
        });

        List<T> front = new ArrayList<>();
        double bestMoney = Double.POSITIVE_INFINITY;
        for (T item : ordered) {
            final double money = costs.apply(item).getMonetary();
            if (money < bestMoney) {
                front.add(item);
                bestMoney = money;
            }
        }
        return front;
    }

    /** Lexicographic minimum: latency then money ({@code byLatency}), or money then latency. */
    private static <T> T extreme(Collection<T> items, Function<T, VectorCost> costs, boolean byLatency,
                                 Comparator<T> tieBreaker) {
        Comparator<T> order = byLatency
                ? Comparator.<T>comparingDouble(t -> costs.apply(t).getLatency())
                .thenComparingDouble(t -> costs.apply(t).getMonetary())
                : Comparator.<T>comparingDouble(t -> costs.apply(t).getMonetary())
                .thenComparingDouble(t -> costs.apply(t).getLatency());
        if (tieBreaker != null) order = order.thenComparing(tieBreaker);
        return items.stream().min(order).orElseThrow(IllegalStateException::new);
    }

    private static <T> T prefer(T existing, T candidate, Function<T, VectorCost> costs, Comparator<T> tieBreaker) {
        final VectorCost ce = costs.apply(existing);
        final VectorCost cc = costs.apply(candidate);
        int cmp = Double.compare(ce.getLatency(), cc.getLatency());
        if (cmp != 0) {
            return cmp < 0 ? existing : candidate;
        }
        cmp = Double.compare(ce.getMonetary(), cc.getMonetary());
        if (cmp != 0) {
            return cmp < 0 ? existing : candidate;
        }
        if (tieBreaker == null) {
            return existing;
        }
        return tieBreaker.compare(existing, candidate) <= 0 ? existing : candidate;
    }
}
