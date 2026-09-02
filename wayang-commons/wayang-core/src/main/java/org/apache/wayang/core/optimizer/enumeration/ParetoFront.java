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
        if (items.size() == 1) {
            return new ArrayList<>(items);
        }
        final Collection<T> representatives = epsilon > 0d
                ? bucketRepresentatives(items, costs, epsilon, tieBreaker)
                : items;
        return exactPareto(representatives, costs, tieBreaker);
    }

    /**
     * Groups items by logarithmic cost buckets. Used for traces; {@link #retain} keeps one plan per group.
     */
    public static <T> Map<List<Long>, List<T>> groupByBuckets(Collection<T> items,
                                                              Function<T, VectorCost> costs,
                                                              double epsilon) {
        Map<List<Long>, List<T>> grouped = new LinkedHashMap<>();
        for (T item : items) {
            final long[] buckets = costs.apply(item).logBuckets(epsilon);
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
