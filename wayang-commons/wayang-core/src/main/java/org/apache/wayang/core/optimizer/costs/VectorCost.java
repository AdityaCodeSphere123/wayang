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

import java.util.Objects;

/**
 * Two-objective cost vector {@code (latency, monetary)} used by the multi-objective optimizer.
 * Both components are non-negative; latency is in milliseconds.
 */
public final class VectorCost {

    private final double latency;
    private final double monetary;

    public VectorCost(double latency, double monetary) {
        this.latency = sanitize(latency);
        this.monetary = sanitize(monetary);
    }

    private static double sanitize(double value) {
        if (!Double.isFinite(value) || value < 0d) {
            return 0d;
        }
        return value;
    }

    public double getLatency() {
        return this.latency;
    }

    public double getMonetary() {
        return this.monetary;
    }

    /**
     * Strict Pareto dominance: weakly better in every objective and strictly better in at least one.
     */
    public boolean dominates(VectorCost that) {
        if (that == null) {
            return false;
        }
        return this.latency <= that.latency
                && this.monetary <= that.monetary
                && (this.latency < that.latency || this.monetary < that.monetary);
    }

    /**
     * {@code α}-dominance: {@code this} is at most a factor {@code alpha} worse than {@code that} in every objective.
     */
    public boolean approximatelyDominates(VectorCost that, double alpha) {
        if (alpha <= 1d) {
            return this.dominates(that) || this.equals(that);
        }
        return this.latency <= alpha * that.latency && this.monetary <= alpha * that.monetary;
    }

    /**
     * Logarithmically coarsens this vector so that values in a ratio-{@code (1 + epsilon)} interval share a bucket.
     */
    public VectorCost coarsen(double epsilon) {
        if (epsilon <= 0d) {
            return this;
        }
        final double base = 1d + epsilon;
        return new VectorCost(coarsenCoordinate(this.latency, base), coarsenCoordinate(this.monetary, base));
    }

    public long[] logBuckets(double epsilon) {
        if (epsilon <= 0d) {
            return new long[]{
                    Double.doubleToLongBits(this.latency),
                    Double.doubleToLongBits(this.monetary)
            };
        }
        final double logBase = Math.log(1d + epsilon);
        return new long[]{bucket(this.latency, logBase), bucket(this.monetary, logBase)};
    }

    private static double coarsenCoordinate(double value, double base) {
        if (value <= 0d) {
            return 0d;
        }
        return Math.pow(base, Math.floor(Math.log(value) / Math.log(base)));
    }

    private static long bucket(double value, double logBase) {
        if (value <= 0d) {
            return Long.MIN_VALUE;
        }
        return (long) Math.floor(Math.log(value) / logBase);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || this.getClass() != o.getClass()) return false;
        VectorCost that = (VectorCost) o;
        return Double.compare(that.latency, this.latency) == 0
                && Double.compare(that.monetary, this.monetary) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(this.latency, this.monetary);
    }

    @Override
    public String toString() {
        return String.format("VectorCost(latency=%.4f, monetary=%.4f)", this.latency, this.monetary);
    }
}
