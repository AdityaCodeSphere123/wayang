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

import org.apache.wayang.core.api.Configuration;

/**
 * All multi-objective knobs in one place, read from {@code wayang.core.optimizer.objectives.*}.
 * The pruning strategy, the cost model and the plan enumeration read the same keys through this class.
 */
public final class ObjectiveSettings {

    public static final String PREFIX = "wayang.core.optimizer.objectives.";

    /** Target approximation factor {@code α = 1 + ε} for the whole plan. {@code 0} = exact Pareto. */
    public static final String EPSILON = PREFIX + "epsilon";

    /** {@code global} (default): split ε over all pruning steps; {@code per-step}: apply ε at every prune. */
    public static final String EPSILON_MODE = PREFIX + "epsilon.mode";

    /** Optional override for the number of pruning steps used by {@code epsilon.mode=global}. */
    public static final String EPSILON_STEPS = PREFIX + "epsilon.steps";

    /** Upper bound on monetary cost (same unit as {@code wayang.<platform>.costs.*}). */
    public static final String BUDGET = PREFIX + "budget";

    /** Upper bound on latency in milliseconds. */
    public static final String DEADLINE = PREFIX + "deadline";

    public static final String WEIGHT_LATENCY = PREFIX + "weight.latency";

    public static final String WEIGHT_MONETARY = PREFIX + "weight.monetary";

    /** Divide each objective by its best value on the final front before weighting (unit-free weights). */
    public static final String WEIGHT_NORMALIZE = PREFIX + "weight.normalize";

    /** Drop partial plans that already break the budget or the deadline. */
    public static final String PRUNE_CONSTRAINTS = PREFIX + "prune.constraints";

    /** {@code single}: one min-cost conversion per boundary; {@code dual}: also the min-latency conversion. */
    public static final String JUNCTIONS = PREFIX + "junctions";

    /** Optional CSV file the final Pareto front of every optimization round is appended to. */
    public static final String FRONT_FILE = PREFIX + "front.file";

    /**
     * How an interval estimate becomes one number, per operator/conversion/start-up, before summing.
     * {@code squash} (default): Wayang's configured cost squasher ({@code wayang.core.optimizer.cost.squash},
     * geometric mean by default) - the money axis is then exactly Wayang's scalar cost.
     * {@code mean}: the expected value (midpoint) of each interval.
     */
    public static final String ESTIMATE = PREFIX + "estimate";

    /**
     * Upper bound on the plans kept per group (0 = unbounded). A group whose front is larger is coarsened until it
     * fits; the fastest and the cheapest plan are always kept. Bounds the work on long plans.
     */
    public static final String MAX_FRONT = PREFIX + "max-front";

    public enum EpsilonMode {GLOBAL, PER_STEP}

    public enum Estimate {SQUASH, MEAN}

    private final double epsilon;
    private final EpsilonMode epsilonMode;
    private final int epsilonSteps;
    private final double budget;
    private final double deadline;
    private final double latencyWeight;
    private final double monetaryWeight;
    private final boolean normalizeWeights;
    private final boolean pruneConstraints;
    private final boolean dualJunctions;
    private final String frontFile;
    private final Estimate estimate;
    private final int maxFront;

    private ObjectiveSettings(Configuration configuration) {
        this.epsilon = VectorCost.finiteEpsilon(configuration.getDoubleProperty(EPSILON, 0d));
        this.maxFront = (int) Math.max(0L, configuration.getLongProperty(MAX_FRONT, 64L));
        this.epsilonMode = "per-step".equalsIgnoreCase(configuration.getStringProperty(EPSILON_MODE, "global"))
                ? EpsilonMode.PER_STEP
                : EpsilonMode.GLOBAL;
        this.epsilonSteps = (int) configuration.getLongProperty(EPSILON_STEPS, 0L);
        this.budget = bound(configuration.getDoubleProperty(BUDGET, Double.POSITIVE_INFINITY));
        this.deadline = bound(configuration.getDoubleProperty(DEADLINE, Double.POSITIVE_INFINITY));
        this.latencyWeight = configuration.getDoubleProperty(WEIGHT_LATENCY, 1d);
        this.monetaryWeight = configuration.getDoubleProperty(WEIGHT_MONETARY, 0d);
        this.normalizeWeights = configuration.getBooleanProperty(WEIGHT_NORMALIZE, true);
        this.pruneConstraints = configuration.getBooleanProperty(PRUNE_CONSTRAINTS, true);
        this.dualJunctions = "dual".equalsIgnoreCase(configuration.getStringProperty(JUNCTIONS, "single"));
        final String file = configuration.getStringProperty(FRONT_FILE, "");
        this.frontFile = file == null || file.trim().isEmpty() ? null : file.trim();
        this.estimate = "mean".equalsIgnoreCase(configuration.getStringProperty(ESTIMATE, "squash"))
                ? Estimate.MEAN
                : Estimate.SQUASH;
    }

    public static ObjectiveSettings of(Configuration configuration) {
        return new ObjectiveSettings(configuration);
    }

    /** NaN, negative or missing bounds mean "no bound". */
    private static double bound(double value) {
        return Double.isNaN(value) || value < 0d ? Double.POSITIVE_INFINITY : value;
    }

    /**
     * The ε to use in one pruning step. With {@link EpsilonMode#GLOBAL}, the overall factor
     * {@code 1 + ε} is split over {@code steps} prunes so that {@code (1 + ε_step)^steps = 1 + ε}.
     * Pruning errors multiply along a plan (an α-cover of an α-cover is an α²-cover), so a fixed
     * per-step ε would give a factor of {@code (1 + ε)^steps} at the end.
     *
     * @param steps upper bound on the number of prunes along one plan (ignored if {@code epsilon.steps} is set)
     */
    public double stepEpsilon(int steps) {
        if (this.epsilon == 0d || this.epsilonMode == EpsilonMode.PER_STEP) {
            return this.epsilon;
        }
        final int n = Math.max(1, this.epsilonSteps > 0 ? this.epsilonSteps : steps);
        return VectorCost.finiteEpsilon(Math.expm1(Math.log1p(this.epsilon) / n));
    }

    public boolean hasConstraints() {
        return Double.isFinite(this.budget) || Double.isFinite(this.deadline);
    }

    public boolean isFeasible(VectorCost cost) {
        return cost != null && cost.getMonetary() <= this.budget && cost.getLatency() <= this.deadline;
    }

    public double getEpsilon() {
        return this.epsilon;
    }

    public EpsilonMode getEpsilonMode() {
        return this.epsilonMode;
    }

    public double getBudget() {
        return this.budget;
    }

    public double getDeadline() {
        return this.deadline;
    }

    public double getLatencyWeight() {
        return this.latencyWeight;
    }

    public double getMonetaryWeight() {
        return this.monetaryWeight;
    }

    public boolean isNormalizeWeights() {
        return this.normalizeWeights;
    }

    public boolean isPruneConstraints() {
        return this.pruneConstraints;
    }

    public boolean isDualJunctions() {
        return this.dualJunctions;
    }

    public int getMaxFront() {
        return this.maxFront;
    }

    public Estimate getEstimate() {
        return this.estimate;
    }

    public String getFrontFile() {
        return this.frontFile;
    }
}
