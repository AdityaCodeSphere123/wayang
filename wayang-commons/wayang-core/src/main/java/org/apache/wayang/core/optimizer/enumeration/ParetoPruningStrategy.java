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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.wayang.core.api.Configuration;
import org.apache.wayang.core.api.Job;
import org.apache.wayang.core.optimizer.costs.ObjectiveSettings;
import org.apache.wayang.core.plan.wayangplan.ExecutionOperator;
import org.apache.wayang.core.plan.wayangplan.PlanTraversal;
import org.apache.wayang.core.plan.wayangplan.WayangPlan;
import org.apache.wayang.core.platform.Platform;
import org.apache.wayang.core.util.Tuple;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Keeps an exact or {@code (1 + ε)}-approximate Pareto set of {@link PlanImplementation}s per group of plans that
 * share the same open interface (same platforms and same still-unconnected operators). Plan enumeration is unchanged.
 *
 * <p><b>Why grouping makes pruning safe.</b> Two partial plans in one group are completed by exactly the same set of
 * suffixes: the junctions they still need depend only on the interface operators, and the platform start-up
 * overhead only on the platform set. Since the cost vector is additive, if {@code p} dominates {@code q} then
 * {@code p + s} dominates {@code q + s} for every suffix {@code s}, so {@code q} can be dropped.</p>
 *
 * <p><b>Approximation.</b> An α-cover of an α-cover is an α²-cover, so applying ε at every prune compounds to
 * {@code (1 + ε)^k} after {@code k} prunes. By default ({@code epsilon.mode=global}) each prune therefore uses
 * {@code (1 + ε)^(1/k) − 1}, where {@code k} is the number of top-level operators of the job (an upper bound on the
 * prunes along one plan), so the final front is within {@code 1 + ε} of the exact one (Trummer and Koch, SIGMOD'14).</p>
 *
 * <p><b>Constraints.</b> Costs only grow when a plan is extended. A partial plan that already exceeds the budget or
 * the deadline can never become feasible, so it is dropped as long as its group still has a feasible plan
 * (if the group has none, the front is kept so the final pick can fall back to the least-violating plan).</p>
 */
public class ParetoPruningStrategy implements PlanEnumerationPruningStrategy {

    private static final Logger logger = LogManager.getLogger(ParetoPruningStrategy.class);

    private ObjectiveSettings settings;

    /** Per-step ε, computed once per {@link Job}. */
    private Job stepEpsilonJob;
    private double stepEpsilon = Double.NaN;

    private long numPrunes, numIn, numOut;

    @Override
    public void configure(Configuration configuration) {
        this.settings = ObjectiveSettings.of(configuration);
    }

    @Override
    public void prune(PlanEnumeration planEnumeration) {
        final Collection<PlanImplementation> current = planEnumeration.getPlanImplementations();
        if (current.size() < 2) {
            return;
        }
        final double epsilon = this.resolveStepEpsilon(current.iterator().next());

        final Collection<List<PlanImplementation>> groups = current.stream()
                .collect(Collectors.groupingBy(
                        ParetoPruningStrategy::getInterestingProperties,
                        LinkedHashMap::new,
                        Collectors.toList()))
                .values();

        final List<PlanImplementation> survivors = new ArrayList<>();
        for (List<PlanImplementation> group : groups) {
            survivors.addAll(this.pruneGroup(group, epsilon));
        }

        this.numPrunes++;
        this.numIn += current.size();
        this.numOut += survivors.size();
        if (logger.isDebugEnabled()) {
            logger.debug("Pareto prune #{} (ε_step={}): {} plans in {} groups -> {} plans.",
                    this.numPrunes, epsilon, current.size(), groups.size(), survivors.size());
        }

        current.clear();
        current.addAll(survivors);
    }

    List<PlanImplementation> pruneGroup(List<PlanImplementation> group, double epsilon) {
        List<PlanImplementation> candidates = group;
        if (this.settings.isPruneConstraints() && this.settings.hasConstraints()) {
            final List<PlanImplementation> feasible = group.stream()
                    .filter(plan -> this.settings.isFeasible(plan.getVectorCostEstimate(true)))
                    .collect(Collectors.toList());
            if (!feasible.isEmpty()) {
                candidates = feasible;
            }
        }
        final List<PlanImplementation> kept = ParetoFront.retain(
                candidates,
                plan -> plan.getVectorCostEstimate(true),
                epsilon,
                PlanImplementation.structuralComparator()
        );
        if (kept.isEmpty() && !candidates.isEmpty()) {
            return Collections.singletonList(candidates.get(0));
        }
        return kept;
    }

    private double resolveStepEpsilon(PlanImplementation sample) {
        final Job job = sample.getOptimizationContext() == null ? null : sample.getOptimizationContext().getJob();
        if (Double.isNaN(this.stepEpsilon) || job != this.stepEpsilonJob) {
            this.stepEpsilonJob = job;
            this.stepEpsilon = this.settings.stepEpsilon(countSteps(job));
            if (logger.isDebugEnabled()) {
                logger.debug("Pareto pruning: target ε={} ({}), per-step ε={}.",
                        this.settings.getEpsilon(), this.settings.getEpsilonMode(), this.stepEpsilon);
            }
        }
        return this.stepEpsilon;
    }

    /** Upper bound on the prunes along one plan: the number of top-level operators of the job. */
    static int countSteps(Job job) {
        if (job == null) return 1;
        try {
            final WayangPlan plan = job.getWayangPlan();
            if (plan == null) return 1;
            return Math.max(1, PlanTraversal.upstream().traverse(plan.getSinks()).getTraversedNodes().size());
        } catch (RuntimeException e) {
            return 1;
        }
    }

    private static Tuple<Set<Platform>, Set<ExecutionOperator>> getInterestingProperties(PlanImplementation implementation) {
        return new Tuple<>(
                implementation.getUtilizedPlatforms(),
                new HashSet<>(implementation.getInterfaceOperators())
        );
    }

    /** For experiments: average number of plans kept per prune. */
    public String getStatistics() {
        return String.format("prunes=%d in=%d out=%d", this.numPrunes, this.numIn, this.numOut);
    }
}
