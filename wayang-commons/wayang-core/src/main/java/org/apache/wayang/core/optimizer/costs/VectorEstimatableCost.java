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
import org.apache.wayang.core.api.exception.WayangException;
import org.apache.wayang.core.optimizer.enumeration.PlanImplementation;
import org.apache.wayang.core.optimizer.enumeration.VectorPlanSelection;
import org.apache.wayang.core.plan.executionplan.Channel;
import org.apache.wayang.core.plan.executionplan.ExecutionPlan;
import org.apache.wayang.core.plan.executionplan.ExecutionStage;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.Collection;
import java.util.Set;

/**
 * Vector-valued cost: latency from Wayang time estimates, monetary from the existing
 * {@link TimeToCostConverter} path ({@code wayang.<platform>.costs.per-ms} / {@code costs.fix}).
 */
public class VectorEstimatableCost extends DefaultEstimatableCost {

    private static final Logger logger = LogManager.getLogger(VectorEstimatableCost.class);

    public static final EstimatableCostFactory FACTORY = new Factory();

    @Override
    public EstimatableCostFactory getFactory() {
        return FACTORY;
    }

    @Override
    public VectorCost getVectorEstimate(PlanImplementation plan, boolean isOverheadIncluded) {
        final TimeEstimate time = plan.getTimeEstimate(isOverheadIncluded);
        // Geometric mean is 0 whenever the lower bound is 0, which is common for time intervals.
        final double geometric = time.getGeometricMeanEstimate();
        final double latency = geometric > 0d ? geometric : time.getAverageEstimate();
        final double monetary = this.getSquashedEstimate(plan, isOverheadIncluded);
        return new VectorCost(latency, monetary);
    }

    @Override
    public PlanImplementation pickBestExecutionPlan(Collection<PlanImplementation> executionPlans,
                                                    ExecutionPlan existingPlan,
                                                    Set<Channel> openChannels,
                                                    Set<ExecutionStage> executedStages) {
        if (executionPlans.isEmpty()) {
            throw new WayangException("Could not find an execution plan.");
        }
        final PlanImplementation any = executionPlans.iterator().next();
        final Configuration configuration = any.getOptimizationContext().getConfiguration();
        final double budget = configuration.getDoubleProperty("wayang.core.optimizer.objectives.budget", Double.POSITIVE_INFINITY);
        final double latencyWeight = configuration.getDoubleProperty("wayang.core.optimizer.objectives.weight.latency", 1d);
        final double monetaryWeight = configuration.getDoubleProperty("wayang.core.optimizer.objectives.weight.monetary", 0d);

        final PlanImplementation picked = VectorPlanSelection.pick(
                executionPlans,
                plan -> plan.getVectorCostEstimate(true),
                budget,
                latencyWeight,
                monetaryWeight,
                PlanImplementation.structuralComparator()
        );
        if (picked == null) {
            throw new WayangException("Could not find an execution plan.");
        }
        logger.info("Picked plan with {} under budget {} (weights latency={}, monetary={}).",
                picked.getVectorCostEstimate(true), budget, latencyWeight, monetaryWeight);
        return picked;
    }

    public static class Factory implements EstimatableCostFactory {
        @Override
        public EstimatableCost makeCost() {
            return new VectorEstimatableCost();
        }
    }
}
