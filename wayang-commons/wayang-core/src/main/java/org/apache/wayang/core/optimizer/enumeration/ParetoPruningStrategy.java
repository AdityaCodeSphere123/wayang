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

import org.apache.wayang.core.api.Configuration;
import org.apache.wayang.core.plan.wayangplan.ExecutionOperator;
import org.apache.wayang.core.platform.Platform;
import org.apache.wayang.core.util.Tuple;

import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Keeps an exact or {@code (1 + ε)}-approximate Pareto set of {@link PlanImplementation}s that share
 * the same open interface (same platforms and still-unconnected operators). Enumeration itself is unchanged.
 */
public class ParetoPruningStrategy implements PlanEnumerationPruningStrategy {

    private double epsilon;

    @Override
    public void configure(Configuration configuration) {
        this.epsilon = configuration.getDoubleProperty("wayang.core.optimizer.objectives.epsilon", 0.1d);
        if (!Double.isFinite(this.epsilon) || this.epsilon < 0d) {
            this.epsilon = 0d;
        }
    }

    @Override
    public void prune(PlanEnumeration planEnumeration) {
        if (planEnumeration.getPlanImplementations().size() < 2) {
            return;
        }
        final Collection<List<PlanImplementation>> competingPlans =
                planEnumeration.getPlanImplementations().stream()
                        .collect(Collectors.groupingBy(
                                ParetoPruningStrategy::getInterestingProperties,
                                LinkedHashMap::new,
                                Collectors.toList()))
                        .values();
        final List<PlanImplementation> survivors = competingPlans.stream()
                .flatMap(group -> ParetoFront.retain(
                        group,
                        plan -> plan.getVectorCostEstimate(true),
                        this.epsilon,
                        PlanImplementation.structuralComparator()
                ).stream())
                .collect(Collectors.toList());
        final Collection<PlanImplementation> current = planEnumeration.getPlanImplementations();
        current.clear();
        current.addAll(survivors);
    }

    private static Tuple<Set<Platform>, Set<ExecutionOperator>> getInterestingProperties(PlanImplementation implementation) {
        return new Tuple<>(
                implementation.getUtilizedPlatforms(),
                new HashSet<>(implementation.getInterfaceOperators())
        );
    }
}
