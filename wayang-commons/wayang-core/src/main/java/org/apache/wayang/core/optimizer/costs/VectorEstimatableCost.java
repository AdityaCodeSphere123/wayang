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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.wayang.core.api.Configuration;
import org.apache.wayang.core.api.exception.WayangException;
import org.apache.wayang.core.optimizer.OptimizationContext;
import org.apache.wayang.core.optimizer.ProbabilisticDoubleInterval;
import org.apache.wayang.core.optimizer.enumeration.LoopImplementation;
import org.apache.wayang.core.optimizer.enumeration.ParetoFront;
import org.apache.wayang.core.optimizer.enumeration.PlanImplementation;
import org.apache.wayang.core.optimizer.enumeration.VectorPlanSelection;
import org.apache.wayang.core.plan.executionplan.Channel;
import org.apache.wayang.core.plan.executionplan.ExecutionPlan;
import org.apache.wayang.core.plan.executionplan.ExecutionStage;
import org.apache.wayang.core.platform.Junction;
import org.apache.wayang.core.platform.Platform;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;

/**
 * Two-objective cost model: {@code (latency, monetary)}.
 *
 * <p>Both numbers are sums over execution operators, conversion operators (junctions) and loop bodies {@code e},
 * plus a start-up term per platform {@code p} the plan uses:</p>
 * <pre>
 *   latency  = Σ_e  s(t_e)                       + Σ_p init_p
 *   monetary = Σ_e  s(rate(p_e) · t_e)           + Σ_p s(fix_p + rate_p · init_p)
 * </pre>
 * <p>{@code t_e} is the time interval Wayang estimates, {@code rate}/{@code fix} come from
 * {@code wayang.<platform>.costs.per-ms}/{@code costs.fix}, and {@code s} turns an interval into one number
 * ({@code wayang.core.optimizer.objectives.estimate}):</p>
 * <ul>
 *   <li>{@code squash} (default): Wayang's configured cost squasher (geometric mean). The monetary axis is then
 *       <i>exactly</i> Wayang's scalar cost, so with no budget and weights {@code (0,1)} - or equal platform rates -
 *       the pick is the plan original Wayang picks.</li>
 *   <li>{@code mean}: the expected value (midpoint) of each interval.</li>
 * </ul>
 * <p>Either way {@code s} is applied per element and the results are summed, so the cost of a concatenated plan is
 * the sum of the costs of its parts. That additivity is what makes Pareto pruning of partial plans safe.
 * (Squashing the <i>total</i> interval, as v1 did, is not additive: {@code gm(a+b) != gm(a)+gm(b)}.)</p>
 */
public class VectorEstimatableCost extends DefaultEstimatableCost {

    private static final Logger logger = LogManager.getLogger(VectorEstimatableCost.class);

    public static final EstimatableCostFactory FACTORY = new Factory();

    /**
     * Each {@link PlanImplementation} gets its own instance through {@link #FACTORY}, so we can cache the
     * vector of the plan we were last asked about. The fingerprint guards against a plan that grew in between.
     */
    private PlanImplementation cachedPlan;
    private long cachedFingerprint;
    private VectorCost cachedWithOverhead, cachedWithoutOverhead;

    @Override
    public EstimatableCostFactory getFactory() {
        return FACTORY;
    }

    @Override
    public VectorCost getVectorEstimate(PlanImplementation plan, boolean isOverheadIncluded) {
        final long fingerprint = fingerprint(plan);
        if (this.cachedPlan != plan || this.cachedFingerprint != fingerprint) {
            this.computeVectorEstimates(plan);
            this.cachedPlan = plan;
            this.cachedFingerprint = fingerprint;
        }
        return isOverheadIncluded ? this.cachedWithOverhead : this.cachedWithoutOverhead;
    }

    private static long fingerprint(PlanImplementation plan) {
        return 31L * 31L * plan.getOperators().size()
                + 31L * plan.getJunctions().size()
                + plan.getLoopImplementations().size();
    }

    private void computeVectorEstimates(PlanImplementation plan) {
        final OptimizationContext optCtx = plan.getOptimizationContext();
        final Configuration configuration = optCtx.getConfiguration();
        final boolean isMean = "mean".equalsIgnoreCase(
                configuration.getStringProperty(ObjectiveSettings.ESTIMATE, "squash"));
        final ToDoubleFunction<ProbabilisticDoubleInterval> squasher = isMean
                ? ProbabilisticDoubleInterval::getAverageEstimate
                : configuration.getCostSquasherProvider().provide();
        double latency = 0d, monetary = 0d;

        // Execution operators. OperatorContext#getCostEstimate() is the time converted with the operator's
        // platform rate, without fix costs (fix costs are charged once per platform below).
        for (var operator : plan.getOperators()) {
            final OptimizationContext.OperatorContext opCtx = optCtx.getOperatorContext(operator);
            if (opCtx == null) continue;
            latency += time(opCtx, squasher);
            monetary += money(opCtx, isMean);
        }

        // Conversion operators between platforms/channels.
        for (OptimizationContext defaultCtx : optCtx.getDefaultOptimizationContexts()) {
            for (Junction junction : plan.getJunctions().values()) {
                for (OptimizationContext.OperatorContext opCtx : junction.getConversionOperatorContexts(defaultCtx)) {
                    latency += time(opCtx, squasher);
                    monetary += money(opCtx, isMean);
                }
            }
        }

        // Loops: each iteration body without overhead (overhead is charged once at top level), as Wayang does.
        for (LoopImplementation loop : plan.getLoopImplementations().values()) {
            for (LoopImplementation.IterationImplementation iteration : loop.getIterationImplementations()) {
                final VectorCost body = iteration.getBodyImplementation().getVectorCostEstimate(false);
                latency += body.getLatency();
                monetary += body.getMonetary();
            }
        }

        this.cachedWithoutOverhead = new VectorCost(latency, monetary);

        // Platform start-up: paid once per platform the plan touches.
        for (Platform platform : plan.getUtilizedPlatforms()) {
            final long startUp = configuration.getPlatformStartUpTimeProvider().provideFor(platform);
            final TimeToCostConverter converter = configuration.getTimeToCostConverterProvider().provideFor(platform);
            latency += startUp;
            monetary += squasher.applyAsDouble(converter.convert(new TimeEstimate(startUp, startUp, 1d)));
        }
        this.cachedWithOverhead = new VectorCost(latency, monetary);
    }

    private static double time(OptimizationContext.OperatorContext opCtx,
                               ToDoubleFunction<ProbabilisticDoubleInterval> squasher) {
        final TimeEstimate t = opCtx.getTimeEstimate();
        if (t == null) return 0d;
        return squasher.applyAsDouble(
                new ProbabilisticDoubleInterval(t.getLowerEstimate(), t.getUpperEstimate(), t.getCorrectnessProbability()));
    }

    private static double money(OptimizationContext.OperatorContext opCtx, boolean isMean) {
        if (!isMean) {
            // Same number Wayang's scalar cost model adds up for this operator.
            return opCtx.getSquashedCostEstimate();
        }
        final ProbabilisticDoubleInterval c = opCtx.getCostEstimate();
        return c == null ? 0d : c.getAverageEstimate();
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
        final ObjectiveSettings settings = ObjectiveSettings.of(any.getOptimizationContext().getConfiguration());

        // The final candidates are already pruned per group; across groups they can still dominate each other.
        final List<PlanImplementation> front = ParetoFront.retain(
                executionPlans, plan -> plan.getVectorCostEstimate(true), 0d, PlanImplementation.structuralComparator());

        final PlanImplementation picked = VectorPlanSelection.pick(
                front,
                plan -> plan.getVectorCostEstimate(true),
                settings.getBudget(),
                settings.getDeadline(),
                settings.getLatencyWeight(),
                settings.getMonetaryWeight(),
                settings.isNormalizeWeights(),
                PlanImplementation.structuralComparator()
        );
        if (picked == null) {
            throw new WayangException("Could not find an execution plan.");
        }

        if (logger.isInfoEnabled()) {
            logger.info("Pareto front of {} complete plans ({} candidates): {}",
                    front.size(), executionPlans.size(),
                    front.stream().map(p -> describe(p)).collect(Collectors.joining("; ")));
            logger.info("Picked {} (budget={}, deadline={}, weights latency={} monetary={}).",
                    describe(picked), settings.getBudget(), settings.getDeadline(),
                    settings.getLatencyWeight(), settings.getMonetaryWeight());
        }
        if (settings.getFrontFile() != null) {
            appendFront(settings.getFrontFile(), front, picked, executionPlans.size());
        }
        return picked;
    }

    private static String describe(PlanImplementation plan) {
        final VectorCost cost = plan.getVectorCostEstimate(true);
        final String platforms = plan.getUtilizedPlatforms().stream()
                .map(Platform::getName).sorted().collect(Collectors.joining("+"));
        return String.format("[%s] latency=%.2f money=%.4f", platforms, cost.getLatency(), cost.getMonetary());
    }

    /**
     * Appends {@code round,candidates,front_size,latency,monetary,platforms,picked} rows so experiments can
     * plot the front without parsing logs.
     */
    private static synchronized void appendFront(String file,
                                                 List<PlanImplementation> front,
                                                 PlanImplementation picked,
                                                 int numCandidates) {
        final Path path = Paths.get(file);
        try {
            final boolean isNew = !Files.exists(path);
            if (path.getParent() != null) Files.createDirectories(path.getParent());
            try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND))) {
                if (isNew) out.println("timestamp,candidates,front_size,latency_ms,monetary,platforms,picked");
                final long now = System.currentTimeMillis();
                for (PlanImplementation plan : front) {
                    final VectorCost cost = plan.getVectorCostEstimate(true);
                    final String platforms = plan.getUtilizedPlatforms().stream()
                            .map(Platform::getName).sorted().collect(Collectors.joining("+"));
                    out.printf("%d,%d,%d,%.6f,%.9f,%s,%s%n", now, numCandidates, front.size(),
                            cost.getLatency(), cost.getMonetary(), platforms, plan == picked);
                }
            }
        } catch (IOException e) {
            logger.warn("Could not write Pareto front to {}: {}", file, e.getMessage());
        }
    }

    public static class Factory implements EstimatableCostFactory {
        @Override
        public EstimatableCost makeCost() {
            return new VectorEstimatableCost();
        }
    }
}
