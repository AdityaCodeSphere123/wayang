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

package org.apache.wayang.core.optimizer.costs.learned;

import com.fasterxml.jackson.databind.JsonNode;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.wayang.core.api.Configuration;
import org.apache.wayang.core.optimizer.OptimizationContext;
import org.apache.wayang.core.optimizer.cardinality.CardinalityEstimate;
import org.apache.wayang.core.optimizer.costs.LoadProfile;
import org.apache.wayang.core.optimizer.costs.TimeEstimate;
import org.apache.wayang.core.plan.wayangplan.ExecutionOperator;

import java.util.HashMap;
import java.util.Map;

/**
 * Operator times from a two-layer model trained on the server data (Kaggle notebooks
 * {@code wayang-java-cost-model.ipynb} and {@code wayang-spark-combined-cost-model.ipynb}), for the platform named in
 * the exported file. Features, as in the notebooks' "wayang" cardinality mode:
 * <ul>
 *   <li>{@code log_rows}, {@code log_right_rows}, {@code log_out_rows_wayang}: Wayang's cardinality estimates (middle of
 *       the interval);</li>
 *   <li>{@code log_bytes_per_row}, {@code log_right_bytes}: Wayang does not track row widths, so the training mean bytes
 *       per row is used;</li>
 *   <li>Java: {@code heap_gb} of this JVM, {@code cpus} capped at 2 (training range; a Java operator runs on one
 *       thread);</li>
 *   <li>Spark (local mode): {@code log_slots} from {@code spark.master}, input partitions estimated from the input size
 *       (32 MB splits, at least 2; the right input of a join likewise), {@code log_input_parallelism},
 *       {@code remote_share} = 0, {@code heap_gb_per_slot}, {@code deploy} = "local";</li>
 *   <li>{@code operator}, {@code source} ("text"), {@code gc} of this JVM.</li>
 * </ul>
 * Predictions are for the training machine (Intel Xeon Silver 4314). They are multiplied by {@code machine_factor} from
 * the export (measured / predicted on this machine's profiler runs), or by
 * {@code wayang.core.optimizer.learned-cost-model.machine-factor.<platform>} if set. Operators that are not in the
 * training data (sources, collect sinks, conversions) keep Wayang's estimate.
 */
public final class TwoLayerTimeModel implements OperatorTimeModel {

    public static final String MACHINE_FACTOR_PROPERTY = "wayang.core.optimizer.learned-cost-model.machine-factor";
    private static final Logger logger = LogManager.getLogger(TwoLayerTimeModel.class);
    private static final int MAX_TRAINING_JAVA_CPUS = 2;
    private static final double SPLIT_BYTES = 32d * 1024 * 1024;

    private final TwoLayerNetwork net;
    private final String platform;
    private final double exportedMachineFactor;
    private final String gc;

    TwoLayerTimeModel(TwoLayerNetwork net, JsonNode doc) {
        this.net = net;
        this.platform = doc.path("platform").asText("java");
        this.exportedMachineFactor = doc.path("machine_factor").asDouble(1d);
        this.gc = LearnedOperatorTimeModel.gcKind();
        final double worst = referenceError(net, doc);
        logger.info("Two-layer {} model: max relative error vs Python on {} reference rows: {}; machine factor {}",
                this.platform, doc.path("reference").size(), worst, this.exportedMachineFactor);
        if (worst > 1e-3) throw new IllegalStateException("Two-layer model does not reproduce Python: " + worst);
    }

    /** Largest relative difference to the Python pipeline on the exported reference rows. */
    static double referenceError(TwoLayerNetwork net, JsonNode doc) {
        double worst = 0;
        for (JsonNode r : doc.path("reference")) {
            final Map<String, Double> num = new HashMap<>();
            final Map<String, String> cat = new HashMap<>();
            r.get("features").fields().forEachRemaining(e -> {
                if (e.getValue().isNumber()) num.put(e.getKey(), e.getValue().asDouble());
                else cat.put(e.getKey(), e.getValue().asText());
            });
            final double expected = r.get("expected_time_ns").asDouble();
            final double got = net.predictTimeNs(num, cat);
            worst = Math.max(worst, Math.abs(got - expected) / Math.max(expected, 1e3));
        }
        return worst;
    }

    public String getPlatform() {
        return this.platform;
    }

    @Override
    public TimeEstimate estimate(OptimizationContext.OperatorContext ctx, LoadProfile profile, TimeEstimate wayang,
                                 Configuration configuration) {
        final Map<String, Double> num = new HashMap<>();
        final Map<String, String> cat = new HashMap<>();
        if (!this.features(ctx, configuration, num, cat)) return null;
        final double factor = configuration.getDoubleProperty(MACHINE_FACTOR_PROPERTY + "." + this.platform,
                this.exportedMachineFactor);
        final double ns = this.net.predictTimeNs(num, cat) * factor * Math.max(1, ctx.getNumExecutions());
        final long ms = Math.max(1L, Math.round(ns / 1e6));
        if (logger.isDebugEnabled()) {
            logger.debug("Two-layer estimate for {}: {} ms (Wayang: {}), features {} {}", ctx.getOperator(), ms, wayang, num, cat);
        }
        return new TimeEstimate(ms, ms, wayang == null ? 1d : wayang.getCorrectnessProbability());
    }

    boolean features(OptimizationContext.OperatorContext ctx, Configuration configuration,
                     Map<String, Double> num, Map<String, String> cat) {
        if (!(ctx.getOperator() instanceof ExecutionOperator)) return false;
        final ExecutionOperator op = (ExecutionOperator) ctx.getOperator();
        if (!this.platform.equals(op.getPlatform().getConfigurationName())) return false;
        final String prefix = this.platform.equals("spark") ? "Spark" : "Java";
        final String kind = LearnedOperatorTimeModel.operatorKind(op.getClass().getSimpleName(), prefix);
        if (!this.net.knows("operator", kind)) return false;

        final boolean binary = op.getNumInputs() >= 2;
        final double nLeft = op.getNumInputs() > 0 ? mid(ctx.getInputCardinality(0)) : 0d;
        final double nRight = binary ? mid(ctx.getInputCardinality(1)) : 0d;
        final double nOut = op.getNumOutputs() > 0 ? mid(ctx.getOutputCardinality(0)) : 0d;
        final double logBytesPerRow = this.net.trainingMean("log_bytes_per_row");
        final double bytesPerRow = Math.pow(2, logBytesPerRow);
        final double heapGb = Math.round(Runtime.getRuntime().maxMemory() / (double) (1L << 30));
        final int machineCpus = Runtime.getRuntime().availableProcessors();

        num.put("log_rows", log2(Math.max(1, nLeft)));
        num.put("log_right_rows", log2(1 + nRight));
        num.put("log_bytes_per_row", logBytesPerRow);
        num.put("log_right_bytes", binary ? log2(1 + nRight * bytesPerRow) : 0d);
        num.put("log_out_rows_wayang", log2(1 + nOut));
        if (this.platform.equals("spark")) {
            final double slots = LearnedOperatorTimeModel.sparkSlots(configuration, machineCpus);
            final double inParts = Math.max(2d, Math.ceil(nLeft * bytesPerRow / SPLIT_BYTES));
            final double rightParts = binary ? Math.max(2d, Math.ceil(nRight * bytesPerRow / SPLIT_BYTES)) : 0d;
            num.put("log_slots", log2(Math.max(1, slots)));
            num.put("log_in_partitions", log2(inParts));
            num.put("log_right_in_partitions", log2(1 + rightParts));
            num.put("log_input_parallelism", log2(Math.max(1, Math.min(slots, inParts + rightParts))));
            num.put("remote_share", 0d);
            num.put("heap_gb_per_slot", heapGb / Math.max(1, slots));
            cat.put("deploy", "local");
        } else {
            num.put("heap_gb", heapGb);
            num.put("cpus", (double) Math.min(MAX_TRAINING_JAVA_CPUS, machineCpus));
        }
        cat.put("operator", kind);
        cat.put("source", "text");
        cat.put("gc", this.gc);
        return true;
    }

    private static double mid(CardinalityEstimate c) {
        return c == null ? 0d : (c.getLowerEstimate() + c.getUpperEstimate()) / 2d;
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2);
    }
}
