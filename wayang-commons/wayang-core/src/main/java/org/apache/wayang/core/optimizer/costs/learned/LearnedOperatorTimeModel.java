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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.wayang.core.api.Configuration;
import org.apache.wayang.core.optimizer.OptimizationContext;
import org.apache.wayang.core.optimizer.cardinality.CardinalityEstimate;
import org.apache.wayang.core.optimizer.costs.LoadEstimate;
import org.apache.wayang.core.optimizer.costs.LoadProfile;
import org.apache.wayang.core.optimizer.costs.TimeEstimate;
import org.apache.wayang.core.plan.wayangplan.ExecutionOperator;
import org.apache.wayang.core.util.fs.FileSystems;

import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Replaces Wayang's per-operator time estimate with the prediction of the trained cost model
 * ({@code Cost-Model/model}). Enabled with
 * <pre>wayang.core.optimizer.learned-cost-model = /path/to/model.json</pre>
 * Everything downstream (plan cost, pruning, plan choice) then uses the learned times; the scalar optimizer and
 * the Pareto optimizer both work unchanged.
 *
 * <p>Inputs are built as in {@code features.py} / the profilers:</p>
 * <ul>
 *   <li>categories: operator kind and implementation (class name), platform, GC of this JVM, source "text";
 *       a text-file source is the model's "scan" (source &rarr; parse &rarr; sink in the profilers);</li>
 *   <li>config: cores (Java: machine CPUs; Spark: task slots of {@code spark.master}), heap of this JVM, Spark shuffle
 *       partitions (2 &times; slots, as Wayang sets them), input partitions;</li>
 *   <li>context: words per row, bytes per row, filter selectivity (from Wayang's cardinalities), binary;</li>
 *   <li>row counts (basis): Wayang's estimated input/output cardinalities (middle of the interval), keys;</li>
 *   <li>Wayang's own estimate per resource: this operator's load profile (incl. UDF sub-profiles), with CPU cycles
 *       converted at the clock the model was trained with (2700 MHz), independent of {@code wayang.*.cpu.mhz}.</li>
 * </ul>
 * Operators the model was not trained on (channel conversions, collection sources, callback sinks, ...) keep Wayang's
 * estimate, as in {@code Cost-Model/job-eval}.
 */
public final class LearnedOperatorTimeModel implements OperatorTimeModel {

    /** Clip normalized config/context features to this many standard deviations of the training data. */
    public static final String Z_CLIP_PROPERTY = "wayang.core.optimizer.learned-cost-model.z-clip";

    private static final Logger logger = LogManager.getLogger(LearnedOperatorTimeModel.class);
    private static final Pattern LOCAL_MASTER = Pattern.compile("local\\[(\\*|\\d+)(?:,\\d+)?\\]");

    /** Clock the training data's Wayang CPU estimates were converted with (wayang_cpu_mhz in pairs.csv). */
    private static final double TRAINING_WAYANG_MHZ = 2700d;
    /** Training defaults for context features Wayang cannot know (median of the profiler data). */
    private static final double DEFAULT_WORDS = 8, DEFAULT_BYTES_PER_ROW = 79, DEFAULT_KEYS = 1000;
    /** Hadoop's local file system splits text files into 32 MB blocks; Spark reads at least 2 partitions. */
    private static final double SPLIT_BYTES = 32d * 1024 * 1024;

    private final LearnedCostNetwork net;
    private final String gc;
    private final AtomicLong predicted = new AtomicLong(), fallback = new AtomicLong();

    private LearnedOperatorTimeModel(LearnedCostNetwork net) {
        this.net = net;
        this.gc = gcKind();
    }

    /** Loads an exported model.json (Cost-Model/model/export_weights.py) and checks it against PyTorch. */
    static LearnedOperatorTimeModel load(String path) throws java.io.IOException {
        final LearnedCostNetwork net = LearnedCostNetwork.load(path);
        final double err = net.maxReferenceError();
        logger.info("Loaded learned cost model {} (trained from {}); max relative error vs PyTorch on {} "
                + "reference rows: {}", path, net.getSource(), net.numReferenceRows(), err);
        if (err > 1e-3) {
            throw new IllegalStateException("Learned cost model does not reproduce PyTorch: " + err);
        }
        return new LearnedOperatorTimeModel(net);
    }

    /**
     * The learned time estimate for {@code ctx}, or {@code null} to keep Wayang's.
     *
     * @param wayangEstimate Wayang's estimate (only its probability is reused)
     */
    @Override
    public TimeEstimate estimate(OptimizationContext.OperatorContext ctx, LoadProfile profile,
                                 TimeEstimate wayangEstimate, Configuration configuration) {
        final Features f = this.features(ctx, profile, configuration);
        if (f == null) {
            this.fallback.incrementAndGet();
            return null;
        }
        final double zClip = configuration.getDoubleProperty(Z_CLIP_PROPERTY, 3d);
        final double[] out = this.net.forward(f.cat, f.cfg, f.ctx, f.basis, f.wayang, f.aux, zClip);
        final double seconds = out[4] * Math.max(1, ctx.getNumExecutions());
        final long ms = Math.max(1L, Math.round(seconds * 1000d));
        this.predicted.incrementAndGet();
        if (logger.isDebugEnabled()) {
            logger.debug("Learned estimate for {}: {} ms (Wayang: {}); in={} out={} keys={} cpu={}cs",
                    ctx.getOperator(), ms, wayangEstimate, f.basis[1] * 1e6, f.basis[3] * 1e6, f.basis[5] * 1e6,
                    f.wayang[0]);
        }
        return new TimeEstimate(ms, ms, wayangEstimate == null ? 1d : wayangEstimate.getCorrectnessProbability());
    }

    public long getNumPredicted() {
        return this.predicted.get();
    }

    public long getNumFallbacks() {
        return this.fallback.get();
    }

    // ------------------------------------------------------------------ features
    static final class Features {
        int[] cat;
        double[] cfg, ctx, basis, wayang, aux;
    }

    Features features(OptimizationContext.OperatorContext ctx, LoadProfile profile, Configuration configuration) {
        if (!(ctx.getOperator() instanceof ExecutionOperator) || profile == null) return null;
        final ExecutionOperator op = (ExecutionOperator) ctx.getOperator();
        final String platform = op.getPlatform().getConfigurationName().toLowerCase();
        if (!this.net.knows("platform", platform)) return null;

        // Implementation and operator kind, named as in the profilers.
        final String cls = op.getClass().getSimpleName();
        final String prefix = platform.equals("spark") ? "Spark" : platform.substring(0, 1).toUpperCase() + platform.substring(1);
        final boolean scan = cls.endsWith("TextFileSource");
        final String impl = scan ? prefix + "Scan" : cls;
        if (!this.net.knows("impl", impl)) return null;
        final String kind = scan ? "scan" : operatorKind(cls, prefix);
        if (!this.net.knows("operator", kind)) return null;

        // Row counts: Wayang's estimates (middle of the interval).
        final boolean binary = op.getNumInputs() >= 2;
        final double nOut = op.getNumOutputs() > 0 ? mid(ctx.getOutputCardinality(0)) : 0d;
        double nLeft = op.getNumInputs() > 0 ? mid(ctx.getInputCardinality(0)) : nOut;
        final double nRight = binary ? mid(ctx.getInputCardinality(1)) : 0d;
        if (scan) nLeft = nOut;                                             // profilers: scan rows = rows read
        double keys;
        switch (kind) {
            case "reduceby": case "groupby": case "distinct": keys = nOut; break;
            case "join": keys = Math.min(nLeft, nRight); break;
            default: keys = DEFAULT_KEYS;
        }
        keys = Math.min(Math.max(1d, keys), Math.max(1d, nLeft + nRight));

        // Platform configuration.
        final int machineCpus = Runtime.getRuntime().availableProcessors();
        final double heapBytes = Runtime.getRuntime().maxMemory();
        final boolean spark = platform.equals("spark");
        final double slots = spark ? sparkSlots(configuration, machineCpus) : 1d;
        final double cores = spark ? slots : machineCpus;                    // features.py: task_slots, else cpus
        final double shuffle = spark ? 2 * slots : 0d;                      // Wayang: 2 x cores shuffle partitions
        double bytesPerRow = DEFAULT_BYTES_PER_ROW, inPartitions = spark ? 2d : 1d;
        if (scan) {
            final OptionalLong size = fileSize(op);
            if (size.isPresent() && nOut > 0) {
                bytesPerRow = size.getAsLong() / nOut;
                if (spark) inPartitions = Math.max(2d, Math.ceil(size.getAsLong() / SPLIT_BYTES));
            }
        }
        if (spark && binary) inPartitions *= 2;                              // left + right input partitions

        final double sel = kind.equals("filter") && nLeft > 0 ? Math.min(1d, Math.max(0d, nOut / nLeft)) : 1d;

        final Features f = new Features();
        f.cat = new int[this.net.getCategories().size()];
        for (int i = 0; i < f.cat.length; i++) {
            final String c = this.net.getCategories().get(i);
            final String v;
            switch (c) {
                case "operator": v = kind; break;
                case "impl": v = impl; break;
                case "platform": v = platform; break;
                case "gc": v = this.gc; break;
                case "source": v = "text"; break;
                default: v = null;
            }
            f.cat[i] = this.net.index(c, v);
        }
        f.cfg = new double[]{log2(Math.max(1, cores)), log2(Math.max(1e-3, heapBytes / 1e9)), log2(1 + shuffle),
                log2(1 + inPartitions)};
        f.ctx = new double[]{log2(DEFAULT_WORDS), log2(Math.max(1, bytesPerRow)), sel, binary ? 1 : 0};
        final double u = 1e-6, nIn = nLeft + nRight;
        f.basis = new double[]{1, nLeft * u, nRight * u, nOut * u, nIn * u * log2(1 + nIn) / 20, keys * u};
        f.wayang = new double[]{
                midOf(profile, LoadProfile::getCpuUsage) / (TRAINING_WAYANG_MHZ * 1e6),
                midOf(profile, LoadProfile::getDiskUsage) / 1e9,
                midOf(profile, LoadProfile::getNetworkUsage) / 1e9,
                midOf(profile, LoadProfile::getRamUsage) / 1e9};
        f.aux = new double[]{slots, heapBytes / 1e9, 1d};
        return f;
    }

    /** JavaMaterializedGroupByOperator -> groupby, SparkUnionAllOperator -> union, ... (the profilers' names). */
    static String operatorKind(String cls, String prefix) {
        String s = cls.startsWith(prefix) ? cls.substring(prefix.length()) : cls;
        if (s.endsWith("Operator")) s = s.substring(0, s.length() - "Operator".length());
        switch (s) {
            case "MaterializedGroupBy": return "groupby";
            case "UnionAll": return "union";
            case "TextFileSink": return "textsink";
            default: return s.toLowerCase();
        }
    }

    /** Spark task slots: local[N] -> N, local[*] -> machine CPUs, else Wayang's machines x cores. */
    static double sparkSlots(Configuration configuration, int machineCpus) {
        final String master = configuration.getStringProperty("spark.master", "local[*]");
        final Matcher m = LOCAL_MASTER.matcher(master == null ? "" : master.trim());
        if (m.matches()) return m.group(1).equals("*") ? machineCpus : Integer.parseInt(m.group(1));
        if ("local".equals(master)) return 1;
        return configuration.getLongProperty("wayang.spark.machines", 1)
                * configuration.getLongProperty("wayang.spark.cores-per-machine", machineCpus);
    }

    private static OptionalLong fileSize(ExecutionOperator op) {
        try {
            final Object url = op.getClass().getMethod("getInputUrl").invoke(op);
            return url == null ? OptionalLong.empty() : FileSystems.getFileSize(url.toString());
        } catch (ReflectiveOperationException | RuntimeException e) {
            return OptionalLong.empty();
        }
    }

    private static double mid(CardinalityEstimate c) {
        return c == null ? 0d : (c.getLowerEstimate() + c.getUpperEstimate()) / 2d;
    }

    /** Middle of a resource's interval, summed over the profile and its UDF sub-profiles (as the profilers do). */
    private static double midOf(LoadProfile lp, Function<LoadProfile, LoadEstimate> resource) {
        final List<LoadProfile> all = new ArrayList<>();
        all.add(lp);
        all.addAll(lp.getSubprofiles());
        double lo = 0, hi = 0;
        for (LoadProfile p : all) {
            final LoadEstimate e = resource.apply(p);
            if (e == null) continue;
            lo += Math.max(0, e.getLowerEstimate());
            hi += Math.max(0, e.getUpperEstimate());
        }
        return (lo + hi) / 2d;
    }

    private static double log2(double x) {
        return Math.log(x) / Math.log(2);
    }

    /** Same buckets as features.gc_kind. */
    static String gcKind() {
        final StringBuilder names = new StringBuilder();
        for (GarbageCollectorMXBean b : ManagementFactory.getGarbageCollectorMXBeans()) names.append(b.getName()).append(' ');
        final String n = names.toString().toLowerCase();
        if (n.contains("g1")) return "G1";
        if (n.contains("ps ") || n.contains("parallel")) return "Parallel";
        if (n.contains("copy") || n.contains("marksweepcompact")) return "Serial";
        if (n.contains("zgc")) return "Z";
        return "other";
    }
}
