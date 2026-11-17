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
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Forward pass of the trained operator cost model ({@code Cost-Model/model/model.py}) in plain Java, from the JSON
 * written by {@code Cost-Model/model/export_weights.py}. No ML runtime is needed.
 *
 * <pre>
 *   x = [embeddings(operator, impl, platform, gc, source), gelu(adapter_platform(cfg)), ctx]
 *   h = gelu(W2 gelu(W1 x + b1) + b2)
 *   demand_r = a_r(h) * wayang_r + sum_k b_rk(h) * basis_k          (a, b = softplus(expert_r(h)))
 *   runtime  = overhead(h) + (t_cpu^p + t_io^p + t_net^p)^(1/p) * (1 + memory pressure)
 * </pre>
 */
public final class LearnedCostNetwork {

    public static final String[] RESOURCES = {"cpu", "io", "net", "mem"};

    private final List<String> categories = new ArrayList<>();
    private final Map<String, Map<String, Integer>> vocab = new HashMap<>();
    private final double[] cfgMean, cfgStd, ctxMean, ctxStd;
    private final Map<String, double[][]> emb = new HashMap<>();
    private final double[][][] adapterW;   // [platform][cfg][adapter]
    private final double[][] adapterB;     // [platform][adapter]
    private final double[][] w1, w2, wRt;
    private final double[] b1, b2, bRt;
    private final Map<String, double[][]> expertW = new HashMap<>();
    private final Map<String, double[]> expertB = new HashMap<>();
    private final Map<String, Double> basisUnit = new HashMap<>();
    private final List<String> wayangOnly = new ArrayList<>();
    private final double[] logBwIo, logBwNet, pRaw, memRaw;
    private final JsonNode reference;
    private final String source;

    private LearnedCostNetwork(JsonNode doc) {
        this.source = doc.path("source").asText("");
        for (JsonNode c : doc.get("categories")) this.categories.add(c.asText());
        doc.get("vocab").fields().forEachRemaining(e -> {
            final Map<String, Integer> m = new HashMap<>();
            e.getValue().fields().forEachRemaining(v -> m.put(v.getKey(), v.getValue().asInt()));
            this.vocab.put(e.getKey(), m);
        });
        this.cfgMean = vec(doc.at("/norm/mean/cfg"));
        this.cfgStd = vec(doc.at("/norm/std/cfg"));
        this.ctxMean = vec(doc.at("/norm/mean/ctx"));
        this.ctxStd = vec(doc.at("/norm/std/ctx"));
        final JsonNode w = doc.get("weights");
        for (String c : this.categories) this.emb.put(c, mat(w.get("emb." + c + ".weight")));
        this.adapterW = ten3(w.get("adapter_w"));
        this.adapterB = mat(w.get("adapter_b"));
        this.w1 = mat(w.get("encoder.0.weight"));
        this.b1 = vec(w.get("encoder.0.bias"));
        this.w2 = mat(w.get("encoder.3.weight"));
        this.b2 = vec(w.get("encoder.3.bias"));
        for (String r : RESOURCES) {
            this.expertW.put(r, mat(w.get("experts." + r + "_expert.weight")));
            this.expertB.put(r, vec(w.get("experts." + r + "_expert.bias")));
        }
        this.wRt = mat(w.get("runtime_head.weight"));
        this.bRt = vec(w.get("runtime_head.bias"));
        this.logBwIo = vec(w.get("log_bw_io"));
        this.logBwNet = vec(w.get("log_bw_net"));
        this.pRaw = vec(w.get("p_raw"));
        this.memRaw = vec(w.get("mem_raw"));
        doc.get("basis_unit").fields().forEachRemaining(e -> this.basisUnit.put(e.getKey(), e.getValue().asDouble()));
        for (JsonNode r : doc.path("wayang_only")) this.wayangOnly.add(r.asText());
        this.reference = doc.path("reference");
    }

    public static LearnedCostNetwork load(String path) throws IOException {
        return new LearnedCostNetwork(new ObjectMapper().readTree(new File(path)));
    }

    /** Index of a category value, 0 (= unknown) if the model has not seen it. */
    public int index(String category, String value) {
        final Map<String, Integer> m = this.vocab.get(category);
        if (m == null || value == null) return 0;
        return m.getOrDefault(value, 0);
    }

    public boolean knows(String category, String value) {
        return this.index(category, value) != 0;
    }

    public List<String> getCategories() {
        return this.categories;
    }

    public String getSource() {
        return this.source;
    }

    /**
     * Runs the model on raw (unnormalized) inputs, exactly as {@code features.build_arrays} produces them.
     *
     * @param zClip if &gt; 0, normalized config/context values are clipped to [-zClip, zClip] so the network is not
     *              driven far outside its training range (row counts are unaffected: they only enter the basis)
     * @return {cpu core-s, io GB, net GB, mem GB, runtime s}
     */
    public double[] forward(int[] cat, double[] cfgRaw, double[] ctxRaw, double[] basis, double[] wayang,
                            double[] aux, double zClip) {
        final int plat = cat[this.categories.indexOf("platform")];
        final double[] cfg = normalize(cfgRaw, this.cfgMean, this.cfgStd, zClip);
        final double[] ctx = normalize(ctxRaw, this.ctxMean, this.ctxStd, zClip);

        final List<Double> x = new ArrayList<>();
        for (int i = 0; i < this.categories.size(); i++) {
            for (double v : this.emb.get(this.categories.get(i))[cat[i]]) x.add(v);
        }
        final int nAdapter = this.adapterB[plat].length;
        for (int a = 0; a < nAdapter; a++) {
            double s = this.adapterB[plat][a];
            for (int c = 0; c < cfg.length; c++) s += cfg[c] * this.adapterW[plat][c][a];
            x.add(gelu(s));
        }
        for (double v : ctx) x.add(v);
        final double[] in = new double[x.size()];
        for (int i = 0; i < in.length; i++) in[i] = x.get(i);

        final double[] h = gelu(affine(this.w2, this.b2, gelu(affine(this.w1, this.b1, in))));

        final double[] out = new double[5];
        for (int i = 0; i < RESOURCES.length; i++) {
            final String r = RESOURCES[i];
            if (this.wayangOnly.contains(r)) {
                out[i] = wayang[i];
                continue;
            }
            final double[] z = affine(this.expertW.get(r), this.expertB.get(r), h);
            double v = softplus(z[0]) * wayang[i];
            final double unit = this.basisUnit.getOrDefault(r, 1d);
            for (int k = 1; k < z.length; k++) v += softplus(z[k]) * unit * basis[k - 1];
            out[i] = v;
        }

        final double threads = aux[0], heapGb = aux[1];
        final double[] rt = affine(this.wRt, this.bRt, h);
        final double overhead = softplus(rt[0]) * 0.1;
        final double eff = 1d / (1d + Math.exp(-rt[1]));
        final double tCpu = out[0] / (1 + eff * (threads - 1));
        final double tIo = out[1] / Math.exp(this.logBwIo[plat]);
        final double tNet = out[2] / Math.exp(this.logBwNet[plat]);
        final double p = 1 + softplus(this.pRaw[plat]);
        final double[] logs = {p * Math.log(Math.max(tCpu, 1e-9)), p * Math.log(Math.max(tIo, 1e-9)),
                p * Math.log(Math.max(tNet, 1e-9))};
        final double max = Math.max(logs[0], Math.max(logs[1], logs[2]));
        final double lse = max + Math.log(Math.exp(logs[0] - max) + Math.exp(logs[1] - max) + Math.exp(logs[2] - max));
        final double core = Math.exp(lse / p);
        final double pressure = softplus(this.memRaw[plat]) * Math.max(0d, out[3] / Math.max(heapGb, 1e-3) - 0.5);
        out[4] = overhead + core * (1 + pressure);
        return out;
    }

    /**
     * Largest relative difference between this implementation and PyTorch on the exported reference rows.
     */
    public double maxReferenceError() {
        double worst = 0;
        for (JsonNode r : this.reference) {
            final int[] cat = new int[r.get("cat").size()];
            for (int i = 0; i < cat.length; i++) cat[i] = r.get("cat").get(i).asInt();
            final double[] out = this.forward(cat, vec(r.get("cfg")), vec(r.get("ctx")), vec(r.get("basis")),
                    vec(r.get("wayang")), vec(r.get("aux")), 0d);
            final String[] keys = {"cpu", "io", "net", "mem", "runtime"};
            for (int i = 0; i < keys.length; i++) {
                final double e = r.get("expected").get(keys[i]).asDouble();
                worst = Math.max(worst, Math.abs(out[i] - e) / Math.max(Math.abs(e), 1e-6));
            }
        }
        return worst;
    }

    public int numReferenceRows() {
        return this.reference.size();
    }

    // ------------------------------------------------------------------ math
    private static double[] normalize(double[] raw, double[] mean, double[] std, double zClip) {
        final double[] z = new double[raw.length];
        for (int i = 0; i < raw.length; i++) {
            z[i] = (raw[i] - mean[i]) / std[i];
            if (zClip > 0) z[i] = Math.max(-zClip, Math.min(zClip, z[i]));
        }
        return z;
    }

    private static double[] affine(double[][] w, double[] b, double[] x) {
        final double[] y = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            double s = b[i];
            for (int j = 0; j < x.length; j++) s += w[i][j] * x[j];
            y[i] = s;
        }
        return y;
    }

    private static double[] gelu(double[] x) {
        final double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) y[i] = gelu(x[i]);
        return y;
    }

    /** Exact GELU, as torch.nn.GELU(): x * Phi(x). */
    static double gelu(double x) {
        return 0.5 * x * (1 + erf(x / Math.sqrt(2)));
    }

    /** torch.nn.functional.softplus with beta = 1, threshold = 20. */
    static double softplus(double x) {
        return x > 20 ? x : Math.log1p(Math.exp(x));
    }

    /** erf with relative error below 1.2e-7 (Numerical Recipes, erfc via Chebyshev fit). */
    static double erf(double x) {
        final double z = Math.abs(x);
        final double t = 1d / (1d + 0.5 * z);
        final double ans = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
                + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
                + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? 1 - ans : ans - 1;
    }

    private static double[] vec(JsonNode n) {
        final double[] v = new double[n.size()];
        for (int i = 0; i < v.length; i++) v[i] = n.get(i).asDouble();
        return v;
    }

    private static double[][] mat(JsonNode n) {
        final double[][] m = new double[n.size()][];
        for (int i = 0; i < m.length; i++) m[i] = vec(n.get(i));
        return m;
    }

    private static double[][][] ten3(JsonNode n) {
        final double[][][] t = new double[n.size()][][];
        for (int i = 0; i < t.length; i++) t[i] = mat(n.get(i));
        return t;
    }
}
