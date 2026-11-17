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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * The two-layer Java cost model of {@code wayang-java-cost-model.ipynb} (exported by
 * {@code Cost-Model/model/export_two_layer_java.py}), in plain Java:
 * <pre>
 *   x       = [StandardScaler(numeric), OneHot(categorical)]           (one preprocessor per network)
 *   z_res   = mean over folds of MultiHeadNet_k(x_k)                   (cpu, held memory, allocation, disk write)
 *   z_time  = MultiHeadNet_time([x_2, ResourceScaler(z_res)])
 *   time_ns = time_scale * expm1(max(z_time, 0))
 * </pre>
 * MultiHeadNet: Linear-ReLU-Linear-ReLU trunk, per output a Linear-ReLU-Linear head (dropout is off at inference).
 */
public final class TwoLayerNetwork {

    /** StandardScaler + OneHotEncoder(handle_unknown="ignore"), as one sklearn ColumnTransformer. */
    static final class Preprocessor {
        final List<String> numeric = new ArrayList<>(), categorical = new ArrayList<>();
        final double[] mean, scale;
        final List<List<String>> categories = new ArrayList<>();

        Preprocessor(JsonNode n) {
            for (JsonNode c : n.get("numeric")) this.numeric.add(c.asText());
            for (JsonNode c : n.get("categorical")) this.categorical.add(c.asText());
            this.mean = vec(n.get("mean"));
            this.scale = vec(n.get("scale"));
            for (JsonNode cats : n.get("categories")) {
                final List<String> l = new ArrayList<>();
                for (JsonNode c : cats) l.add(c.asText());
                this.categories.add(l);
            }
        }

        double[] transform(Map<String, Double> num, Map<String, String> cat) {
            int width = this.numeric.size();
            for (List<String> c : this.categories) width += c.size();
            final double[] x = new double[width];
            for (int i = 0; i < this.numeric.size(); i++) {
                x[i] = (num.get(this.numeric.get(i)) - this.mean[i]) / this.scale[i];
            }
            int off = this.numeric.size();
            for (int j = 0; j < this.categorical.size(); j++) {
                final int k = this.categories.get(j).indexOf(cat.get(this.categorical.get(j)));
                if (k >= 0) x[off + k] = 1;                  // unknown value: all zeros
                off += this.categories.get(j).size();
            }
            return x;
        }

        boolean knows(String categorical, String value) {
            final int j = this.categorical.indexOf(categorical);
            return j >= 0 && this.categories.get(j).contains(value);
        }
    }

    /** MultiHeadNet(hidden=(256, 128), head=64). */
    static final class MultiHeadNet {
        final double[][] w0, w3;
        final double[] b0, b3;
        final List<double[][]> hw0 = new ArrayList<>(), hw2 = new ArrayList<>();
        final List<double[]> hb0 = new ArrayList<>(), hb2 = new ArrayList<>();

        MultiHeadNet(JsonNode n) {
            this.w0 = mat(n.get("trunk.0.weight"));
            this.b0 = vec(n.get("trunk.0.bias"));
            this.w3 = mat(n.get("trunk.3.weight"));
            this.b3 = vec(n.get("trunk.3.bias"));
            for (int i = 0; n.has("heads." + i + ".0.weight"); i++) {
                this.hw0.add(mat(n.get("heads." + i + ".0.weight")));
                this.hb0.add(vec(n.get("heads." + i + ".0.bias")));
                this.hw2.add(mat(n.get("heads." + i + ".2.weight")));
                this.hb2.add(vec(n.get("heads." + i + ".2.bias")));
            }
        }

        double[] forward(double[] x) {
            final double[] h = relu(affine(this.w3, this.b3, relu(affine(this.w0, this.b0, x))));
            final double[] out = new double[this.hw0.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = affine(this.hw2.get(i), this.hb2.get(i), relu(affine(this.hw0.get(i), this.hb0.get(i), h)))[0];
            }
            return out;
        }
    }

    private final List<Preprocessor> foldPre = new ArrayList<>();
    private final List<MultiHeadNet> foldNets = new ArrayList<>();
    private final Preprocessor pre2;
    private final MultiHeadNet timeNet;
    private final double[] resMean, resScale;
    private final double timeScale;

    private TwoLayerNetwork(JsonNode doc) {
        for (JsonNode f : doc.get("folds")) {
            this.foldPre.add(new Preprocessor(f.get("preprocessor")));
            this.foldNets.add(new MultiHeadNet(f.get("net")));
        }
        final JsonNode l2 = doc.get("layer2");
        this.pre2 = new Preprocessor(l2.get("preprocessor"));
        this.timeNet = new MultiHeadNet(l2.get("net"));
        this.resMean = vec(l2.at("/resource_scaler/mean"));
        this.resScale = vec(l2.at("/resource_scaler/scale"));
        this.timeScale = doc.path("time_scale").asDouble(1e6);
    }

    static TwoLayerNetwork fromJson(JsonNode doc) {
        return new TwoLayerNetwork(doc);
    }

    /** Predicted execution time in ns (on the machine the model was trained on). */
    public double predictTimeNs(Map<String, Double> numeric, Map<String, String> categorical) {
        final double[] zRes = new double[this.resMean.length];
        for (int k = 0; k < this.foldNets.size(); k++) {
            final double[] z = this.foldNets.get(k).forward(this.foldPre.get(k).transform(numeric, categorical));
            for (int j = 0; j < zRes.length; j++) zRes[j] += z[j] / this.foldNets.size();
        }
        final double[] x = this.pre2.transform(numeric, categorical);
        final double[] x2 = new double[x.length + zRes.length];
        System.arraycopy(x, 0, x2, 0, x.length);
        for (int j = 0; j < zRes.length; j++) x2[x.length + j] = (zRes[j] - this.resMean[j]) / this.resScale[j];
        final double z = this.timeNet.forward(x2)[0];
        return this.timeScale * Math.expm1(Math.max(z, 0));
    }

    public boolean knows(String categorical, String value) {
        return this.pre2.knows(categorical, value);
    }

    /** Training mean of a numeric feature (layer-2 scaler). */
    public double trainingMean(String numeric) {
        return this.pre2.mean[this.pre2.numeric.indexOf(numeric)];
    }

    // ------------------------------------------------------------------ math
    private static double[] affine(double[][] w, double[] b, double[] x) {
        final double[] y = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            double s = b[i];
            for (int j = 0; j < x.length; j++) s += w[i][j] * x[j];
            y[i] = s;
        }
        return y;
    }

    private static double[] relu(double[] x) {
        for (int i = 0; i < x.length; i++) x[i] = Math.max(0, x[i]);
        return x;
    }

    static double[] vec(JsonNode n) {
        final double[] v = new double[n.size()];
        for (int i = 0; i < v.length; i++) v[i] = n.get(i).asDouble();
        return v;
    }

    static double[][] mat(JsonNode n) {
        final double[][] m = new double[n.size()][];
        for (int i = 0; i < m.length; i++) m[i] = vec(n.get(i));
        return m;
    }
}
