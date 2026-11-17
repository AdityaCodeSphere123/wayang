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
import org.apache.logging.log4j.LogManager;
import org.apache.wayang.core.api.Configuration;
import org.apache.wayang.core.optimizer.OptimizationContext;
import org.apache.wayang.core.optimizer.costs.LoadProfile;
import org.apache.wayang.core.optimizer.costs.TimeEstimate;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A learned per-operator time model. {@code wayang.core.optimizer.learned-cost-model} lists one or more exported model
 * files (comma-separated); for each operator the first model that knows it gives the time, otherwise Wayang's own
 * estimate is kept.
 */
public interface OperatorTimeModel {

    String PROPERTY = "wayang.core.optimizer.learned-cost-model";

    /**
     * @return the learned time estimate, or {@code null} if this model does not cover the operator
     */
    TimeEstimate estimate(OptimizationContext.OperatorContext ctx, LoadProfile profile, TimeEstimate wayangEstimate,
                          Configuration configuration);

    /** The configured model(s), or {@code null} if none. Each file is loaded once per JVM. */
    static OperatorTimeModel forConfiguration(Configuration configuration) {
        final String spec = configuration.getStringProperty(PROPERTY, "");
        if (spec == null || spec.trim().isEmpty()) return null;
        return Holder.CACHE.computeIfAbsent(spec.trim(), s -> {
            final List<OperatorTimeModel> models = new ArrayList<>();
            for (String path : s.split(",")) {
                if (!path.trim().isEmpty()) models.add(load(path.trim()));
            }
            if (models.size() == 1) return models.get(0);
            return (ctx, profile, wayang, conf) -> {
                for (OperatorTimeModel m : models) {
                    final TimeEstimate t = m.estimate(ctx, profile, wayang, conf);
                    if (t != null) return t;
                }
                return null;
            };
        });
    }

    /** Loads a model file, choosing the implementation by its {@code format} field. */
    static OperatorTimeModel load(String path) {
        try {
            final JsonNode doc = new ObjectMapper().readTree(new File(path));
            final String format = doc.path("format").asText("");
            final OperatorTimeModel model;
            if (format.startsWith("wayang-two-layer/")) {
                model = new TwoLayerTimeModel(TwoLayerNetwork.fromJson(doc), doc);
            } else if (format.startsWith("wayang-learned-cost-model/")) {
                model = LearnedOperatorTimeModel.load(path);
            } else {
                throw new IllegalArgumentException("Unknown learned cost model format \"" + format + "\" in " + path);
            }
            LogManager.getLogger(OperatorTimeModel.class).info("Loaded learned cost model {} ({})", path, format);
            return model;
        } catch (Exception e) {
            throw new IllegalStateException("Cannot load learned cost model " + path, e);
        }
    }

    final class Holder {
        static final Map<String, OperatorTimeModel> CACHE = new ConcurrentHashMap<>();

        private Holder() {
        }
    }
}
