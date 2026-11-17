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

import org.apache.wayang.core.api.Configuration;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.io.File;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LearnedCostNetworkTest {

    /** model.json from Cost-Model/model/export_weights.py; override with -Dwayang.learned-model=... */
    private static File modelFile() {
        final String p = System.getProperty("wayang.learned-model",
                "../../../Cost-Model/model/runs/C-all-data/model.json");
        return new File(p);
    }

    @Test
    void reproducesPyTorch() throws Exception {
        final File f = modelFile();
        Assumptions.assumeTrue(f.isFile(), "no exported model at " + f.getAbsolutePath());
        final LearnedCostNetwork net = LearnedCostNetwork.load(f.getPath());
        assertTrue(net.numReferenceRows() > 0);
        final double err = net.maxReferenceError();
        assertTrue(err < 1e-4, "max relative error vs PyTorch: " + err);
    }

    @Test
    void twoLayerModelsReproducePython() throws Exception {
        for (String run : new String[]{"kaggle-java", "kaggle-spark-combined"}) {
            final File f = new File("../../../Cost-Model/model/runs/" + run + "/model.json");
            Assumptions.assumeTrue(f.isFile(), "no exported two-layer model at " + f.getAbsolutePath());
            final com.fasterxml.jackson.databind.JsonNode doc = new com.fasterxml.jackson.databind.ObjectMapper().readTree(f);
            final double err = TwoLayerTimeModel.referenceError(TwoLayerNetwork.fromJson(doc), doc);
            assertTrue(doc.path("reference").size() > 0);
            assertTrue(err < 1e-4, run + ": max relative error vs Python: " + err);
        }
    }

    @Test
    void mathMatchesTorch() {
        // torch.erf(1) = 0.8427007929, torch.nn.functional.gelu(1) = 0.8413447461, softplus(0) = ln 2
        assertEquals(0.8427007929, LearnedCostNetwork.erf(1), 2e-7);
        assertEquals(-0.8427007929, LearnedCostNetwork.erf(-1), 2e-7);
        assertEquals(0.8413447461, LearnedCostNetwork.gelu(1), 2e-7);
        assertEquals(Math.log(2), LearnedCostNetwork.softplus(0), 1e-12);
        assertEquals(25, LearnedCostNetwork.softplus(25), 0);
    }

    @Test
    void operatorNamesMatchTheProfilers() {
        assertEquals("map", LearnedOperatorTimeModel.operatorKind("JavaMapOperator", "Java"));
        assertEquals("reduceby", LearnedOperatorTimeModel.operatorKind("SparkReduceByOperator", "Spark"));
        assertEquals("groupby", LearnedOperatorTimeModel.operatorKind("JavaMaterializedGroupByOperator", "Java"));
        assertEquals("union", LearnedOperatorTimeModel.operatorKind("SparkUnionAllOperator", "Spark"));
        assertEquals("textsink", LearnedOperatorTimeModel.operatorKind("JavaTextFileSink", "Java"));
        assertEquals("globalreduce", LearnedOperatorTimeModel.operatorKind("JavaGlobalReduceOperator", "Java"));
    }

    @Test
    void sparkSlotsFromMaster() {
        final Configuration c = new Configuration();
        c.setProperty("spark.master", "local[4]");
        assertEquals(4, LearnedOperatorTimeModel.sparkSlots(c, 12), 0);
        c.setProperty("spark.master", "local[*]");
        assertEquals(12, LearnedOperatorTimeModel.sparkSlots(c, 12), 0);
    }
}
