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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Head-to-head under a monetary budget: how many partial plans each pruning method stores, how many plans it reports
 * at the end, and whether its final pick is the best plan within the budget.
 *
 * <ul>
 *   <li>{@code original}: Wayang's scalar pruning, one fastest plan per group, no budget awareness.</li>
 *   <li>{@code pareto-k8}: Pareto front per group capped at 8 (plus fastest and cheapest), budget-aware.</li>
 *   <li>{@code exact}: full Pareto front per group, budget-aware.</li>
 *   <li>{@code ours}: exact + exact cross-group pruning, budget-aware.</li>
 *   <li>{@code ours-k8}: ours with the cap of 8.</li>
 * </ul>
 * Budget: 40% of the way from the cheapest to the most expensive plan on the true front (the fastest plan is out).
 * Opt-in: {@code -Dwayang.report=true}; CSV goes to {@code target/pruning-report.csv}.
 */
@EnabledIfSystemProperty(named = "wayang.report", matches = "true")
class PruningComparisonReportTest {

    private static final int K = 8;
    private static final double BUDGET_FRACTION = 0.4;
    private static final String[] MODES = {"original", "pareto-k8", "exact", "ours", "ours-k8"};

    @Test
    void report() throws IOException {
        final List<PlanDagSimulator.Graph> graphs = new ArrayList<>();
        graphs.add(PlanDagSimulator.chain("chain", 30, 4));
        graphs.add(PlanDagSimulator.chain("chain", 60, 8));
        graphs.add(PlanDagSimulator.threeDiamonds("3-diamond", 4));
        graphs.add(PlanDagSimulator.bushy("bushy", 4));
        graphs.add(PlanDagSimulator.complexBranching("branching", 4));
        for (int tables : new int[]{8, 16, 32}) graphs.add(PlanDagSimulator.leftDeepInterleaved("join" + tables, tables, 4));
        for (int n : new int[]{12, 16, 20}) {
            for (long seed = 1; seed <= 3; seed++) graphs.add(PlanDagSimulator.randomConnected("dag" + n + "-" + seed, n, 4, seed));
        }
        graphs.add(PlanDagSimulator.randomConnected("dag25-1", 25, 4, 1));
        graphs.add(PlanDagSimulator.randomConnected("dag25-2", 25, 4, 2));

        final Path csv = Paths.get("target", "pruning-report.csv");
        final Path frontsCsv = Paths.get("target", "pruning-fronts.csv");
        Files.createDirectories(csv.getParent());
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(csv));
             PrintWriter fronts = new PrintWriter(Files.newBufferedWriter(frontsCsv))) {
            out.println("graph,ops,platforms,brute_force,mode,generated,stored_total,peak,final,pick_latency,pick_money,"
                    + "budget,feasible,latency_vs_best,time_ms,pick_plan,best_latency,best_money,best_plan");
            fronts.println("graph,mode,latency,money,plan,picked,budget");
            System.out.printf("%-11s %4s | %-9s %11s %11s %9s %6s %9s %8s%n",
                    "graph", "ops", "mode", "generated", "stored", "peak", "final", "feasible", "lat/best");

            final Map<String, List<double[]>> ratios = new LinkedHashMap<>();
            for (PlanDagSimulator.Graph g : graphs) {
                // True front (no budget) -> budget and the best plan within it.
                final List<PlanDagSimulator.Partial> trueFront = PlanDagSimulator.optimize(g, 0d, 0, 0d).front;
                final double budget = PlanDagSimulator.budgetAt(trueFront, BUDGET_FRACTION);
                final PlanDagSimulator.Partial best = PlanDagSimulator.pick(trueFront, budget);
                final double brute = Math.pow(g.nPlats, g.nOps());

                final Map<String, PlanDagSimulator.SearchResult> results = new LinkedHashMap<>();
                for (String mode : MODES) {
                    final long t = System.nanoTime();
                    final PlanDagSimulator.SearchResult r = run(g, mode, budget);
                    final long ms = (System.nanoTime() - t) / 1_000_000;
                    results.put(mode, r);
                    final PlanDagSimulator.Partial pick = mode.equals("original")
                            ? r.front.get(0)                              // Wayang returns its single best plan
                            : PlanDagSimulator.pick(r.front, budget);
                    final boolean feasible = pick.money <= budget + 1e-9;
                    final double latVsBest = pick.lat / best.lat;
                    System.out.printf("%-11s %4d | %-9s %11d %11d %9d %6d %9s %8.4f%n",
                            g.name, g.nOps(), mode, r.generated, r.keptSum, r.peakKept, r.finalKept,
                            feasible ? "yes" : "NO", latVsBest);
                    out.printf("%s,%d,%d,%.0f,%s,%d,%d,%d,%d,%.4f,%.4f,%.4f,%b,%.6f,%d,%s,%.4f,%.4f,%s%n",
                            g.name, g.nOps(), g.nPlats, brute, mode, r.generated, r.keptSum, r.peakKept, r.finalKept,
                            pick.lat, pick.money, budget, feasible, latVsBest, ms, plan(pick), best.lat, best.money,
                            plan(best));
                    for (PlanDagSimulator.Partial f : r.front) {
                        fronts.printf("%s,%s,%.4f,%.4f,%s,%b,%.4f%n",
                                g.name, mode, f.lat, f.money, plan(f), f == pick, budget);
                    }

                    // Exact methods must find the best plan within the budget.
                    if (mode.equals("exact") || mode.equals("ours")) {
                        assertTrue(feasible, g.name + "/" + mode + " infeasible");
                        assertEquals(best.lat, pick.lat, 1e-6, g.name + "/" + mode + " not optimal");
                    }
                }
                final double e = results.get("exact").keptSum, o = results.get("ours").keptSum;
                ratios.computeIfAbsent(family(g.name), k -> new ArrayList<>()).add(new double[]{
                        e / o,                                                   // exact / ours
                        e / results.get("pareto-k8").keptSum,                    // exact / pareto-k8
                        e / results.get("original").keptSum,                     // exact / original
                        o / results.get("original").keptSum,                     // ours / original
                        e / results.get("ours-k8").keptSum,                      // exact / ours-k8
                });
                System.out.println();
            }

            System.out.println("Stored partial plans, geometric mean of ratios per family:");
            System.out.printf("%-10s %4s | %12s %14s %15s %14s %13s%n",
                    "family", "n", "exact/ours", "exact/pareto-k8", "exact/original", "ours/original", "exact/ours-k8");
            for (Map.Entry<String, List<double[]>> e : ratios.entrySet()) {
                System.out.printf("%-10s %4d | %11.1fx %13.1fx %14.1fx %13.2fx %12.1fx%n", e.getKey(), e.getValue().size(),
                        geo(e.getValue(), 0), geo(e.getValue(), 1), geo(e.getValue(), 2), geo(e.getValue(), 3),
                        geo(e.getValue(), 4));
            }
        }
        System.out.println("CSV: " + csv.toAbsolutePath() + " and " + frontsCsv.toAbsolutePath());
    }

    private static PlanDagSimulator.SearchResult run(PlanDagSimulator.Graph g, String mode, double budget) {
        switch (mode) {
            case "original":  return PlanDagSimulator.optimize(g, 0d, -1, -1d, Double.POSITIVE_INFINITY);
            case "pareto-k8": return PlanDagSimulator.optimize(g, 0d, K, -1d, budget);
            case "exact":     return PlanDagSimulator.optimize(g, 0d, 0, -1d, budget);
            case "ours":      return PlanDagSimulator.optimize(g, 0d, 0, 0d, budget);
            case "ours-k8":   return PlanDagSimulator.optimize(g, 0d, K, 0d, budget);
            default: throw new IllegalArgumentException(mode);
        }
    }

    /** Platform per operator, e.g. {@code 0012} = operators 0,1 on platform 0, operator 2 on 1, operator 3 on 2. */
    private static String plan(PlanDagSimulator.Partial p) {
        final StringBuilder sb = new StringBuilder(p.plat.length);
        for (byte b : p.plat) sb.append(b);
        return sb.toString();
    }

    private static String family(String name) {
        if (name.startsWith("dag")) return name.substring(0, name.indexOf('-'));
        if (name.startsWith("join")) return "joins";
        if (name.startsWith("chain")) return "chains";
        return "small-dags";
    }

    private static double geo(List<double[]> rows, int col) {
        double s = 0;
        for (double[] r : rows) s += Math.log(r[col]);
        return Math.exp(s / rows.size());
    }
}
