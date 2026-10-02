#!/usr/bin/env python3
"""
Campus IQ - Model & System Evaluation Report Generator
Produces a comprehensive statistical & machine learning benchmark report in the terminal.
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_FILE = os.path.join(BASE_DIR, "models_saved", "pipeline_summary.json")

def print_evaluation_report():
    if not os.path.exists(SUMMARY_FILE):
        print(f"Error: Summary file {SUMMARY_FILE} not found. Run run_pipeline.py first.")
        return

    with open(SUMMARY_FILE, "r") as f:
        data = json.load(f)

    meta = data["dataset_metadata"]
    print("\n" + "=" * 80)
    print("           CAMPUS IQ: STATISTICAL MACHINE LEARNING BENCHMARK REPORT")
    print("=" * 80)
    print(f" Dataset Scope:       {meta['total_records']} Total Campus Observations (Train: {meta['train_records']}, Test: {meta['test_records']})")
    print(f" Methodological Note: {meta['source_transparency']}")
    print(f" Overcrowding Rate:   {meta['overcrowded_rate_pct']}% of observed intervals")
    print(f" Average Campus Pulse:{meta['avg_campus_pulse_index']} / 100")
    print("-" * 80)

    # 1. Sports Models
    print("\n[1] SPORTS COMPLEX OVERCROWDING & DEMAND MODELS (PRIMARY FOCUS)")
    print("  Regression Benchmarks (Target: Continuous Visitor Count):")
    print("  {:<30} {:<10} {:<10} {:<10}".format("Model Candidate", "MAE", "RMSE", "R² Score"))
    print("  " + "-" * 62)
    for name, m in data["sports_models"]["regressors"].items():
        star = " *" if name == data["sports_models"]["best_regressor"] else ""
        print("  {:<30} {:<10} {:<10} {:<10}{}".format(name, m["MAE"], m["RMSE"], m["R2"], star))

    print("\n  Classification Benchmarks (Target: Overcrowding Risk Flag):")
    print("  {:<30} {:<10} {:<10} {:<10} {:<10}".format("Model Candidate", "Accuracy", "Precision", "Recall", "F1-Score"))
    print("  " + "-" * 72)
    for name, m in data["sports_models"]["classifiers"].items():
        star = " *" if name == data["sports_models"]["best_classifier"] else ""
        print("  {:<30} {:<10} {:<10} {:<10} {:<10}{}".format(name, m["Accuracy"], m["Precision"], m["Recall"], m["F1-Score"], star))

    # 2. Food Models
    print("\n[2] FOOD OUTLETS DEMAND & WASTE MINIMIZATION MODELS (SDG 12)")
    print("  Regression Benchmarks (Target: Order Demand):")
    print("  {:<30} {:<10} {:<10} {:<10}".format("Model Candidate", "MAE", "RMSE", "R² Score"))
    print("  " + "-" * 62)
    for name, m in data["food_models"]["regressors"].items():
        star = " *" if name == data["food_models"]["best_regressor"] else ""
        print("  {:<30} {:<10} {:<10} {:<10}{}".format(name, m["MAE"], m["RMSE"], m["R2"], star))

    # 3. Clustering
    print("\n[3] CAMPUS ACTIVITY PATTERNS (K-MEANS + PCA)")
    cluster_info = data["clustering_and_pca"]
    print(f"  Optimal Clusters: K = {cluster_info['optimal_k']}")
    print(f"  PCA Variance Explained: {cluster_info['pca_variance_explained']} (Total: {sum(cluster_info['pca_variance_explained'])*100:.1f}%)")
    print("  Cluster Persona Profiles:")
    for cid, p in cluster_info["cluster_profiles"].items():
        print(f"   * Cluster {cid}: {p['label']} ({p['percentage_of_data']}%)")
        print(f"     Avg Sports Load: {p['avg_sports_load_pct']}% | Food Orders: {p['avg_food_orders']} | Library: {p['avg_library_load_pct']}%")
        print(f"     Recommendation:  {p['recommended_action']}\n")

    # 4. Statistical Hypotheses
    print("[4] FORMAL STATISTICAL HYPOTHESIS TESTS (ALPHA = 0.05)")
    for i, h in enumerate(data["hypothesis_tests"], 1):
        print(f"  Test {i}: {h['hypothesis']}")
        print(f"   - Group A ({h['sample_a_label']}): Mean = {h['sample_a_mean']} ± {h['sample_a_std']}")
        print(f"   - Group B ({h['sample_b_label']}): Mean = {h['sample_b_mean']} ± {h['sample_b_std']}")
        print(f"   - Two-Sample t-test: t = {h['t_statistic']}, p = {h['p_value_ttest']:.2e}")
        print(f"   - Mann-Whitney U:    U = {h['mann_whitney_u']}, p = {h['p_value_mwu']:.2e}")
        print(f"   - Cohen's d Effect:  d = {h['cohens_d']} ({'Large' if abs(h['cohens_d']) >= 0.8 else 'Medium'})")
        print(f"   - Verdict:           {h['conclusion']}\n")

    # 5. Outliers
    sports_out = data["outlier_analysis"]["sports"]
    print("[5] IQR OUTLIER INVESTIGATION (SECTION 7)")
    print(f"  Sports Complex Crowd Outlier Threshold: > {sports_out['upper_bound']} visitors (Q75: {sports_out['q75']}, IQR: {sports_out['iqr']})")
    print(f"  Total Outliers Detected: {sports_out['outlier_count']}")
    print("  Action: Retained in dataset because genuine crowd spikes (festivals, tournaments) reflect real operational volatility.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    print_evaluation_report()
