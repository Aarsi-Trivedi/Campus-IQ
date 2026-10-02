#!/usr/bin/env python3
"""
Campus IQ - Master Training & System Pipeline
Executes the full 12-step pipeline defined in Section 9 of the project summary:
1. Campus Data Loading & Integrity Check
2. Data Preprocessing & Leakage-Free Chronological Splitting
3. Statistical Analysis & Outlier Detection
4. Feature Engineering
5. Sports Demand & Overcrowding Models
6. Food Demand & Surplus/Shortage Models
7. Library / Study-Space Analysis
8. Campus Activity Pattern Discovery (K-Means + PCA)
9. Campus Demand Flow Analysis
10. Campus Pulse Index Calibration
11. Action-Oriented Recommendation & Explainability Setup
12. Model Artifact Serialization for Dashboard Serving
"""

import os
import sys
import json
import joblib
import pandas as pd

# Add src to python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_processing import load_data, get_train_test_split, save_splits
from statistical_analysis import compute_descriptive_stats, compute_correlations, detect_outliers_iqr, run_hypothesis_tests, compute_trend_analysis
from models.sports_models import SportsModelSuite
from models.food_models import FoodModelSuite
from models.library_models import LibraryModelSuite
from models.clustering_pca import CampusClusterSuite
from cpi_engine import CampusPulseEngine
from demand_flow import CampusDemandFlowAnalyzer
from recommender import RecommendationEngine
from explainability import CampusExplainability

MODELS_DIR = os.path.join(BASE_DIR, "models_saved")
os.makedirs(MODELS_DIR, exist_ok=True)

def run_full_pipeline():
    print("=" * 70)
    print("CAMPUS IQ: END-TO-END STATISTICAL ML SYSTEM PIPELINE")
    print("=" * 70)

    # Step 1 & 2: Load and Split Data
    print("\n[Step 1-2] Loading Data and Creating Chronological Train/Test Splits...")
    df = load_data()
    train_df, test_df = get_train_test_split(df, test_size=0.2, chronological=True)
    save_splits(train_df, test_df)
    print(f"Total dataset: {len(df)} observations | Train: {len(train_df)} | Test: {len(test_df)}")

    # Step 3: Statistical Analysis & Outliers
    print("\n[Step 3] Running Descriptive Statistics, Outlier Analysis & Hypothesis Testing...")
    desc_df = compute_descriptive_stats(df)
    correlations = compute_correlations(df)
    outliers_sports = detect_outliers_iqr(df, "sports_visitor_count")
    outliers_food = detect_outliers_iqr(df, "food_orders_demand")
    hypotheses = run_hypothesis_tests(df)
    trends = compute_trend_analysis(df)
    print(f"Descriptive stats computed for {len(desc_df)} core metrics.")
    print(f"Sports crowd IQR outliers detected: {outliers_sports['outlier_count']} (investigated and contextualized).")
    print(f"Hypothesis tests concluded: {len(hypotheses)} formal tests.")

    # Step 5: Sports Demand & Overcrowding Models
    print("\n[Step 5] Training Sports Visitor Regressors & Overcrowding Classifiers...")
    sports_suite = SportsModelSuite()
    sports_reg_metrics, sports_clf_metrics = sports_suite.train_and_evaluate(train_df, test_df)
    print(f"Best Sports Regressor: {sports_suite.best_regressor_name} (R² = {sports_reg_metrics[sports_suite.best_regressor_name]['R2']}, MAE = {sports_reg_metrics[sports_suite.best_regressor_name]['MAE']})")
    print(f"Best Sports Classifier: {sports_suite.best_classifier_name} (F1 = {sports_clf_metrics[sports_suite.best_classifier_name]['F1-Score']}, Acc = {sports_clf_metrics[sports_suite.best_classifier_name]['Accuracy']})")

    # Step 6: Food Demand & Surplus/Shortage Models
    print("\n[Step 6] Training Food Demand Regressors & Surplus/Shortage Classifiers (SDG 12)...")
    food_suite = FoodModelSuite()
    food_reg_metrics, food_clf_metrics = food_suite.train_and_evaluate(train_df, test_df)
    print(f"Best Food Regressor: {food_suite.best_regressor_name} (R² = {food_reg_metrics[food_suite.best_regressor_name]['R2']}, MAE = {food_reg_metrics[food_suite.best_regressor_name]['MAE']})")
    print(f"Best Food Classifier: {food_suite.best_classifier_name} (F1 = {food_clf_metrics[food_suite.best_classifier_name]['F1-Score']}, Acc = {food_clf_metrics[food_suite.best_classifier_name]['Accuracy']})")

    # Step 7: Library / Study-Space Models
    print("\n[Step 7] Training Library Occupancy & Demand Models...")
    library_suite = LibraryModelSuite()
    lib_reg_metrics, lib_clf_metrics = library_suite.train_and_evaluate(train_df, test_df)
    print(f"Best Library Regressor: {library_suite.best_regressor_name} (R² = {lib_reg_metrics[library_suite.best_regressor_name]['R2']})")

    # Step 8: Campus Activity Pattern Discovery (K-Means + PCA)
    print("\n[Step 8] Discovering Campus Activity Regimes (K-Means + PCA)...")
    cluster_suite = CampusClusterSuite()
    clustering_results = cluster_suite.fit_and_evaluate_k(df, k_min=2, k_max=6)
    print(f"Optimal Clusters: K = {clustering_results['optimal_k']}")
    print(f"PCA Variance Explained: {clustering_results['pca_variance_explained']}")
    for cid, profile in clustering_results['cluster_profiles'].items():
        print(f" - Cluster {cid}: {profile['label']} ({profile['percentage_of_data']}%)")

    # Step 9: Campus Demand Flow Analysis
    print("\n[Step 9] Analyzing Cross-Facility Campus Demand Flow...")
    flow_analyzer = CampusDemandFlowAnalyzer(df)
    flow_results = flow_analyzer.analyze_cross_facility_flow()
    print(f"Sports-to-Food Lag Correlation: {flow_results['sports_to_food_lag_correlation']}")

    # Step 10 & 11: Recommendation & Explainability
    print("\n[Step 10-11] Initializing Recommendation Engine & Explainability...")
    recommender = RecommendationEngine(sports_suite, food_suite, library_suite)
    test_context = {
        "day_of_week_num": 4, "is_weekend": 0, "month": 3, "semester_week": 6,
        "classes_ending_near": 3, "is_exam_week": 0, "is_midterm": 0, "is_assignment_deadline": 0,
        "campus_event_flag": 1, "sports_event_flag": 1, "tournament_flag": 0,
        "cultural_major_event_flag": 0, "expected_event_attendance": 100,
        "temperature": 24.5, "rain_flag": 0, "outdoor_suitability": 1.0,
        "sports_active_courts": 5, "sports_prev_crowd": 110, "sports_hist_avg_crowd": 90,
        "sports_hist_peak_crowd": 170, "library_prev_occupancy": 120,
        "library_hist_avg_occupancy": 130, "library_hist_peak_occupancy": 220,
        "food_prev_orders": 85, "food_hist_demand": 110
    }
    sports_rec = recommender.recommend_sports_time(test_context, requested_hour=18)
    print(f"Recommendation test: Requested slot={sports_rec['user_slot_evaluation']['time_window']} (Predicted load: {sports_rec['user_slot_evaluation']['predicted_load_pct']}%)")
    print(f"Best alternative slot: {sports_rec['recommended_lower_demand_slots'][0]['time_window']} ({sports_rec['recommended_lower_demand_slots'][0]['predicted_load_pct']}%)")

    # Step 12: Serialize Artifacts
    print("\n[Step 12] Serializing Model Artifacts and Evaluation Metadata...")
    joblib.dump(sports_suite, os.path.join(MODELS_DIR, "sports_suite.joblib"))
    joblib.dump(food_suite, os.path.join(MODELS_DIR, "food_suite.joblib"))
    joblib.dump(library_suite, os.path.join(MODELS_DIR, "library_suite.joblib"))
    joblib.dump(cluster_suite, os.path.join(MODELS_DIR, "cluster_suite.joblib"))

    # Save summary json for web dashboard
    pipeline_summary = {
        "dataset_metadata": {
            "total_records": len(df),
            "train_records": len(train_df),
            "test_records": len(test_df),
            "source_transparency": "Calibrated Empirical Simulation (Labeled as Simulated per Section 11)",
            "overcrowded_rate_pct": round(float(df["sports_overcrowded_flag"].mean() * 100), 1),
            "avg_campus_pulse_index": round(float(df["campus_pulse_index"].mean()), 2)
        },
        "descriptive_stats": desc_df.to_dict(orient="records"),
        "correlations": correlations,
        "outlier_analysis": {
            "sports": outliers_sports,
            "food": outliers_food
        },
        "hypothesis_tests": hypotheses,
        "temporal_trends": trends,
        "sports_models": {
            "regressors": sports_reg_metrics,
            "classifiers": sports_clf_metrics,
            "best_regressor": sports_suite.best_regressor_name,
            "best_classifier": sports_suite.best_classifier_name,
            "feature_importances": sports_suite.get_feature_importances()
        },
        "food_models": {
            "regressors": food_reg_metrics,
            "classifiers": food_clf_metrics,
            "best_regressor": food_suite.best_regressor_name,
            "best_classifier": food_suite.best_classifier_name,
            "feature_importances": food_suite.get_feature_importances()
        },
        "library_models": {
            "regressors": lib_reg_metrics,
            "classifiers": lib_clf_metrics,
            "best_regressor": library_suite.best_regressor_name,
            "best_classifier": library_suite.best_classifier_name
        },
        "clustering_and_pca": clustering_results,
        "campus_demand_flow": flow_results,
        "cpi_weights": CampusPulseEngine.WEIGHTS
    }

    def sanitize_for_json(obj):
        import math
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(elem) for elem in obj]
        return obj

    clean_summary = sanitize_for_json(pipeline_summary)
    summary_path = os.path.join(MODELS_DIR, "pipeline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(clean_summary, f, indent=2)

    print(f"\nAll artifacts successfully saved to: {MODELS_DIR}")
    print("=" * 70)
    print("CAMPUS IQ PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    return pipeline_summary

if __name__ == "__main__":
    run_full_pipeline()
