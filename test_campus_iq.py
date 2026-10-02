#!/usr/bin/env python3
"""
Campus IQ - Automated Verification & Test Suite
Validates all pipeline stages, statistical procedures, ML metrics, recommendations, and APIs.
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_processing import load_data, clean_and_prepare, get_train_test_split
from statistical_analysis import compute_descriptive_stats, compute_correlations, detect_outliers_iqr, run_hypothesis_tests, compute_trend_analysis
from models.sports_models import SportsModelSuite
from models.food_models import FoodModelSuite
from models.library_models import LibraryModelSuite
from models.clustering_pca import CampusClusterSuite
from cpi_engine import CampusPulseEngine
from demand_flow import CampusDemandFlowAnalyzer
from recommender import RecommendationEngine
from explainability import CampusExplainability

class TestCampusIQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_data()
        cls.train_df, cls.test_df = get_train_test_split(cls.df, test_size=0.2, chronological=True)

    def test_01_dataset_integrity(self):
        """Verify dataset size, required columns, and transparency labeling."""
        self.assertGreaterEqual(len(self.df), 400)
        self.assertIn("sports_visitor_count", self.df.columns)
        self.assertIn("sports_overcrowded_flag", self.df.columns)
        self.assertIn("food_orders_demand", self.df.columns)
        self.assertIn("library_occupancy", self.df.columns)
        self.assertIn("campus_pulse_index", self.df.columns)
        self.assertIn("data_source_label", self.df.columns)
        # Check transparency label
        self.assertIn(self.df["data_source_label"].iloc[0], ["CALIBRATED_SIMULATION_BENCHMARK", "USER_PREPARED_CAMPUS_DATA", "USER_UPLOADED_OBSERVATION"])

    def test_02_data_splitting_no_leakage(self):
        """Verify train/test chronological split contains no overlap or leakage."""
        self.assertEqual(len(self.train_df) + len(self.test_df), len(self.df))
        self.assertGreater(len(self.train_df), len(self.test_df))
        # Ensure chronological ordering: max date in train <= min date in test
        self.assertLessEqual(self.train_df["date"].max(), self.test_df["date"].min())

    def test_03_statistical_analysis(self):
        """Verify descriptive statistics, correlations, IQR outliers, and hypothesis tests."""
        desc = compute_descriptive_stats(self.df)
        self.assertGreater(len(desc), 5)
        
        # Check IQR outliers for sports crowd
        outliers = detect_outliers_iqr(self.df, "sports_visitor_count")
        self.assertIn("outlier_count", outliers)
        self.assertIn("outliers", outliers)
        
        # Check hypothesis testing
        hyps = run_hypothesis_tests(self.df)
        self.assertEqual(len(hyps), 3)
        for h in hyps:
            self.assertIn("p_value_ttest", h)
            self.assertIn("t_statistic", h)
            self.assertIn("cohens_d", h)
            self.assertIn("conclusion", h)

    def test_04_sports_models(self):
        """Verify Sports complex regression & classification models."""
        suite = SportsModelSuite()
        reg_metrics, clf_metrics = suite.train_and_evaluate(self.train_df, self.test_df)
        
        # Verify 3 candidate regressors evaluated
        self.assertIn("Linear Regression", reg_metrics)
        self.assertIn("Random Forest Regressor", reg_metrics)
        self.assertIn("Gradient Boosting Regressor", reg_metrics)
        self.assertGreater(reg_metrics[suite.best_regressor_name]["R2"], 0.85)

        # Verify 3 candidate classifiers evaluated
        self.assertIn("Logistic Regression", clf_metrics)
        self.assertIn("Random Forest Classifier", clf_metrics)
        self.assertIn("Gradient Boosting Classifier", clf_metrics)
        self.assertGreater(clf_metrics[suite.best_classifier_name]["Accuracy"], 0.85)

        # Verify single inference
        sample_input = {k: self.test_df.iloc[0][k] for k in suite.feature_names}
        pred = suite.predict(sample_input)
        self.assertIn("predicted_visitor_count", pred)
        self.assertIn("predicted_load_pct", pred)
        self.assertIn("risk_label", pred)

    def test_05_food_models_sdg12(self):
        """Verify Food outlet demand planning and waste mitigation models."""
        suite = FoodModelSuite()
        reg_metrics, clf_metrics = suite.train_and_evaluate(self.train_df, self.test_df)
        self.assertGreater(reg_metrics[suite.best_regressor_name]["R2"], 0.85)

        sample_input = {k: self.test_df.iloc[0][k] for k in suite.feature_names}
        pred = suite.predict(sample_input, planned_prep_qty=180)
        self.assertIn("predicted_orders_demand", pred)
        self.assertIn("recommended_preparation_qty", pred)
        self.assertIn("planned_prep_evaluation", pred)

    def test_06_library_models(self):
        """Verify Library occupancy and study space availability models."""
        suite = LibraryModelSuite()
        reg_metrics, clf_metrics = suite.train_and_evaluate(self.train_df, self.test_df)
        self.assertGreater(reg_metrics[suite.best_regressor_name]["R2"], 0.80)

        sample_input = {k: self.test_df.iloc[0][k] for k in suite.feature_names}
        pred = suite.predict(sample_input)
        self.assertIn("predicted_occupancy", pred)
        self.assertIn("study_space_availability", pred)

    def test_07_clustering_and_pca(self):
        """Verify K-Means clustering, silhouette scores, and PCA 2D projections."""
        cluster_suite = CampusClusterSuite()
        res = cluster_suite.fit_and_evaluate_k(self.df, k_min=2, k_max=5)
        self.assertIn("optimal_k", res)
        self.assertIn("cluster_profiles", res)
        self.assertIn("pca_variance_explained", res)
        self.assertEqual(len(res["pca_variance_explained"]), 2)

    def test_08_cpi_engine(self):
        """Verify Campus Pulse Index calculation and bounded score."""
        cpi_res = CampusPulseEngine.calculate_cpi(
            sports_load_pct=85.0,
            library_load_pct=90.0,
            food_demand=220.0,
            academic_workload="High"
        )
        self.assertIn("cpi_score", cpi_res)
        self.assertGreaterEqual(cpi_res["cpi_score"], 0.0)
        self.assertLessEqual(cpi_res["cpi_score"], 100.0)
        self.assertIn(cpi_res["status"], ["Low Activity", "Moderate Load", "High Congestion", "Campus Surge"])

    def test_09_recommender_actionable_slots(self):
        """Verify lower-demand time slot recommendation output."""
        sports_suite = SportsModelSuite()
        sports_suite.train_and_evaluate(self.train_df, self.test_df)
        food_suite = FoodModelSuite()
        food_suite.train_and_evaluate(self.train_df, self.test_df)
        lib_suite = LibraryModelSuite()
        lib_suite.train_and_evaluate(self.train_df, self.test_df)

        rec = RecommendationEngine(sports_suite, food_suite, lib_suite)
        sample_context = {k: self.test_df.iloc[0][k] for k in sports_suite.feature_names}
        
        sports_rec = rec.recommend_sports_time(sample_context, requested_hour=18)
        self.assertIn("recommended_lower_demand_slots", sports_rec)
        self.assertGreaterEqual(len(sports_rec["recommended_lower_demand_slots"]), 1)
        self.assertIn("capacity_reduction", sports_rec["recommended_lower_demand_slots"][0])

    def test_10_explainability(self):
        """Verify instance-level prediction explainability factor attribution."""
        sports_suite = SportsModelSuite()
        sports_suite.train_and_evaluate(self.train_df, self.test_df)
        sample_context = {
            "starting_hour": 18,
            "classes_ending_near": 4,
            "tournament_flag": 1,
            "is_exam_week": 0,
            "rain_flag": 0,
            "sports_prev_crowd": 130
        }
        exp = CampusExplainability.explain_sports_prediction(sample_context, sports_suite)
        self.assertIn("contributing_factors", exp)
        self.assertGreaterEqual(len(exp["contributing_factors"]), 2)
        self.assertIn("top_global_features", exp)

if __name__ == "__main__":
    unittest.main(verbosity=2)
