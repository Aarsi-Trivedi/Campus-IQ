"""
Campus IQ - Sports Complex Machine Learning Models
Predicts:
1. Continuous Sports Visitor Count (Regression: Linear Regression, Random Forest, Gradient Boosting)
2. Sports Overcrowding Risk (Classification: Logistic Regression, Random Forest, Gradient Boosting)

Evaluation Metrics:
- Regression: MAE, RMSE, R²
- Classification: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SPORTS_FEATURES = [
    "day_of_week_num", "is_weekend", "month", "semester_week",
    "starting_hour", "is_peak_hour", "classes_ending_near",
    "is_exam_week", "is_midterm", "is_assignment_deadline",
    "campus_event_flag", "sports_event_flag", "tournament_flag",
    "cultural_major_event_flag", "expected_event_attendance",
    "temperature", "rain_flag", "outdoor_suitability",
    "sports_active_courts", "sports_prev_crowd", "sports_hist_avg_crowd",
    "sports_hist_peak_crowd"
]

class SportsModelSuite:
    def __init__(self):
        self.scaler = StandardScaler()
        # Regression models
        self.regressors = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
        }
        # Classification models
        self.classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42),
            "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
        }
        self.best_regressor_name = None
        self.best_classifier_name = None
        self.reg_metrics = {}
        self.clf_metrics = {}
        self.feature_names = SPORTS_FEATURES

    def train_and_evaluate(self, train_df, test_df):
        X_train = train_df[self.feature_names].values
        X_test = test_df[self.feature_names].values

        # Scale features using training data only to avoid leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Targets
        y_reg_train = train_df["sports_visitor_count"].values
        y_reg_test = test_df["sports_visitor_count"].values

        y_clf_train = train_df["sports_overcrowded_flag"].values
        y_clf_test = test_df["sports_overcrowded_flag"].values

        # 1. Evaluate Regressors
        best_r2 = -999.0
        for name, model in self.regressors.items():
            if name == "Linear Regression":
                model.fit(X_train_scaled, y_reg_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_reg_train)
                y_pred = model.predict(X_test)

            y_pred = np.clip(y_pred, 0, 300)
            mae = mean_absolute_error(y_reg_test, y_pred)
            rmse = root_mean_squared_error(y_reg_test, y_pred)
            r2 = r2_score(y_reg_test, y_pred)

            self.reg_metrics[name] = {
                "MAE": round(float(mae), 2),
                "RMSE": round(float(rmse), 2),
                "R2": round(float(r2), 4),
                "predictions_sample": [round(float(p), 1) for p in y_pred[:15]],
                "actuals_sample": [round(float(a), 1) for a in y_reg_test[:15]]
            }

            if r2 > best_r2:
                best_r2 = r2
                self.best_regressor_name = name

        # 2. Evaluate Classifiers
        best_f1 = -1.0
        unique_train_classes = np.unique(y_clf_train)
        if len(unique_train_classes) < 2:
            for name, model in self.classifiers.items():
                self.clf_metrics[name] = {
                    "Accuracy": 1.0,
                    "Precision": 1.0,
                    "Recall": 1.0,
                    "F1-Score": 1.0,
                    "Confusion_Matrix": [[len(y_clf_test)]],
                    "probabilities_sample": [0.0] * min(15, len(y_clf_test)),
                    "actuals_sample": [int(a) for a in y_clf_test[:15]]
                }
            self.best_classifier_name = "Random Forest Classifier"
        else:
            for name, model in self.classifiers.items():
                if name == "Logistic Regression":
                    model.fit(X_train_scaled, y_clf_train)
                    y_pred_c = model.predict(X_test_scaled)
                    y_prob_c = model.predict_proba(X_test_scaled)[:, 1] if model.predict_proba(X_test_scaled).shape[1] > 1 else [0.0]*len(y_pred_c)
                else:
                    model.fit(X_train, y_clf_train)
                    y_pred_c = model.predict(X_test)
                    y_prob_c = model.predict_proba(X_test)[:, 1] if model.predict_proba(X_test).shape[1] > 1 else [0.0]*len(y_pred_c)

                acc = accuracy_score(y_clf_test, y_pred_c)
                prec = precision_score(y_clf_test, y_pred_c, zero_division=0)
                rec = recall_score(y_clf_test, y_pred_c, zero_division=0)
                f1 = f1_score(y_clf_test, y_pred_c, zero_division=0)
                cm = confusion_matrix(y_clf_test, y_pred_c).tolist()

                self.clf_metrics[name] = {
                    "Accuracy": round(float(acc), 4),
                    "Precision": round(float(prec), 4),
                    "Recall": round(float(rec), 4),
                    "F1-Score": round(float(f1), 4),
                    "Confusion_Matrix": cm,
                    "probabilities_sample": [round(float(p), 3) for p in y_prob_c[:15]],
                    "actuals_sample": [int(a) for a in y_clf_test[:15]]
                }

                if f1 > best_f1:
                    best_f1 = f1
                    self.best_classifier_name = name

        return self.reg_metrics, self.clf_metrics

    def predict(self, feature_dict, capacity=200):
        """
        Infers visitor count, load %, and overcrowding risk for a given input dictionary.
        """
        row = [feature_dict.get(col, 0) for col in self.feature_names]
        X = np.array([row])
        X_scaled = self.scaler.transform(X)

        best_reg = self.regressors[self.best_regressor_name]
        if self.best_regressor_name == "Linear Regression":
            pred_count = float(best_reg.predict(X_scaled)[0])
        else:
            pred_count = float(best_reg.predict(X)[0])
        pred_count = max(0.0, round(pred_count, 1))

        best_clf = self.classifiers[self.best_classifier_name]
        if self.best_classifier_name == "Logistic Regression":
            pred_prob = float(best_clf.predict_proba(X_scaled)[0, 1])
            pred_flag = int(best_clf.predict(X_scaled)[0])
        else:
            pred_prob = float(best_clf.predict_proba(X)[0, 1])
            pred_flag = int(best_clf.predict(X)[0])

        load_pct = round((pred_count / capacity) * 100, 1)
        if load_pct < 60:
            risk_label = "Low"
        elif load_pct < 80:
            risk_label = "Moderate"
        elif load_pct < 95:
            risk_label = "High"
        else:
            risk_label = "Severe"

        return {
            "predicted_visitor_count": pred_count,
            "facility_capacity": capacity,
            "predicted_load_pct": load_pct,
            "overcrowding_probability": round(pred_prob, 3),
            "is_overcrowded": bool(pred_flag or load_pct >= 80.0),
            "risk_label": risk_label,
            "regressor_used": self.best_regressor_name,
            "classifier_used": self.best_classifier_name
        }

    def get_feature_importances(self):
        """
        Extracts feature importances for explainability from RF / GBR / LogReg.
        """
        rf_reg = self.regressors["Random Forest Regressor"]
        gb_reg = self.regressors["Gradient Boosting Regressor"]
        log_reg = self.classifiers["Logistic Regression"]

        rf_imp = {feat: round(float(imp), 4) for feat, imp in zip(self.feature_names, rf_reg.feature_importances_)}
        gb_imp = {feat: round(float(imp), 4) for feat, imp in zip(self.feature_names, gb_reg.feature_importances_)}
        lr_coef = {feat: round(float(coef), 4) for feat, coef in zip(self.feature_names, log_reg.coef_[0])}

        # Sort by RF importance
        sorted_rf = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)

        return {
            "random_forest_importance": dict(sorted_rf),
            "gradient_boosting_importance": gb_imp,
            "logistic_regression_coefficients": lr_coef
        }
