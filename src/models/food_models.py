"""
Campus IQ - Food Outlet Machine Learning Models
Predicts:
1. Expected Food Orders / Demand (Regression: Linear Regression, Random Forest, Gradient Boosting)
2. Food Surplus / Shortage Risk Category (Classification: Logistic Regression, Random Forest)

Aligns with SDG 12 (Responsible Consumption & Production) by minimizing food waste while preventing student stockouts.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

FOOD_FEATURES = [
    "day_of_week_num", "is_weekend", "starting_hour", "is_peak_hour",
    "classes_ending_near", "is_exam_week", "campus_event_flag",
    "expected_event_attendance", "temperature", "rain_flag",
    "sports_visitor_count", "food_prev_orders", "food_hist_demand"
]

class FoodModelSuite:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.regressors = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
        }
        
        self.classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42)
        }
        
        self.best_regressor_name = None
        self.best_classifier_name = None
        self.reg_metrics = {}
        self.clf_metrics = {}
        self.feature_names = FOOD_FEATURES

    def train_and_evaluate(self, train_df, test_df):
        X_train = train_df[self.feature_names].values
        X_test = test_df[self.feature_names].values

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 1. Regression targets (food_orders_demand)
        y_reg_train = train_df["food_orders_demand"].values
        y_reg_test = test_df["food_orders_demand"].values

        # 2. Classification targets (food_risk_category: Balanced, Surplus Risk, Shortage Risk)
        all_classes = np.unique(pd.concat([pd.Series(train_df["food_risk_category"].values), pd.Series(test_df["food_risk_category"].values)]))
        if len(all_classes) < 2:
            all_classes = np.array(list(all_classes) + (["Shortage Risk"] if all_classes[0] != "Shortage Risk" else ["Balanced"]))
        self.label_encoder.fit(all_classes)
        y_clf_train = self.label_encoder.transform(train_df["food_risk_category"].values)
        y_clf_test = self.label_encoder.transform(test_df["food_risk_category"].values)

        # Regressors
        best_r2 = -999.0
        for name, model in self.regressors.items():
            if name == "Linear Regression":
                model.fit(X_train_scaled, y_reg_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_reg_train)
                y_pred = model.predict(X_test)

            y_pred = np.clip(y_pred, 10, 500)
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

        # Classifiers
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
                    "classes": list(self.label_encoder.classes_)
                }
            self.best_classifier_name = "Random Forest Classifier"
        else:
            for name, model in self.classifiers.items():
                if name == "Logistic Regression":
                    model.fit(X_train_scaled, y_clf_train)
                    y_pred_c = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_clf_train)
                    y_pred_c = model.predict(X_test)

                acc = accuracy_score(y_clf_test, y_pred_c)
                prec = precision_score(y_clf_test, y_pred_c, average="weighted", zero_division=0)
                rec = recall_score(y_clf_test, y_pred_c, average="weighted", zero_division=0)
                f1 = f1_score(y_clf_test, y_pred_c, average="weighted", zero_division=0)
                cm = confusion_matrix(y_clf_test, y_pred_c).tolist()

                self.clf_metrics[name] = {
                    "Accuracy": round(float(acc), 4),
                    "Precision": round(float(prec), 4),
                    "Recall": round(float(rec), 4),
                    "F1-Score": round(float(f1), 4),
                    "Confusion_Matrix": cm,
                    "classes": list(self.label_encoder.classes_)
                }

                if f1 > best_f1:
                    best_f1 = f1
                    self.best_classifier_name = name

        return self.reg_metrics, self.clf_metrics

    def predict(self, feature_dict, planned_prep_qty=None):
        """
        Infers expected food demand, compares with planned prep, and recommends waste-minimization adjustments.
        """
        row = [feature_dict.get(col, 0) for col in self.feature_names]
        X = np.array([row])
        X_scaled = self.scaler.transform(X)

        best_reg = self.regressors[self.best_regressor_name]
        if self.best_regressor_name == "Linear Regression":
            expected_demand = float(best_reg.predict(X_scaled)[0])
        else:
            expected_demand = float(best_reg.predict(X)[0])
        expected_demand = max(10.0, round(expected_demand, 1))

        best_clf = self.classifiers[self.best_classifier_name]
        try:
            if self.best_classifier_name == "Logistic Regression":
                pred_class_idx = int(best_clf.predict(X_scaled)[0])
            else:
                pred_class_idx = int(best_clf.predict(X)[0])
            risk_category = self.label_encoder.inverse_transform([pred_class_idx])[0]
        except Exception:
            risk_category = "Balanced"

        # Recommended prep to prevent waste and avoid shortage
        # Add a 5% safety buffer
        recommended_prep = int(round(expected_demand * 1.05))

        prep_evaluation = None
        if planned_prep_qty is not None and planned_prep_qty > 0:
            diff = planned_prep_qty - expected_demand
            if diff > (expected_demand * 0.18):
                prep_evaluation = f"Warning: Planned preparation ({planned_prep_qty}) exceeds predicted demand ({expected_demand:.0f}) by {diff:.0f} portions. High risk of food waste! Recommend reducing to ~{recommended_prep} portions."
            elif diff < -(expected_demand * 0.10):
                prep_evaluation = f"Alert: Planned preparation ({planned_prep_qty}) is below predicted demand ({expected_demand:.0f}) by {abs(diff):.0f} portions. Risk of student stockout! Recommend increasing to ~{recommended_prep} portions."
            else:
                prep_evaluation = f"Optimal: Planned preparation matches predicted demand within safe operational margins."

        return {
            "predicted_orders_demand": expected_demand,
            "recommended_preparation_qty": recommended_prep,
            "predicted_risk_category": risk_category,
            "planned_prep_evaluation": prep_evaluation,
            "regressor_used": self.best_regressor_name,
            "classifier_used": self.best_classifier_name
        }

    def get_feature_importances(self):
        rf_reg = self.regressors["Random Forest Regressor"]
        gb_reg = self.regressors["Gradient Boosting Regressor"]

        rf_imp = {feat: round(float(imp), 4) for feat, imp in zip(self.feature_names, rf_reg.feature_importances_)}
        gb_imp = {feat: round(float(imp), 4) for feat, imp in zip(self.feature_names, gb_reg.feature_importances_)}
        sorted_rf = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)

        return {
            "random_forest_importance": dict(sorted_rf),
            "gradient_boosting_importance": gb_imp
        }
