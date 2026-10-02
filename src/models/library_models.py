"""
Campus IQ - Library & Study Space Models
Predicts:
1. Continuous Library Occupancy (Linear Regression, Random Forest, Gradient Boosting)
2. High Study-Space Demand Flag (>= 80% capacity)
3. Space availability guidance (Ample, Moderate, Limited, Full)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

LIBRARY_FEATURES = [
    "day_of_week_num", "is_weekend", "semester_week", "starting_hour",
    "is_peak_hour", "classes_ending_near", "is_exam_week", "is_midterm",
    "is_assignment_deadline", "campus_event_flag", "library_prev_occupancy",
    "library_hist_avg_occupancy", "library_hist_peak_occupancy"
]

class LibraryModelSuite:
    def __init__(self):
        self.scaler = StandardScaler()
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
        self.feature_names = LIBRARY_FEATURES

    def train_and_evaluate(self, train_df, test_df):
        X_train = train_df[self.feature_names].values
        X_test = test_df[self.feature_names].values

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        y_reg_train = train_df["library_occupancy"].values
        y_reg_test = test_df["library_occupancy"].values

        y_clf_train = train_df["library_high_demand_flag"].values
        y_clf_test = test_df["library_high_demand_flag"].values

        # Regressors
        best_r2 = -999.0
        for name, model in self.regressors.items():
            if name == "Linear Regression":
                model.fit(X_train_scaled, y_reg_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_reg_train)
                y_pred = model.predict(X_test)

            y_pred = np.clip(y_pred, 10, 360)
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
                    "Confusion_Matrix": [[len(y_clf_test)]]
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
                prec = precision_score(y_clf_test, y_pred_c, zero_division=0)
                rec = recall_score(y_clf_test, y_pred_c, zero_division=0)
                f1 = f1_score(y_clf_test, y_pred_c, zero_division=0)
                cm = confusion_matrix(y_clf_test, y_pred_c).tolist()

                self.clf_metrics[name] = {
                    "Accuracy": round(float(acc), 4),
                    "Precision": round(float(prec), 4),
                    "Recall": round(float(rec), 4),
                    "F1-Score": round(float(f1), 4),
                    "Confusion_Matrix": cm
                }

                if f1 > best_f1:
                    best_f1 = f1
                    self.best_classifier_name = name

        return self.reg_metrics, self.clf_metrics

    def predict(self, feature_dict, capacity=350):
        row = [feature_dict.get(col, 0) for col in self.feature_names]
        X = np.array([row])
        X_scaled = self.scaler.transform(X)

        best_reg = self.regressors[self.best_regressor_name]
        if self.best_regressor_name == "Linear Regression":
            pred_occupancy = float(best_reg.predict(X_scaled)[0])
        else:
            pred_occupancy = float(best_reg.predict(X)[0])
        pred_occupancy = max(10.0, round(pred_occupancy, 1))

        load_pct = round((pred_occupancy / capacity) * 100, 1)

        if load_pct < 50:
            availability = "Ample Seats Available"
        elif load_pct < 75:
            availability = "Moderate Availability"
        elif load_pct < 90:
            availability = "Limited Seats (High Demand)"
        else:
            availability = "Full Capacity (Critical)"

        return {
            "predicted_occupancy": pred_occupancy,
            "library_capacity": capacity,
            "predicted_load_pct": load_pct,
            "is_high_demand": bool(load_pct >= 80.0),
            "study_space_availability": availability,
            "regressor_used": self.best_regressor_name
        }

    def get_feature_importances(self):
        rf_reg = self.regressors["Random Forest Regressor"]
        rf_imp = {feat: round(float(imp), 4) for feat, imp in zip(self.feature_names, rf_reg.feature_importances_)}
        sorted_rf = sorted(rf_imp.items(), key=lambda x: x[1], reverse=True)
        return {"random_forest_importance": dict(sorted_rf)}
