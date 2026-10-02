"""
Campus IQ - Campus Activity Pattern Discovery (K-Means Clustering + PCA)
Discovers systemic, data-driven campus operational regimes across sports, food, and library spaces.
Evaluates Silhouette Scores across K=2..7, performs PCA 2D/3D projection,
and provides qualitative interpretations of cluster personas.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

CLUSTER_FEATURES = [
    "sports_load_pct", "food_orders_demand", "library_load_pct",
    "campus_pulse_index", "starting_hour", "classes_ending_near",
    "is_exam_week", "campus_event_flag", "temperature"
]

class CampusClusterSuite:
    def __init__(self, n_clusters=4):
        self.scaler = StandardScaler()
        self.pca_2d = PCA(n_components=2, random_state=42)
        self.pca_3d = PCA(n_components=3, random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.feature_names = CLUSTER_FEATURES
        self.k_range_eval = {}
        self.optimal_k = n_clusters
        self.cluster_profiles = {}
        self.pca_variance_explained = []

    def fit_and_evaluate_k(self, df, k_min=2, k_max=7):
        X = df[self.feature_names].values
        X_scaled = self.scaler.fit_transform(X)

        best_k = k_min
        best_sil = -1.0
        evaluation_results = []

        for k in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            sil = float(silhouette_score(X_scaled, labels))
            ch = float(calinski_harabasz_score(X_scaled, labels))
            inertia = float(km.inertia_)

            evaluation_results.append({
                "k": k,
                "silhouette_score": round(sil, 4),
                "calinski_harabasz_score": round(ch, 2),
                "inertia": round(inertia, 2)
            })

            if sil > best_sil:
                best_sil = sil
                best_k = k

        self.k_range_eval = evaluation_results
        self.optimal_k = best_k

        # Re-fit with optimal K
        self.kmeans = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        final_labels = self.kmeans.fit_predict(X_scaled)

        # Fit PCA
        pca_coords_2d = self.pca_2d.fit_transform(X_scaled)
        pca_coords_3d = self.pca_3d.fit_transform(X_scaled)
        self.pca_variance_explained = [round(float(v), 4) for v in self.pca_2d.explained_variance_ratio_]

        # Generate qualitative cluster interpretations
        df_clustered = df.copy()
        df_clustered["cluster"] = final_labels

        profiles = {}
        for c in range(self.optimal_k):
            c_data = df_clustered[df_clustered["cluster"] == c]
            avg_sports = float(c_data["sports_load_pct"].mean())
            avg_food = float(c_data["food_orders_demand"].mean())
            avg_lib = float(c_data["library_load_pct"].mean())
            avg_cpi = float(c_data["campus_pulse_index"].mean())
            exam_ratio = float(c_data["is_exam_week"].mean())
            event_ratio = float(c_data["campus_event_flag"].mean())
            common_slot = c_data["time_slot"].mode()[0] if not c_data.empty else "N/A"

            # Qualitative Persona Tagging
            if exam_ratio > 0.6:
                name = "Academic Intensive / Exam Pressure"
                desc = "Characterized by peak library occupancy (~85-98%), subdued sports activity, and moderate food demand during study breaks."
                recommendation = "Expand quiet study zones; extend library overnight hours; decrease sports staffing."
            elif avg_sports > 75:
                name = "Sports Complex Prime & Rush"
                desc = "Characterized by high sports load (>75%), evening activity concentration, and subsequent dinner rush at adjacent food kiosks."
                recommendation = "Activate court reservation systems; pre-stage evening food kiosk meal combos; alert gym attendants."
            elif avg_food > 180 or event_ratio > 0.4:
                name = "Campus Event & Dining Surge"
                desc = "Triggered by campus cultural fests, symposiums, or tournaments, leading to heightened food outlet demand and broad campus circulation."
                recommendation = "Increase food prep by 20% to avoid stockouts; deploy directional crowd signage."
            else:
                name = "Off-Peak / Balanced Circulation"
                desc = "Steady, low-to-moderate resource utilization across all facilities with ample seating and available courts."
                recommendation = "Ideal window for maintenance, facility deep-cleaning, and general walk-in visits."

            profiles[c] = {
                "cluster_id": c,
                "label": name,
                "description": desc,
                "recommended_action": recommendation,
                "sample_count": len(c_data),
                "percentage_of_data": round((len(c_data) / len(df)) * 100, 1),
                "avg_sports_load_pct": round(avg_sports, 1),
                "avg_food_orders": round(avg_food, 1),
                "avg_library_load_pct": round(avg_lib, 1),
                "avg_campus_pulse_index": round(avg_cpi, 1),
                "predominant_time_slot": common_slot
            }

        self.cluster_profiles = profiles

        # Package data points with PCA 2D coordinates for interactive scatter plotting
        pca_points = []
        for idx, row in df_clustered.iterrows():
            pca_points.append({
                "record_id": row.get("record_id", f"CIQ-{idx+1:04d}"),
                "pca_x": round(float(pca_coords_2d[idx, 0]), 3),
                "pca_y": round(float(pca_coords_2d[idx, 1]), 3),
                "cluster": int(final_labels[idx]),
                "cluster_name": profiles[int(final_labels[idx])]["label"],
                "sports_load": round(float(row["sports_load_pct"]), 1),
                "library_load": round(float(row["library_load_pct"]), 1),
                "food_orders": round(float(row["food_orders_demand"]), 1),
                "cpi": round(float(row["campus_pulse_index"]), 1),
                "time_slot": row.get("time_slot", "N/A"),
                "is_exam_week": int(row.get("is_exam_week", 0))
            })

        return {
            "optimal_k": self.optimal_k,
            "k_evaluation": self.k_range_eval,
            "pca_variance_explained": self.pca_variance_explained,
            "cluster_profiles": self.cluster_profiles,
            "pca_points": pca_points[:200]  # Representative subset for high-speed dashboard visualization
        }

    def predict_cluster(self, feature_dict):
        row = [feature_dict.get(col, 0) for col in self.feature_names]
        X = np.array([row])
        X_scaled = self.scaler.transform(X)
        cluster_id = int(self.kmeans.predict(X_scaled)[0])
        pca_coords = self.pca_2d.transform(X_scaled)[0]

        profile = self.cluster_profiles.get(cluster_id, {})
        return {
            "assigned_cluster": cluster_id,
            "cluster_label": profile.get("label", "General Activity"),
            "cluster_description": profile.get("description", ""),
            "operational_recommendation": profile.get("recommended_action", ""),
            "pca_coordinates": [round(float(pca_coords[0]), 3), round(float(pca_coords[1]), 3)]
        }
