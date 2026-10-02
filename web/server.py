#!/usr/bin/env python3
"""
Campus IQ - Interactive Web Server and REST API
Provides backend inference, recommendation, and statistical analysis endpoints.
Uses Python's standard library http.server for maximum reliability and portability.
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import joblib
import pandas as pd
import numpy as np
import base64
import io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models_saved")
WEB_DIR = os.path.join(BASE_DIR, "web")
DATA_DIR = os.path.join(BASE_DIR, "data")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from recommender import RecommendationEngine
from explainability import CampusExplainability
from cpi_engine import CampusPulseEngine
from demand_flow import CampusDemandFlowAnalyzer
from excel_importer import import_and_process_file
from run_pipeline import run_full_pipeline

# Global model suites and summary
sports_suite = None
food_suite = None
library_suite = None
cluster_suite = None
recommender = None
pipeline_summary = {}

def load_system_artifacts():
    global sports_suite, food_suite, library_suite, cluster_suite, recommender, pipeline_summary
    print("[Server] Loading trained models and pipeline metadata...")
    summary_path = os.path.join(MODELS_DIR, "pipeline_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            pipeline_summary = json.load(f)

    sports_suite = joblib.load(os.path.join(MODELS_DIR, "sports_suite.joblib"))
    food_suite = joblib.load(os.path.join(MODELS_DIR, "food_suite.joblib"))
    library_suite = joblib.load(os.path.join(MODELS_DIR, "library_suite.joblib"))
    cluster_suite = joblib.load(os.path.join(MODELS_DIR, "cluster_suite.joblib"))
    recommender = RecommendationEngine(sports_suite, food_suite, library_suite)
    print("[Server] All ML suites loaded successfully!")

def sanitize_dict_for_json(obj):
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_dict_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_dict_for_json(elem) for elem in obj]
    return obj

class CampusIQRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, data, status=200):
        clean_data = sanitize_dict_for_json(data)
        response_bytes = json.dumps(clean_data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/summary":
            self._send_json(pipeline_summary)
            return

        elif path == "/api/clusters":
            if cluster_suite:
                self._send_json({
                    "optimal_k": cluster_suite.optimal_k,
                    "k_evaluation": cluster_suite.k_range_eval,
                    "cluster_profiles": cluster_suite.cluster_profiles,
                    "pca_variance_explained": cluster_suite.pca_variance_explained,
                    "pca_points": pipeline_summary.get("clustering_and_pca", {}).get("pca_points", [])
                })
            else:
                self._send_json({"error": "Clustering suite not loaded"}, 500)
            return

        elif path == "/api/flow":
            self._send_json(pipeline_summary.get("campus_demand_flow", {}))
            return

        elif path == "/api/stats":
            self._send_json({
                "descriptive": pipeline_summary.get("descriptive_stats", []),
                "hypotheses": pipeline_summary.get("hypothesis_tests", []),
                "correlations": pipeline_summary.get("correlations", {}),
                "outliers": pipeline_summary.get("outlier_analysis", {}),
                "trends": pipeline_summary.get("temporal_trends", {})
            })
            return

        elif path == "/api/models":
            self._send_json({
                "sports": pipeline_summary.get("sports_models", {}),
                "food": pipeline_summary.get("food_models", {}),
                "library": pipeline_summary.get("library_models", {})
            })
            return

        elif path == "/api/dataset":
            # Return sample records from dataset
            csv_path = os.path.join(DATA_DIR, "campus_iq_simulated.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                sample = df.tail(60).to_dict(orient="records")
                self._send_json({
                    "total_records": len(df),
                    "columns": list(df.columns),
                    "records": sample,
                    "transparency_note": "Campus IQ Calibrated Simulation Benchmark (Simulated Dataset per Section 11)"
                })
            else:
                self._send_json({"error": "Dataset not found"}, 404)
            return

        # Default: serve static files from WEB_DIR
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if path == "/api/predict/sports":
            # Predict sports visitor count & overcrowding
            hour = int(payload.get("starting_hour", 18))
            capacity = int(payload.get("sports_capacity", 200))
            pred = sports_suite.predict(payload, capacity=capacity)
            explanation = CampusExplainability.explain_sports_prediction(payload, sports_suite)
            rec = recommender.recommend_sports_time(payload, requested_hour=hour)
            
            self._send_json({
                "prediction": pred,
                "explanation": explanation,
                "recommendation": rec
            })
            return

        elif path == "/api/predict/food":
            # Predict food demand & surplus/shortage risk (SDG 12)
            planned_prep = int(payload.get("planned_prep_qty", 150))
            pred = food_suite.predict(payload, planned_prep_qty=planned_prep)
            prep_rec = recommender.recommend_food_prep(payload, current_outlet=payload.get("food_outlet_name", "Central Cafeteria"), planned_prep=planned_prep)

            self._send_json({
                "prediction": pred,
                "recommendation": prep_rec
            })
            return

        elif path == "/api/predict/library":
            capacity = int(payload.get("library_capacity", 350))
            pred = library_suite.predict(payload, capacity=capacity)
            hour = int(payload.get("starting_hour", 14))
            rec = recommender.recommend_study_time(payload, requested_hour=hour)

            self._send_json({
                "prediction": pred,
                "recommendation": rec
            })
            return

        elif path == "/api/pulse":
            sports_load = float(payload.get("sports_load_pct", 55.0))
            library_load = float(payload.get("library_load_pct", 60.0))
            food_demand = float(payload.get("food_orders_demand", 140.0))
            workload = payload.get("academic_workload", "Medium")

            cpi_res = CampusPulseEngine.calculate_cpi(sports_load, library_load, food_demand, workload)
            self._send_json(cpi_res)
            return

        elif path == "/api/recommend":
            hour = int(payload.get("starting_hour", 18))
            facility = payload.get("facility", "sports")

            if facility == "sports":
                rec = recommender.recommend_sports_time(payload, requested_hour=hour)
            elif facility == "library":
                rec = recommender.recommend_study_time(payload, requested_hour=hour)
            else:
                rec = recommender.recommend_food_prep(payload, planned_prep=int(payload.get("planned_prep_qty", 150)))

            self._send_json(rec)
            return

        elif path == "/api/data/upload_xlsx":
            try:
                b64_content = payload.get("file_content_base64", "")
                mode = payload.get("mode", "replace")
                filename = payload.get("filename", "uploaded_campus_data.xlsx")

                if "," in b64_content:
                    b64_content = b64_content.split(",", 1)[1]

                file_bytes = base64.b64decode(b64_content)
                bio = io.BytesIO(file_bytes)

                print(f"[Server] Received Excel file upload ({len(file_bytes)} bytes): {filename}")
                new_df = import_and_process_file(bio, default_label="USER_UPLOADED_EXCEL_DATA")

                csv_path = os.path.join(DATA_DIR, "campus_iq_simulated.csv")
                if mode == "append" and os.path.exists(csv_path):
                    existing_df = pd.read_csv(csv_path)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df["record_id"] = [f"CIQ-{i+1:04d}" for i in range(len(combined_df))]
                    combined_df.to_csv(csv_path, index=False)
                    final_count = len(combined_df)
                else:
                    new_df["record_id"] = [f"CIQ-{i+1:04d}" for i in range(len(new_df))]
                    new_df.to_csv(csv_path, index=False)
                    final_count = len(new_df)

                print(f"[Server] Retraining pipeline on newly uploaded {len(new_df)} observations...")
                run_full_pipeline()
                load_system_artifacts()

                self._send_json({
                    "success": True,
                    "filename": filename,
                    "rows_imported": len(new_df),
                    "total_records": final_count,
                    "message": f"Successfully imported {len(new_df)} rows and retrained all classical ML models on your Excel data!"
                })
            except Exception as e:
                print(f"[Server] Error processing uploaded Excel file: {e}")
                self._send_json({"error": str(e)}, 400)
            return

        elif path == "/api/data/append":
            # Add user observation to dataset
            csv_path = os.path.join(DATA_DIR, "campus_iq_simulated.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                new_row = payload.copy()
                new_row["record_id"] = f"CIQ-{len(df)+1:04d}"
                new_row["data_source_label"] = "USER_SUBMITTED_CAMPUS_OBSERVATION"
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv(csv_path, index=False)
                self._send_json({"success": True, "new_total_records": len(df), "record_id": new_row["record_id"]})
            else:
                self._send_json({"error": "Dataset not found"}, 404)
            return

        self._send_json({"error": "Endpoint not found"}, 404)

def run_server(port=8090):
    load_system_artifacts()
    server_address = ("", port)
    httpd = HTTPServer(server_address, CampusIQRequestHandler)
    print(f"\n=======================================================")
    print(f" Campus IQ Web Server running at: http://localhost:{port}")
    print(f" Open http://localhost:{port} in your browser to view dashboard")
    print(f"=======================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
        httpd.server_close()

if __name__ == "__main__":
    port = 8090
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
