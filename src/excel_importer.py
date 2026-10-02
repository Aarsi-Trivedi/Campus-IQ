"""
Campus IQ - Excel (.xlsx / .xls) & CSV Data Importer
Handles smart fuzzy column mapping, data validation, missing value imputation,
derived feature calculations (loads, CPI, surplus), and model retraining.
"""

import os
import re
import math
import numpy as np
import pandas as pd
from datetime import datetime

# Alias mappings for flexible user-provided column headers
COLUMN_ALIASES = {
    "date": ["date", "timestamp", "observation_date", "day_date", "dt"],
    "day_of_week": ["day_of_week", "day", "weekday", "day_name"],
    "is_weekend": ["is_weekend", "weekend", "weekend_flag"],
    "time_slot": ["time_slot", "slot", "period", "session", "slot_name"],
    "starting_hour": ["starting_hour", "hour", "start_hour", "time", "hour_of_day"],
    "is_peak_hour": ["is_peak_hour", "peak_hour", "peak", "is_peak"],
    "classes_ending_near": ["classes_ending_near", "classes_ending_nearby", "classes_ending", "classes", "batches", "classes_dismissed"],
    "academic_workload": ["academic_workload", "workload", "academic_load", "workload_level"],
    "semester_stage": ["semester_stage", "semester_phase", "phase", "term_phase"],
    "is_exam_week": ["is_exam_week", "exam_week", "exams", "exam", "is_exam"],
    "is_midterm": ["is_midterm", "midterm", "midterms"],
    "is_assignment_deadline": ["is_assignment_deadline", "deadline", "assignment_deadline"],
    "campus_event_flag": ["campus_event_flag", "campus_event", "event_flag", "event", "has_event"],
    "event_type": ["event_type", "type_of_event", "event_name"],
    "sports_event_flag": ["sports_event_flag", "sports_event", "match", "match_flag"],
    "tournament_flag": ["tournament_flag", "tournament", "tournament_event"],
    "expected_event_attendance": ["expected_event_attendance", "attendance", "event_attendance", "crowd_expected"],
    "temperature": ["temperature", "temp", "temp_c", "degrees_c"],
    "rain_flag": ["rain_flag", "rain", "raining", "is_rain", "is_raining"],
    "weather_condition": ["weather_condition", "weather", "condition"],
    "sports_visitor_count": ["sports_visitor_count", "sports_visitors", "sports_crowd", "sports", "gym_visitors", "gym_crowd", "visitor_count", "visitors"],
    "sports_prev_crowd": ["sports_prev_crowd", "prev_sports_crowd", "sports_lag1", "prev_crowd", "previous_crowd"],
    "sports_capacity": ["sports_capacity", "capacity_sports", "gym_capacity"],
    "sports_active_courts": ["sports_active_courts", "active_courts", "courts", "open_courts"],
    "sports_overcrowded_flag": ["sports_overcrowded_flag", "sports_overcrowded", "overcrowded", "is_overcrowded"],
    "sports_load_pct": ["sports_load_pct", "sports_load", "gym_load"],
    "food_orders_demand": ["food_orders_demand", "food_orders", "food_demand", "orders", "cafeteria_orders", "meals_ordered", "food_count"],
    "food_prepared_qty": ["food_prepared_qty", "food_prepared", "prepared_qty", "prepared", "meals_prepared"],
    "food_prev_orders": ["food_prev_orders", "prev_food_orders", "prev_orders", "food_lag1"],
    "food_risk_category": ["food_risk_category", "food_surplus_shortage_risk", "food_risk", "food_risk_flag"],
    "food_outlet_name": ["food_outlet_name", "outlet_name", "outlet", "canteen_name", "cafe"],
    "library_occupancy": ["library_occupancy", "library_count", "library_crowd", "library_seats", "library", "study_space_occupancy"],
    "library_capacity": ["library_capacity", "capacity_library", "seats_capacity"],
    "library_load_pct": ["library_load_pct", "library_load", "study_load"],
    "library_prev_occupancy": ["library_prev_occupancy", "prev_library_occupancy", "prev_library", "library_lag1"],
    "campus_pulse_index": ["campus_pulse_index", "cpi", "pulse_index", "pulse", "campus_pulse"]
}

def normalize_column_name(col_name):
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', str(col_name).strip().lower())
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean

def import_and_process_file(file_path_or_buffer, default_label="USER_UPLOADED_OBSERVATION"):
    """
    Loads an Excel (.xlsx / .xls) or CSV file, maps user columns, derives missing features,
    and returns a clean, fully-compatible DataFrame.
    """
    if hasattr(file_path_or_buffer, 'read'):
        # In-memory buffer
        try:
            raw_df = pd.read_excel(file_path_or_buffer, engine='openpyxl')
        except Exception:
            file_path_or_buffer.seek(0)
            raw_df = pd.read_csv(file_path_or_buffer)
    else:
        # File path
        if str(file_path_or_buffer).endswith(('.xlsx', '.xls')):
            raw_df = pd.read_excel(file_path_or_buffer, engine='openpyxl')
        else:
            raw_df = pd.read_csv(file_path_or_buffer)

    mapped_df = pd.DataFrame()
    raw_cols = {normalize_column_name(c): c for c in raw_df.columns}

    # Match raw columns to standard fields
    for std_col, aliases in COLUMN_ALIASES.items():
        found = False
        for alias in aliases:
            if alias in raw_cols:
                mapped_df[std_col] = raw_df[raw_cols[alias]]
                found = True
                break
        if not found and std_col in raw_df.columns:
            mapped_df[std_col] = raw_df[std_col]

    n_rows = len(mapped_df)
    if n_rows == 0:
        raise ValueError("Uploaded file contains 0 rows of data.")

    # Apply defaults and derivations for missing columns
    if "date" not in mapped_df.columns:
        mapped_df["date"] = [datetime.now().strftime("%Y-%m-%d")] * n_rows
    else:
        mapped_df["date"] = pd.to_datetime(mapped_df["date"]).dt.strftime("%Y-%m-%d")

    # If starting_hour not provided or time_slot has "HH:MM", parse starting_hour
    if "time_slot" in mapped_df.columns:
        first_val = str(mapped_df["time_slot"].dropna().iloc[0] if len(mapped_df["time_slot"].dropna()) > 0 else "")
        if ":" in first_val or "-" in first_val:
            def extract_hour(ts):
                match = re.search(r'(\d{1,2}):', str(ts))
                return int(match.group(1)) if match else 14
            mapped_df["starting_hour"] = mapped_df["time_slot"].apply(extract_hour)
            # Map to standard slot name
            mapped_df["time_slot"] = mapped_df["starting_hour"].apply(
                lambda h: "Morning" if h < 10 else ("Late Morning" if h < 12 else ("Noon / Lunch" if h < 14 else ("Afternoon" if h < 17 else ("Evening Peak" if h < 20 else "Night"))))
            )

    if "starting_hour" not in mapped_df.columns:
        mapped_df["starting_hour"] = 14
    else:
        mapped_df["starting_hour"] = pd.to_numeric(mapped_df["starting_hour"], errors='coerce').fillna(14).astype(int)

    if "day_of_week" not in mapped_df.columns:
        mapped_df["day_of_week"] = pd.to_datetime(mapped_df["date"]).dt.day_name()
    
    if "day_of_week_num" not in mapped_df.columns:
        mapped_df["day_of_week_num"] = pd.to_datetime(mapped_df["date"]).dt.weekday

    if "is_weekend" not in mapped_df.columns:
        mapped_df["is_weekend"] = mapped_df["day_of_week_num"].apply(lambda x: 1 if x in [5, 6] else 0)

    if "time_slot" not in mapped_df.columns:
        mapped_df["time_slot"] = mapped_df["starting_hour"].apply(
            lambda h: "Morning" if h < 10 else ("Late Morning" if h < 12 else ("Noon / Lunch" if h < 14 else ("Afternoon" if h < 17 else ("Evening Peak" if h < 20 else "Night"))))
        )

    if "is_peak_hour" not in mapped_df.columns:
        mapped_df["is_peak_hour"] = mapped_df["starting_hour"].apply(
            lambda h: 1 if (12 <= h <= 14 or 17 <= h <= 20) else 0
        )

    if "month" not in mapped_df.columns:
        mapped_df["month"] = pd.to_datetime(mapped_df["date"]).dt.month

    # Semester week
    dates = pd.to_datetime(mapped_df["date"])
    min_date = dates.min()
    mapped_df["semester_week"] = ((dates - min_date).dt.days // 7 + 1).clip(1, 16)

    # Semester stage & academic workload
    if "semester_stage" in mapped_df.columns:
        raw_phase = mapped_df["semester_stage"].astype(str).str.lower()
        is_mid_term = (raw_phase.str.contains("mid-term|midterm|exam|finals")).astype(int)
        mapped_df["semester_stage"] = mapped_df["semester_stage"].replace({
            "Regular": "Mid",
            "Mid-Term": "Mid",
            "Fest": "Late"
        })
    else:
        mapped_df["semester_stage"] = "Mid"
        is_mid_term = 0

    if "academic_workload" in mapped_df.columns:
        def map_workload(w):
            try:
                num = float(w)
                if num <= 2: return "Low"
                elif num <= 4: return "Medium"
                else: return "High"
            except (ValueError, TypeError):
                val = str(w).strip().capitalize()
                return val if val in ["Low", "Medium", "High"] else "Medium"
        mapped_df["academic_workload"] = mapped_df["academic_workload"].apply(map_workload)
    else:
        mapped_df["academic_workload"] = "Medium"

    if "classes_ending_near" not in mapped_df.columns:
        mapped_df["classes_ending_near"] = 2
    else:
        mapped_df["classes_ending_near"] = pd.to_numeric(mapped_df["classes_ending_near"], errors='coerce').fillna(2).astype(int)

    if "is_exam_week" not in mapped_df.columns:
        mapped_df["is_exam_week"] = is_mid_term
    if "is_midterm" not in mapped_df.columns:
        mapped_df["is_midterm"] = is_mid_term
    if "is_assignment_deadline" not in mapped_df.columns:
        mapped_df["is_assignment_deadline"] = is_mid_term

    if "event_type" in mapped_df.columns:
        mapped_df["event_type"] = mapped_df["event_type"].fillna("None").astype(str).replace({"nan": "None"})
    else:
        mapped_df["event_type"] = "None"

    is_sports_event = mapped_df["event_type"].str.lower().str.contains("sport|tournament|match").astype(int)
    is_fest = mapped_df["event_type"].str.lower().str.contains("fest|cultural").astype(int)

    if "campus_event_flag" not in mapped_df.columns:
        mapped_df["campus_event_flag"] = (is_sports_event | is_fest).astype(int)
    else:
        mapped_df["campus_event_flag"] = pd.to_numeric(mapped_df["campus_event_flag"], errors='coerce').fillna(0).astype(int)

    if "sports_event_flag" not in mapped_df.columns:
        mapped_df["sports_event_flag"] = is_sports_event
    if "tournament_flag" not in mapped_df.columns:
        mapped_df["tournament_flag"] = is_sports_event
    if "cultural_major_event_flag" not in mapped_df.columns:
        mapped_df["cultural_major_event_flag"] = is_fest
    if "expected_event_attendance" not in mapped_df.columns:
        mapped_df["expected_event_attendance"] = np.where(is_sports_event | is_fest, 80, 0)

    if "temperature" not in mapped_df.columns:
        mapped_df["temperature"] = 24.0
    if "rain_flag" not in mapped_df.columns:
        mapped_df["rain_flag"] = 0
    if "weather_condition" not in mapped_df.columns:
        mapped_df["weather_condition"] = "Clear"
    if "outdoor_suitability" not in mapped_df.columns:
        mapped_df["outdoor_suitability"] = 1.0

    # Sports complex features & derivations
    if "sports_capacity" not in mapped_df.columns:
        mapped_df["sports_capacity"] = 200
    if "sports_visitor_count" not in mapped_df.columns:
        mapped_df["sports_visitor_count"] = 85
    else:
        mapped_df["sports_visitor_count"] = pd.to_numeric(mapped_df["sports_visitor_count"], errors='coerce').fillna(85).astype(int)

    if "sports_prev_crowd" not in mapped_df.columns:
        # Lag 1 shift
        mapped_df["sports_prev_crowd"] = mapped_df["sports_visitor_count"].shift(1).fillna(mapped_df["sports_visitor_count"].iloc[0]).astype(int)
    
    if "sports_active_courts" not in mapped_df.columns:
        mapped_df["sports_active_courts"] = 5
    if "sports_hist_avg_crowd" not in mapped_df.columns:
        mapped_df["sports_hist_avg_crowd"] = mapped_df["sports_visitor_count"].mean().round(1)
    if "sports_hist_peak_crowd" not in mapped_df.columns:
        mapped_df["sports_hist_peak_crowd"] = mapped_df["sports_visitor_count"].max()

    if "sports_load_pct" not in mapped_df.columns or mapped_df["sports_load_pct"].isna().all():
        mapped_df["sports_load_pct"] = ((mapped_df["sports_visitor_count"] / mapped_df["sports_capacity"]) * 100).round(1)
    else:
        mapped_df["sports_load_pct"] = pd.to_numeric(mapped_df["sports_load_pct"], errors='coerce').fillna(((mapped_df["sports_visitor_count"] / mapped_df["sports_capacity"]) * 100).round(1)).round(1)

    if "sports_overcrowded_flag" not in mapped_df.columns or mapped_df["sports_overcrowded_flag"].isna().all():
        mapped_df["sports_overcrowded_flag"] = (mapped_df["sports_load_pct"] >= 80.0).astype(int)
    else:
        mapped_df["sports_overcrowded_flag"] = pd.to_numeric(mapped_df["sports_overcrowded_flag"], errors='coerce').fillna((mapped_df["sports_load_pct"] >= 80.0).astype(int)).astype(int)

    mapped_df["sports_risk_level"] = mapped_df["sports_load_pct"].apply(
        lambda l: "Low" if l < 60 else ("Moderate" if l < 80 else ("High" if l < 95 else "Severe"))
    )

    # Food features & derivations
    if "food_orders_demand" not in mapped_df.columns:
        mapped_df["food_orders_demand"] = 130
    else:
        mapped_df["food_orders_demand"] = pd.to_numeric(mapped_df["food_orders_demand"], errors='coerce').fillna(130).astype(int)

    if "food_outlet_name" not in mapped_df.columns:
        mapped_df["food_outlet_name"] = "Central Cafeteria"
    if "food_category" not in mapped_df.columns:
        mapped_df["food_category"] = "Meals"
    if "food_price_tier" not in mapped_df.columns:
        mapped_df["food_price_tier"] = "Standard"
    if "food_prev_orders" not in mapped_df.columns:
        mapped_df["food_prev_orders"] = mapped_df["food_orders_demand"].shift(1).fillna(mapped_df["food_orders_demand"].iloc[0]).astype(int)
    if "food_hist_demand" not in mapped_df.columns:
        mapped_df["food_hist_demand"] = mapped_df["food_orders_demand"].mean().round(1)

    has_user_risk = "food_risk_category" in mapped_df.columns and mapped_df["food_risk_category"].notna().any()
    if has_user_risk:
        def map_risk(val):
            val_str = str(val).strip().lower()
            if val_str in ["1", "1.0", "true", "shortage", "shortage risk"]:
                return "Shortage Risk"
            elif val_str in ["2", "2.0", "surplus", "surplus risk"]:
                return "Surplus Risk"
            elif val_str in ["0", "0.0", "false", "balanced", "none", "no risk"]:
                return "Balanced"
            return str(val) if val_str not in ["nan", "none", ""] else "Balanced"
        mapped_df["food_risk_category"] = mapped_df["food_risk_category"].apply(map_risk)
        if "food_prepared_qty" not in mapped_df.columns:
            mapped_df["food_prepared_qty"] = mapped_df.apply(
                lambda r: int(round(r["food_orders_demand"] * 0.85)) if r["food_risk_category"] == "Shortage Risk" else int(round(r["food_orders_demand"] * 1.05)),
                axis=1
            )
        mapped_df["food_sold_qty"] = np.minimum(mapped_df["food_prepared_qty"], mapped_df["food_orders_demand"])
        mapped_df["food_surplus_qty"] = np.maximum(0, mapped_df["food_prepared_qty"] - mapped_df["food_sold_qty"])
        mapped_df["food_surplus_pct"] = ((mapped_df["food_surplus_qty"] / np.maximum(1, mapped_df["food_prepared_qty"])) * 100).round(1)
        mapped_df["food_shortage_qty"] = np.maximum(0, mapped_df["food_orders_demand"] - mapped_df["food_prepared_qty"])
    else:
        if "food_prepared_qty" not in mapped_df.columns:
            mapped_df["food_prepared_qty"] = (mapped_df["food_orders_demand"] * 1.05).round().astype(int)
        mapped_df["food_sold_qty"] = np.minimum(mapped_df["food_prepared_qty"], mapped_df["food_orders_demand"])
        mapped_df["food_surplus_qty"] = np.maximum(0, mapped_df["food_prepared_qty"] - mapped_df["food_sold_qty"])
        mapped_df["food_surplus_pct"] = ((mapped_df["food_surplus_qty"] / np.maximum(1, mapped_df["food_prepared_qty"])) * 100).round(1)
        mapped_df["food_shortage_qty"] = np.maximum(0, mapped_df["food_orders_demand"] - mapped_df["food_prepared_qty"])
        mapped_df["food_risk_category"] = mapped_df.apply(
            lambda r: "Surplus Risk" if r["food_surplus_pct"] > 18.0 else ("Shortage Risk" if r["food_shortage_qty"] > 10 else "Balanced"), axis=1
        )

    # Library features & derivations
    if "library_capacity" not in mapped_df.columns:
        mapped_df["library_capacity"] = 350
    if "library_occupancy" not in mapped_df.columns:
        mapped_df["library_occupancy"] = 160
    else:
        mapped_df["library_occupancy"] = pd.to_numeric(mapped_df["library_occupancy"], errors='coerce').fillna(160).astype(int)

    if "library_prev_occupancy" not in mapped_df.columns:
        mapped_df["library_prev_occupancy"] = mapped_df["library_occupancy"].shift(1).fillna(mapped_df["library_occupancy"].iloc[0]).astype(int)
    if "library_hist_avg_occupancy" not in mapped_df.columns:
        mapped_df["library_hist_avg_occupancy"] = mapped_df["library_occupancy"].mean().round(1)
    if "library_hist_peak_occupancy" not in mapped_df.columns:
        mapped_df["library_hist_peak_occupancy"] = mapped_df["library_occupancy"].max()

    if "library_load_pct" not in mapped_df.columns or mapped_df["library_load_pct"].isna().all():
        mapped_df["library_load_pct"] = ((mapped_df["library_occupancy"] / mapped_df["library_capacity"]) * 100).round(1)
    else:
        mapped_df["library_load_pct"] = pd.to_numeric(mapped_df["library_load_pct"], errors='coerce').fillna(((mapped_df["library_occupancy"] / mapped_df["library_capacity"]) * 100).round(1)).round(1)

    mapped_df["library_high_demand_flag"] = (mapped_df["library_load_pct"] >= 80.0).astype(int)
    mapped_df["study_space_availability"] = mapped_df["library_load_pct"].apply(
        lambda l: "Ample" if l < 50 else ("Moderate" if l < 75 else ("Limited" if l < 90 else "Full"))
    )

    # Campus Pulse Index
    if "campus_pulse_index" not in mapped_df.columns or mapped_df["campus_pulse_index"].isna().all():
        food_norm = np.clip((mapped_df["food_orders_demand"] / 300.0) * 100, 0, 100)
        workload_score = mapped_df["academic_workload"].map({"Low": 30.0, "Medium": 60.0, "High": 90.0}).fillna(60.0)
        mapped_df["campus_pulse_index"] = (
            0.35 * mapped_df["sports_load_pct"] +
            0.30 * mapped_df["library_load_pct"] +
            0.25 * food_norm +
            0.10 * workload_score
        ).round(2)
    else:
        mapped_df["campus_pulse_index"] = pd.to_numeric(mapped_df["campus_pulse_index"], errors='coerce').fillna(50.0).round(2)

    mapped_df["campus_activity_status"] = mapped_df["campus_pulse_index"].apply(
        lambda c: "Low Activity" if c < 45 else ("Moderate Load" if c < 70 else ("High Congestion" if c < 85 else "Campus Surge"))
    )

    # Record ID and Source Label
    mapped_df["record_id"] = [f"EXP-{i+1:04d}" for i in range(n_rows)]
    mapped_df["data_source_label"] = default_label

    # Ensure zero NaNs across all columns
    mapped_df = mapped_df.fillna(0)

    return mapped_df
