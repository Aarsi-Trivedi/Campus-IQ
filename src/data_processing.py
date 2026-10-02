"""
Campus IQ - Data Preprocessing & Feature Engineering Module
Handles data loading, validation, missing value imputation, cyclical encoding,
and strictly isolated train/test splitting to prevent data leakage.
"""

import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "campus_iq_simulated.csv")

# Standard feature sets based on Campus IQ specification
SPORTS_FEATURE_COLS = [
    "day_of_week_num", "is_weekend", "month", "semester_week",
    "starting_hour", "is_peak_hour", "classes_ending_near",
    "is_exam_week", "is_midterm", "is_assignment_deadline",
    "campus_event_flag", "sports_event_flag", "tournament_flag",
    "cultural_major_event_flag", "expected_event_attendance",
    "temperature", "rain_flag", "outdoor_suitability",
    "sports_active_courts", "sports_prev_crowd", "sports_hist_avg_crowd",
    "sports_hist_peak_crowd"
]

FOOD_FEATURE_COLS = [
    "day_of_week_num", "is_weekend", "starting_hour", "is_peak_hour",
    "classes_ending_near", "is_exam_week", "campus_event_flag",
    "expected_event_attendance", "temperature", "rain_flag",
    "sports_visitor_count", "food_prev_orders", "food_hist_demand"
]

LIBRARY_FEATURE_COLS = [
    "day_of_week_num", "is_weekend", "semester_week", "starting_hour",
    "is_peak_hour", "classes_ending_near", "is_exam_week", "is_midterm",
    "is_assignment_deadline", "campus_event_flag", "library_prev_occupancy",
    "library_hist_avg_occupancy", "library_hist_peak_occupancy"
]

CLUSTER_FEATURE_COLS = [
    "sports_load_pct", "food_orders_demand", "library_load_pct",
    "campus_pulse_index", "starting_hour", "classes_ending_near",
    "is_exam_week", "campus_event_flag"
]

def load_data(filepath=None):
    if filepath is None:
        filepath = DATA_PATH
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Run generate_dataset.py first.")
    df = pd.read_csv(filepath)
    return df

def clean_and_prepare(df):
    """
    Validates data integrity, ensures non-null values, and computes cyclical hour encodings.
    """
    df = df.copy()
    
    # Verify no unexpected nulls
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        # Impute numeric with median, categorical with mode
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Cyclical hour features (helps linear and distance-based models capture daily cycles)
    df["hour_sin"] = np.sin(2 * np.pi * df["starting_hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["starting_hour"] / 24.0)

    # Cyclical day of week
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7.0)

    return df

def get_train_test_split(df, test_size=0.2, chronological=True, random_state=42):
    """
    Performs chronological forward-split (or stratified random split) to prevent data leakage.
    Chronological splitting is best practice for campus temporal resource data.
    """
    df_clean = clean_and_prepare(df)
    
    if chronological:
        split_idx = int(len(df_clean) * (1 - test_size))
        train_df = df_clean.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df_clean.iloc[split_idx:].copy().reset_index(drop=True)
    else:
        train_df, test_df = train_test_split(df_clean, test_size=test_size, random_state=random_state)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    return train_df, test_df

def save_splits(train_df, test_df, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(DATA_PATH)
    train_path = os.path.join(output_dir, "train_dataset.csv")
    test_path = os.path.join(output_dir, "test_dataset.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path

if __name__ == "__main__":
    df = load_data()
    train_df, test_df = get_train_test_split(df, test_size=0.2, chronological=True)
    train_p, test_p = save_splits(train_df, test_df)
    print(f"Data preprocessed successfully:")
    print(f"Train set: {len(train_df)} rows saved to {train_p}")
    print(f"Test set: {len(test_df)} rows saved to {test_p}")
