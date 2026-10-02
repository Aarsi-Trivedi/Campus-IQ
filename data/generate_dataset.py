#!/usr/bin/env python3
"""
Campus IQ - Dataset Generator
Generates a realistic, statistically grounded simulated dataset of campus resource demand.
In strict compliance with Campus IQ Specification Section 11:
'Important: Synthetic or simulated data should not be presented as real campus observations.
If simulation is used for testing, it should be clearly labeled as simulated.'
"""

import math
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_campus_dataset(n_observations=500, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)

    start_date = datetime(2026, 1, 12)  # Semester start (Spring 2026)
    records = []

    time_slots = [
        {"slot": "Morning", "start_hour": 8, "end_hour": 10},
        {"slot": "Late Morning", "start_hour": 10, "end_hour": 12},
        {"slot": "Noon / Lunch", "start_hour": 12, "end_hour": 14},
        {"slot": "Afternoon", "start_hour": 14, "end_hour": 17},
        {"slot": "Evening Peak", "start_hour": 17, "end_hour": 20},
        {"slot": "Night", "start_hour": 20, "end_hour": 22},
    ]

    food_outlets = ["Central Cafeteria", "Sports Complex Kiosk", "Library Coffee Lounge"]
    food_categories = ["Meals", "Snacks", "Beverages"]

    sports_capacity = 200
    library_capacity = 350

    # Lags tracking across timeline
    prev_sports_crowd = 45
    prev_library_occupancy = 110
    prev_food_orders = 70

    current_date = start_date
    slot_idx = 0

    for i in range(n_observations):
        slot_info = time_slots[slot_idx]
        slot_name = slot_info["slot"]
        start_hour = slot_info["start_hour"]

        day_name = current_date.strftime("%A")
        day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
        is_weekend = 1 if day_of_week in [5, 6] else 0
        month = current_date.month

        # Semester timeline (16 weeks)
        days_from_start = (current_date - start_date).days
        semester_week = min(16, max(1, (days_from_start // 7) + 1))

        if semester_week <= 4:
            semester_stage = "Early"
            academic_workload = "Low"
            is_exam_week = 0
            is_midterm = 0
            is_assignment_deadline = 1 if (semester_week == 4 and day_of_week in [3, 4]) else 0
        elif semester_week <= 8:
            semester_stage = "Mid"
            if semester_week == 8:
                academic_workload = "High"
                is_exam_week = 1
                is_midterm = 1
            else:
                academic_workload = "Medium"
                is_exam_week = 0
                is_midterm = 0
            is_assignment_deadline = 1 if day_of_week in [3, 4] else 0
        elif semester_week <= 14:
            semester_stage = "Late"
            academic_workload = "High" if semester_week in [13, 14] else "Medium"
            is_exam_week = 0
            is_midterm = 0
            is_assignment_deadline = 1 if day_of_week in [2, 3, 4] else 0
        else:
            semester_stage = "Finals"
            academic_workload = "High"
            is_exam_week = 1
            is_midterm = 0
            is_assignment_deadline = 0

        # Campus events
        rand_event = random.random()
        campus_event_flag = 0
        event_type = "None"
        sports_event_flag = 0
        tournament_flag = 0
        cultural_major_event_flag = 0
        expected_event_attendance = 0

        if not is_exam_week and rand_event < 0.22:
            campus_event_flag = 1
            ev_roll = random.random()
            if ev_roll < 0.35:
                event_type = "Sports Match"
                sports_event_flag = 1
                expected_event_attendance = int(np.random.normal(120, 25))
            elif ev_roll < 0.55:
                event_type = "Tournament"
                sports_event_flag = 1
                tournament_flag = 1
                expected_event_attendance = int(np.random.normal(220, 40))
            elif ev_roll < 0.80:
                event_type = "Cultural Fest"
                cultural_major_event_flag = 1
                expected_event_attendance = int(np.random.normal(300, 60))
            elif ev_roll < 0.90:
                event_type = "Tech Symposium"
                expected_event_attendance = int(np.random.normal(180, 30))
            else:
                event_type = "Career Fair"
                expected_event_attendance = int(np.random.normal(250, 40))

        # Environmental factors
        base_temp = 22 + 6 * math.sin(2 * math.pi * (month - 3) / 12)
        hour_temp_boost = (start_hour - 8) * 0.8 if start_hour <= 14 else (22 - start_hour) * 0.7
        temperature = round(base_temp + hour_temp_boost + np.random.normal(0, 2), 1)

        rain_prob = 0.12 if month in [1, 2, 3] else 0.28
        rain_flag = 1 if (random.random() < rain_prob and cultural_major_event_flag == 0) else 0
        if rain_flag:
            weather_condition = "Rainy"
            outdoor_suitability = round(max(0.1, 0.3 - np.random.uniform(0, 0.15)), 2)
        elif temperature > 32:
            weather_condition = "Hot"
            outdoor_suitability = 0.55
        elif temperature < 16:
            weather_condition = "Cold"
            outdoor_suitability = 0.65
        elif random.random() < 0.25:
            weather_condition = "Overcast"
            outdoor_suitability = 0.85
        else:
            weather_condition = "Clear"
            outdoor_suitability = 1.0

        # Academic features
        if is_weekend or start_hour >= 20:
            classes_ending_near = 0
        elif start_hour in [10, 12, 16, 17]:
            classes_ending_near = int(np.random.choice([2, 3, 4, 5], p=[0.2, 0.4, 0.3, 0.1]))
        else:
            classes_ending_near = int(np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2]))

        # Peak indicators
        is_sports_peak = 1 if (start_hour in [17, 18, 19] or (is_weekend and start_hour in [10, 16, 17, 18])) else 0
        is_food_peak = 1 if start_hour in [12, 13, 18, 19] else 0
        is_library_peak = 1 if (start_hour in [14, 16, 18, 20] and is_exam_week) or (start_hour in [11, 14, 16] and not is_weekend) else 0
        is_peak_hour = 1 if (is_sports_peak or is_food_peak or is_library_peak) else 0

        # --- Sports Complex Simulation (Core Focus) ---
        sports_base = 35
        if start_hour in [17, 18, 19]:  # Prime evening workout time
            sports_base += 90
        elif start_hour in [6, 8]:     # Early morning
            sports_base += 25
        elif start_hour in [12, 14]:   # Mid-day
            sports_base += 15

        if is_weekend:
            sports_base += 20
        if sports_event_flag:
            sports_base += 45
        if tournament_flag:
            sports_base += 70
        if is_exam_week:
            sports_base -= 30  # Students study during exam week
        if rain_flag:
            sports_base -= 20  # Outdoor turf/courts inaccessible, indoor gym may compress
        if classes_ending_near > 0:
            sports_base += classes_ending_near * 8

        # Autoregressive momentum (previous time slot crowd)
        sports_momentum = 0.25 * prev_sports_crowd
        sports_noise = np.random.normal(0, 8)
        sports_crowd = int(np.clip(sports_base + sports_momentum + sports_noise, 12, 235))

        active_courts = int(np.clip(round(3 + (sports_crowd / sports_capacity) * 5 + np.random.choice([-1, 0, 1])), 2, 8))
        hist_avg_sports = round(sports_base * 0.95 + 10, 1)
        hist_peak_sports = int(min(220, sports_base * 1.35 + 25))

        sports_load_pct = round((sports_crowd / sports_capacity) * 100, 1)
        sports_overcrowded_flag = 1 if sports_load_pct >= 80.0 else 0

        if sports_load_pct < 60:
            sports_risk_level = "Low"
        elif sports_load_pct < 80:
            sports_risk_level = "Moderate"
        elif sports_load_pct < 95:
            sports_risk_level = "High"
        else:
            sports_risk_level = "Severe"

        # --- Food Outlet Simulation ---
        outlet_name = food_outlets[i % len(food_outlets)]
        food_category = food_categories[(i // len(food_outlets)) % len(food_categories)]
        price_range = "Budget" if outlet_name == "Sports Complex Kiosk" else ("Standard" if outlet_name == "Central Cafeteria" else "Premium")

        food_base = 50
        if start_hour in [12, 13]:  # Lunch rush
            food_base += 130
        elif start_hour in [18, 19]:  # Dinner rush
            food_base += 95
        elif start_hour in [10, 16]:  # Snack / Coffee break
            food_base += 40

        # Campus Demand Flow link: Sports complex crowd spills into food outlets!
        food_base += int(sports_crowd * 0.22)
        if classes_ending_near > 0:
            food_base += classes_ending_near * 12
        if campus_event_flag:
            food_base += int(expected_event_attendance * 0.18)
        if is_weekend:
            food_base -= 25

        food_momentum = 0.20 * prev_food_orders
        food_demand = int(np.clip(food_base + food_momentum + np.random.normal(0, 10), 20, 360))
        hist_food_demand = round(food_base * 0.92 + 8, 1)

        # Realistic Food Waste / Surplus modeling: Outlets prepare based on expected normal, sometimes over or under
        prep_bias = np.random.choice([0.88, 0.95, 1.05, 1.15, 1.25], p=[0.15, 0.30, 0.30, 0.15, 0.10])
        food_prepared = int(round(hist_food_demand * prep_bias + np.random.normal(0, 8)))
        food_prepared = max(25, food_prepared)

        food_sold = min(food_prepared, food_demand)
        food_surplus = max(0, food_prepared - food_sold)
        food_surplus_pct = round((food_surplus / max(1, food_prepared)) * 100, 1)
        food_shortage = max(0, food_demand - food_prepared)

        if food_surplus_pct > 18.0:
            food_risk_category = "Surplus Risk"
        elif food_shortage > 15:
            food_risk_category = "Shortage Risk"
        else:
            food_risk_category = "Balanced"

        # --- Library / Study Space Simulation ---
        lib_base = 80
        if is_exam_week:
            lib_base += 160  # Massive exam surge
        elif is_assignment_deadline:
            lib_base += 60
        elif academic_workload == "High":
            lib_base += 45

        if start_hour in [14, 16, 18, 20]:
            lib_base += 70
        elif start_hour in [10, 12]:
            lib_base += 40

        if is_weekend:
            lib_base = lib_base * 0.75 if not is_exam_week else lib_base * 1.1
        if classes_ending_near > 0:
            lib_base += classes_ending_near * 9

        lib_momentum = 0.22 * prev_library_occupancy
        lib_occupancy = int(np.clip(lib_base + lib_momentum + np.random.normal(0, 12), 25, 360))
        lib_load_pct = round((lib_occupancy / library_capacity) * 100, 1)
        lib_hist_avg = round(lib_base * 0.94 + 15, 1)
        lib_hist_peak = int(min(350, lib_base * 1.3 + 20))
        lib_high_demand_flag = 1 if lib_load_pct >= 80.0 else 0

        if lib_load_pct < 50:
            study_space_availability = "Ample"
        elif lib_load_pct < 75:
            study_space_availability = "Moderate"
        elif lib_load_pct < 90:
            study_space_availability = "Limited"
        else:
            study_space_availability = "Full"

        # --- Campus Pulse Index (CPI) Formulation ---
        # Normalized weighted composite index:
        # CPI = 0.35 * SportsLoad + 0.25 * FoodLoad + 0.30 * LibraryLoad + 0.10 * WorkloadScore
        food_load_pct = min(100.0, (food_demand / 300.0) * 100)
        workload_score = 90.0 if academic_workload == "High" else (60.0 if academic_workload == "Medium" else 30.0)
        cpi = round((0.35 * sports_load_pct) + (0.25 * food_load_pct) + (0.30 * lib_load_pct) + (0.10 * workload_score), 2)
        cpi = min(100.0, max(0.0, cpi))

        if cpi < 45.0:
            cpi_status = "Low Activity"
        elif cpi < 70.0:
            cpi_status = "Moderate Load"
        elif cpi < 85.0:
            cpi_status = "High Congestion"
        else:
            cpi_status = "Campus Surge"

        # Construct Record
        rec = {
            "record_id": f"CIQ-{i+1:04d}",
            "data_source_label": "CALIBRATED_SIMULATION_BENCHMARK",
            # Time features
            "date": current_date.strftime("%Y-%m-%d"),
            "day_of_week": day_name,
            "day_of_week_num": day_of_week,
            "is_weekend": is_weekend,
            "month": month,
            "semester_week": semester_week,
            "semester_stage": semester_stage,
            "time_slot": slot_name,
            "starting_hour": start_hour,
            "is_peak_hour": is_peak_hour,
            # Academic features
            "classes_ending_near": classes_ending_near,
            "academic_workload": academic_workload,
            "is_exam_week": is_exam_week,
            "is_midterm": is_midterm,
            "is_assignment_deadline": is_assignment_deadline,
            # Event features
            "campus_event_flag": campus_event_flag,
            "event_type": event_type,
            "sports_event_flag": sports_event_flag,
            "tournament_flag": tournament_flag,
            "cultural_major_event_flag": cultural_major_event_flag,
            "expected_event_attendance": expected_event_attendance,
            # Environmental features
            "temperature": temperature,
            "rain_flag": rain_flag,
            "weather_condition": weather_condition,
            "outdoor_suitability": outdoor_suitability,
            # Sports complex features
            "sports_capacity": sports_capacity,
            "sports_active_courts": active_courts,
            "sports_prev_crowd": prev_sports_crowd,
            "sports_hist_avg_crowd": hist_avg_sports,
            "sports_hist_peak_crowd": hist_peak_sports,
            "sports_visitor_count": sports_crowd,
            "sports_load_pct": sports_load_pct,
            "sports_overcrowded_flag": sports_overcrowded_flag,
            "sports_risk_level": sports_risk_level,
            # Food outlet features
            "food_outlet_name": outlet_name,
            "food_category": food_category,
            "food_price_tier": price_range,
            "food_prev_orders": prev_food_orders,
            "food_hist_demand": hist_food_demand,
            "food_orders_demand": food_demand,
            "food_prepared_qty": food_prepared,
            "food_sold_qty": food_sold,
            "food_surplus_qty": food_surplus,
            "food_surplus_pct": food_surplus_pct,
            "food_shortage_qty": food_shortage,
            "food_risk_category": food_risk_category,
            # Library features
            "library_capacity": library_capacity,
            "library_prev_occupancy": prev_library_occupancy,
            "library_hist_avg_occupancy": lib_hist_avg,
            "library_hist_peak_occupancy": lib_hist_peak,
            "library_occupancy": lib_occupancy,
            "library_load_pct": lib_load_pct,
            "library_high_demand_flag": lib_high_demand_flag,
            "study_space_availability": study_space_availability,
            # Derived campus metrics
            "campus_pulse_index": cpi,
            "campus_activity_status": cpi_status,
        }
        records.append(rec)

        # Update lags for next observation
        prev_sports_crowd = sports_crowd
        prev_food_orders = food_demand
        prev_library_occupancy = lib_occupancy

        slot_idx += 1
        if slot_idx >= len(time_slots):
            slot_idx = 0
            current_date += timedelta(days=1)

    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_campus_dataset(n_observations=500, random_seed=42)
    csv_path = os.path.join(out_dir, "campus_iq_simulated.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} observations successfully at: {csv_path}")
    print(f"Columns ({len(df.columns)}): {list(df.columns[:10])}...")
    print(f"Sports Overcrowding Rate: {(df['sports_overcrowded_flag'].mean()*100):.1f}%")
    print(f"Average CPI: {df['campus_pulse_index'].mean():.2f}")
