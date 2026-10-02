"""
Campus IQ - Action-Oriented Recommendation Engine
Converts predictive ML models into concrete, actionable interventions:
1. Optimal Lower-Demand Time Slot Recommendations for Sports Complex
2. Library & Study Space Off-Peak Routing
3. Food Prep Optimization & Waste Reduction (SDG 12)
4. Facility Management Operations Advisories (SDG 11)
"""

SLOT_SCHEDULE = [
    {"slot": "Morning", "hour": 8, "label": "08:00 - 10:00 AM"},
    {"slot": "Late Morning", "hour": 10, "label": "10:00 - 12:00 PM"},
    {"slot": "Noon / Lunch", "hour": 12, "label": "12:00 - 02:00 PM"},
    {"slot": "Afternoon", "hour": 14, "label": "02:00 - 05:00 PM"},
    {"slot": "Evening Peak", "hour": 17, "label": "05:00 - 08:00 PM"},
    {"slot": "Night", "hour": 20, "label": "08:00 - 10:00 PM"},
]

class RecommendationEngine:
    def __init__(self, sports_suite, food_suite, library_suite):
        self.sports_suite = sports_suite
        self.food_suite = food_suite
        self.library_suite = library_suite

    def recommend_sports_time(self, base_context, requested_hour=18):
        """
        Evaluates all daily time slots for the Sports Complex, identifies overcrowding risk
        at the user's requested hour, and ranks the best alternative lower-demand time slots.
        """
        slot_evaluations = []
        user_slot_pred = None

        for slot_meta in SLOT_SCHEDULE:
            ctx = base_context.copy()
            ctx["starting_hour"] = slot_meta["hour"]
            ctx["is_peak_hour"] = 1 if (slot_meta["hour"] in [17, 18, 19] or (ctx.get("is_weekend", 0) and slot_meta["hour"] in [10, 16, 17])) else 0

            pred = self.sports_suite.predict(ctx)
            entry = {
                "slot_name": slot_meta["slot"],
                "hour": slot_meta["hour"],
                "time_window": slot_meta["label"],
                "predicted_crowd": pred["predicted_visitor_count"],
                "predicted_load_pct": pred["predicted_load_pct"],
                "risk_label": pred["risk_label"],
                "is_overcrowded": pred["is_overcrowded"]
            }
            slot_evaluations.append(entry)

            if abs(slot_meta["hour"] - requested_hour) <= 1:
                user_slot_pred = entry

        if user_slot_pred is None:
            user_slot_pred = slot_evaluations[4]  # Default evening

        # Filter alternatives that have lower demand than the requested slot (or absolute low demand < 70%)
        lower_demand_slots = [s for s in slot_evaluations if s["hour"] != user_slot_pred["hour"] and s["predicted_load_pct"] < user_slot_pred["predicted_load_pct"]]
        if not lower_demand_slots:
            # Fallback to any other slots sorted by lowest load
            lower_demand_slots = [s for s in slot_evaluations if s["hour"] != user_slot_pred["hour"]]
        lower_demand_slots.sort(key=lambda x: x["predicted_load_pct"])

        # Construct actionable recommendations
        if user_slot_pred["predicted_load_pct"] >= 80.0:
            status_summary = "High Overcrowding Warning"
            action_verdict = f"Your requested slot ({user_slot_pred['time_window']}) has a predicted load of {user_slot_pred['predicted_load_pct']}%, exceeding safety & comfort thresholds."
            should_shift = True
        elif user_slot_pred["predicted_load_pct"] >= 65.0:
            status_summary = "Moderate Demand Advisory"
            action_verdict = f"Your requested slot ({user_slot_pred['time_window']}) will experience active traffic ({user_slot_pred['predicted_load_pct']}% load) with potential wait times for popular equipment."
            should_shift = False
        else:
            status_summary = "Optimal Time Confirmed"
            action_verdict = f"Great timing! Your requested slot ({user_slot_pred['time_window']}) is predicted at only {user_slot_pred['predicted_load_pct']}% capacity."
            should_shift = False

        recommendations = []
        for alt in lower_demand_slots[:3]:
            savings = round(user_slot_pred["predicted_load_pct"] - alt["predicted_load_pct"], 1)
            savings_text = f"{savings}% less crowded than requested time" if savings > 0 else "Lowest alternative load available"
            recommendations.append({
                "time_window": alt["time_window"],
                "predicted_load_pct": alt["predicted_load_pct"],
                "capacity_reduction": savings_text,
                "court_availability": "High (courts & gym equipment readily open)" if alt["predicted_load_pct"] < 60 else "Moderate availability",
                "tip": f"Shift to {alt['time_window']} for immediate access and reduced wait times."
            })

        return {
            "requested_hour": requested_hour,
            "user_slot_evaluation": user_slot_pred,
            "status_summary": status_summary,
            "action_verdict": action_verdict,
            "should_shift": should_shift,
            "recommended_lower_demand_slots": recommendations,
            "full_day_schedule": slot_evaluations
        }

    def recommend_study_time(self, base_context, requested_hour=16):
        """
        Recommends optimal low-demand study space windows or satellite study zones.
        """
        slot_evaluations = []
        for slot_meta in SLOT_SCHEDULE:
            ctx = base_context.copy()
            ctx["starting_hour"] = slot_meta["hour"]
            pred = self.library_suite.predict(ctx)
            slot_evaluations.append({
                "slot_name": slot_meta["slot"],
                "time_window": slot_meta["label"],
                "hour": slot_meta["hour"],
                "predicted_occupancy": pred["predicted_occupancy"],
                "predicted_load_pct": pred["predicted_load_pct"],
                "availability": pred["study_space_availability"],
                "is_high_demand": pred["is_high_demand"]
            })

        lower_slots = [s for s in slot_evaluations if s["predicted_load_pct"] < 60.0]
        lower_slots.sort(key=lambda x: x["predicted_load_pct"])

        return {
            "requested_hour": requested_hour,
            "best_study_windows": lower_slots[:2],
            "all_slots": slot_evaluations,
            "satellite_spaces_available": [
                {"name": "Engineering Hub Quiet Lounge", "typical_load": "35%", "distance": "3 min walk"},
                {"name": "Science Block 3rd Floor Pods", "typical_load": "42%", "distance": "5 min walk"}
            ]
        }

    def recommend_food_prep(self, base_context, current_outlet="Central Cafeteria", planned_prep=160):
        """
        Advises food outlet managers on optimal prep batch to prevent waste (SDG 12)
        and avoid running out of meals.
        """
        pred = self.food_suite.predict(base_context, planned_prep_qty=planned_prep)
        exp_demand = pred["predicted_orders_demand"]
        optimal_prep = pred["recommended_preparation_qty"]

        diff = planned_prep - exp_demand
        waste_risk = "High" if diff > 25 else ("Moderate" if diff > 10 else "Low")
        shortage_risk = "High" if diff < -15 else ("Moderate" if diff < -5 else "Low")

        return {
            "outlet": current_outlet,
            "predicted_demand": exp_demand,
            "planned_prep": planned_prep,
            "optimal_recommended_prep": optimal_prep,
            "waste_risk": waste_risk,
            "shortage_risk": shortage_risk,
            "evaluation": pred["planned_prep_evaluation"],
            "sdg_impact": "Aligns with SDG 12: Prevents perishable food dumping and reduces kitchen operating cost."
        }
