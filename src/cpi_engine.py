"""
Campus IQ - Campus Pulse Index (CPI) Engine
Calculates the normalized composite indicator of overall campus resource pressure.
Weights are mathematically and operationally justified based on facility capacities,
queuing bottlenecks, and resource vulnerability.
"""

class CampusPulseEngine:
    """
    Weights:
    - w_sports  = 0.35: Physical sports complex has strict safety/court limits and severe evening bottlenecks.
    - w_library = 0.30: Study spaces represent core academic operations with critical exam spikes.
    - w_food    = 0.25: Food outlets represent high-throughput perishable demand (SDG 12).
    - w_workload= 0.10: Macro academic stressor and class release driving campus-wide foot traffic.
    Total = 1.00
    """
    WEIGHTS = {
        "sports": 0.35,
        "library": 0.30,
        "food": 0.25,
        "workload": 0.10
    }

    @classmethod
    def calculate_cpi(cls, sports_load_pct, library_load_pct, food_demand, academic_workload="Medium", food_capacity_norm=300.0):
        # Normalize food demand to 0-100 scale based on standard operational ceiling (300 orders)
        food_load_pct = min(100.0, max(0.0, (food_demand / food_capacity_norm) * 100.0))
        
        # Workload numerical scale
        workload_map = {"Low": 30.0, "Medium": 60.0, "High": 90.0}
        workload_score = workload_map.get(academic_workload, 60.0)

        # Weighted calculation
        cpi = (
            cls.WEIGHTS["sports"] * sports_load_pct +
            cls.WEIGHTS["library"] * library_load_pct +
            cls.WEIGHTS["food"] * food_load_pct +
            cls.WEIGHTS["workload"] * workload_score
        )
        cpi = round(min(100.0, max(0.0, cpi)), 2)

        if cpi < 45.0:
            status = "Low Activity"
            advisory = "Campus resources are operating well below capacity. Ample space across sports, food, and library facilities."
            badge_color = "success"
        elif cpi < 70.0:
            status = "Moderate Load"
            advisory = "Balanced campus circulation. Standard operational levels with minor queuing possible in food outlets."
            badge_color = "info"
        elif cpi < 85.0:
            status = "High Congestion"
            advisory = "Elevated resource demand. Sports complex and central study zones are experiencing significant congestion."
            badge_color = "warning"
        else:
            status = "Campus Surge"
            advisory = "Critical facility load. Severe overcrowding risk in sports complex and library. Staggered arrivals strongly advised."
            badge_color = "danger"

        return {
            "cpi_score": cpi,
            "status": status,
            "advisory": advisory,
            "badge_color": badge_color,
            "breakdown": {
                "sports_contribution": round(cls.WEIGHTS["sports"] * sports_load_pct, 2),
                "library_contribution": round(cls.WEIGHTS["library"] * library_load_pct, 2),
                "food_contribution": round(cls.WEIGHTS["food"] * food_load_pct, 2),
                "workload_contribution": round(cls.WEIGHTS["workload"] * workload_score, 2),
                "weights": cls.WEIGHTS
            }
        }
