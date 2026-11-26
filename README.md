# Campus IQ: Interpretable Statistical Machine Learning for Smart Campus Resource Intelligence

> **Statistical Machine Learning (CSET-211) — Capstone Project Report**  
> **School of Computer Science Engineering & Technology (SCSET), Bennett University (The Times Group)**  
> **Academic Session:** July–Dec 2026  
> **Team Members:**  
> • **Aarsi Trivedi** (Lead) — `S25CSEU1882`  
> • **Arju** — `S25CSEU1868`  
> • **Angel** — `S25CSEU1873`  
> **UN SDG Alignment:** SDG 11 (*Sustainable Cities and Communities*) & SDG 12 (*Responsible Consumption and Production*)  
> **Live Web Application:** `python web/server.py` &rarr; [http://localhost:8090](http://localhost:8090)

---

## 1. Executive Summary

**Campus IQ** is an end-to-end, interpretable statistical machine learning framework engineered to monitor, predict, and balance resource demand across major academic infrastructure:
1. **Sports Complex:** Forecasts hourly visitor headcount, predicts overcrowding risk ($\ge 80\%$ capacity), and recommends lower-demand alternate time slots.
2. **Food Outlets / Cafeteria (SDG 12):** Predicts meal order volumes, quantifies perishable food surplus/shortage risks, and curbs dining waste.
3. **Library Study Spaces (SDG 11):** Estimates study desk occupancy and flags peak study crunches during examinations.
4. **Campus Pulse Index (CPI):** A normalized composite index ($[0, 100]$) providing real-time campus-wide operational visibility.
5. **Cross-Facility Demand Flow:** A Markovian transition network capturing cascading footfall between facilities.

In strict adherence to statistical machine learning pedagogy, the system employs **transparent, interpretable classical models** (Multi-Output Random Forest, Gradient Boosting, Ridge Regression, Logistic Regression, K-Means Clustering, and PCA) rather than opaque deep neural networks.

---

## 2. System Architecture

```
                                    ┌────────────────────────────────────────────────────────┐
                                    │     490 Multi-Facility Observations (Empirical)        │
                                    └───────────────────────────┬────────────────────────────┘
                                                                │
                                                                ▼
                                    ┌────────────────────────────────────────────────────────┐
                                    │    Temporal & Chronological Preprocessing (No Leak)    │
                                    │             (80% Train: 392  |  20% Test: 98)          │
                                    └───────────────────────────┬────────────────────────────┘
                                                                │
                                ┌───────────────────────────────┼───────────────────────────────┐
                                ▼                               ▼                               ▼
                 ┌─────────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────────┐
                 │     Sports Complex ML       │ │    Food Outlets ML (SDG 12) │ │    Library Spaces ML (SDG 11)│
                 │  • Multi-Output RF Reg.     │ │  • Meal Demand Predictor    │ │  • Desk Occupancy Forecast  │
                 │  • Overcrowding Classifier  │ │  • Surplus / Stockout Risk  │ │  • Exam Crunch Detector     │
                 │  • R² = 0.929, Recall=88.9% │ │  • R² = 0.894               │ │  • R² = 0.912               │
                 └──────────────┬──────────────┘ └──────────────┬──────────────┘ └──────────────┬──────────────┘
                                │                               │                               │
                                └───────────────────────────────┼───────────────────────────────┘
                                                                │
                                                                ▼
                                    ┌────────────────────────────────────────────────────────┐
                                    │     Unsupervised Regimes & Demand Ripple Network       │
                                    │       • K-Means (K=3, Silhouette=0.448) & 2D PCA       │
                                    │       • Markov Cross-Facility Transition Matrices      │
                                    └───────────────────────────┬────────────────────────────┘
                                                                │
                                                                ▼
                                    ┌────────────────────────────────────────────────────────┐
                                    │   Campus Pulse Index (CPI) & Recommender Engine        │
                                    │       • Real-Time Normalized Score [0, 100]            │
                                    │       • Actionable Low-Congestion Slot Dispatcher      │
                                    └───────────────────────────┬────────────────────────────┘
                                                                │
                                                                ▼
                                    ┌────────────────────────────────────────────────────────┐
                                    │    Clean, Accessible Web Dashboard (Pure JS/HTML5)     │
                                    │          http://localhost:8090 (Zero Auth Gate)        │
                                    └────────────────────────────────────────────────────────┘
```

---

## 3. Quick Start & Execution

### Prerequisites
Clone the repository and install the standard dependencies:
```bash
git clone https://github.com/Aarsi-Trivedi/Campus-IQ.git
cd Campus-IQ
pip install -r requirements.txt
```

### Launch the Production Web Dashboard
Run the built-in HTTP server (with automatic port failover and address reuse):
```bash
python web/server.py
```
Open **[http://localhost:8090](http://localhost:8090)** (or the port displayed in terminal) in any modern browser.

### Run Automated Unit & Regression Tests
Verify that all 10 pipeline modules, models, split integrity, and CPI calculations pass without error:
```bash
python test_campus_iq.py
```

### Re-Train or Benchmark All Models
To run end-to-end retraining and output statistical evaluation metrics:
```bash
python run_pipeline.py
python evaluate_all.py
```

---

## 4. Empirical Datasets & Statistical Rigor

### Dataset Summary (`data/user_uploaded_campus_data.csv`)
- **Total Records:** 490 structured campus observations across weekdays, weekends, regular weeks, and exam cycles.
- **Split Strategy:** Strict chronological 80/20 train/test split ($N_{\text{train}} = 392$, $N_{\text{test}} = 98$) to guarantee zero lookahead data leakage.
- **Key Features:** `starting_hour`, `day_of_week_num`, `is_weekend`, `is_midterm`, `academic_workload`, `campus_event_flag`, `weather_condition`.

### Hypothesis Testing Highlights ($\alpha = 0.05$)
1. **Sports Complex Weekend Surge:**
   - *Test:* Weekend attendance vs. weekday attendance.
   - *Result:* Welch's $t = 4.1967$ ($p = 3.71 \times 10^{-5}$), Mann-Whitney $U = 31798.0$ ($p < 0.001$), Cohen's $d = 0.4145$.
   - *Inference:* **Reject $H_0$**. Significant weekend sports demand surge confirmed.
2. **Library Midterm Crunch:**
   - *Test:* Library occupancy during exams vs. non-exam periods.
   - *Result:* Welch's $t = 41.1962$ ($p = 5.31 \times 10^{-51}$), Cohen's $d = 5.7911$.
   - *Inference:* **Reject $H_0$**. Demand surges by $+260\%$ during midterm periods.
3. **Cafeteria Lunch Stockout Risk:**
   - *Test:* Lunch rush order volume (12:00–14:00) vs. off-peak slots.
   - *Result:* Welch's $t = 27.5346$ ($p < 0.001$), Cohen's $d = 2.2275$.
   - *Inference:* **Reject $H_0$**. Meal preparation needs concentrated pre-staging during lunch.

---

## 5. Machine Learning Benchmarks

Evaluated on the isolated 20% holdout test set ($N_{\text{test}} = 98$):

### Regression Performance (Continuous Demand)
| Target Metric | Selected Model | MAE | RMSE | $R^2$ Score | Benchmark Baseline ($R^2$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Sports Visitors** | Random Forest Regressor | **4.82** | **7.12** | **0.929** | Ridge: 0.684 |
| **Food Meal Orders** | Gradient Boosted Regressor | **6.14** | **8.89** | **0.894** | Linear: 0.541 |
| **Library Occupancy** | Random Forest Regressor | **7.31** | **11.02** | **0.912** | Ridge: 0.710 |

### Classification Performance (Sports Overcrowding Risk $\ge 80\%$)
- **Model:** Random Forest Classifier
- **Accuracy:** **91.84%**
- **Recall (Sensitivity):** **88.89%** (detects 8 out of 9 critical overcrowding incidents)
- **Precision:** **53.33%**
- **F1-Score:** **66.67%**

$$\text{Confusion Matrix} = \begin{pmatrix} \text{TN = 82} & \text{FP = 7} \\ \text{FN = 1} & \text{TP = 8} \end{pmatrix}$$

### Unsupervised Clustering & PCA
- **Algorithm:** $K$-Means clustering with $K=3$ (calibrated via Silhouette Analysis: score $= 0.448$).
- **Principal Component Analysis:** First two principal components capture **54.9%** of cumulative campus variance (PC1 = 31.1%, PC2 = 23.8%).
- **Operational Regimes Identified:**
  - *Cluster 0 (Off-Peak / Balanced, N=176):* Low traffic across all facilities.
  - *Cluster 1 (Exam Crunch, N=138):* Severe library congestion, minimal sports usage.
  - *Cluster 2 (Recreation & Event Surge, N=176):* High sports complex and cafeteria lunch traffic.

---

## 6. Repository Structure

```
Campus-IQ/
├── data/                               # Structured datasets (empirical & processed)
│   ├── user_uploaded_campus_data.csv   # 490 empirical verified campus observations
│   ├── user_uploaded_campus_data.xlsx  # Master Excel workbook
│   ├── train_dataset.csv               # 392 training observations (80%)
│   ├── test_dataset.csv                # 98 testing observations (20%)
│   ├── campus_iq_simulated.csv         # Benchmark dataset
│   ├── campus_iq_simulated_benchmark.csv # Reference baseline benchmark dataset
│   └── generate_dataset.py             # Deterministic data generation script
├── models_saved/                       # Serialized trained model suites & metadata
│   ├── sports_suite.joblib             # Trained Sports complex models
│   ├── food_suite.joblib               # Trained Food outlet models
│   ├── library_suite.joblib            # Trained Library models
│   ├── cluster_suite.joblib            # Trained K-Means & PCA models
│   └── pipeline_summary.json           # Serialized metrics and performance metadata
├── src/                                # Core modular Python package
│   ├── models/                         # Facility-specific ML models
│   │   ├── sports_models.py
│   │   ├── food_models.py
│   │   ├── library_models.py
│   │   └── clustering_pca.py
│   ├── cpi_engine.py                   # Campus Pulse Index calculation engine
│   ├── data_processing.py              # Zero-leakage chronological split & cleaners
│   ├── demand_flow.py                  # Markov chain transition flow matrix
│   ├── excel_importer.py               # Excel workbook validator and parser
│   ├── explainability.py               # Feature importance & local attribution
│   ├── recommender.py                  # Alternate low-congestion slot dispatcher
│   └── statistical_analysis.py         # Descriptive stats, t-tests, Mann-Whitney
├── web/                                # Production interactive web application
│   ├── css/style.css                   # Custom responsive typography & modern theme
│   ├── images/                         # Clean campus vector illustrations
│   │   ├── building_clean.png
│   │   └── campus_map.png
│   ├── js/                             # Modular JavaScript clients
│   │   ├── app.js                      # UI event handling & reactive state
│   │   ├── chart.umd.min.js            # Chart.js library
│   │   └── charts.js                   # Interactive canvas visualizations
│   ├── index.html                      # Main campus dashboard
│   └── server.py                       # Python HTTP server with auto port failover
├── Campus_IQ_PreProcessing_and_EDA.ipynb # Preprocessing, EDA, and statistical distributions
├── evaluate_all.py                     # Full benchmark evaluation script
├── import_xlsx.py                      # Master Excel converter script
├── run_pipeline.py                     # One-click end-to-end ML pipeline runner
├── test_campus_iq.py                   # Automated 10-test unit and regression suite
├── requirements.txt                    # Python runtime dependencies
└── README.md                           # System documentation
```

---

## 7. UN Sustainable Development Goals (SDG) Alignment

- **SDG 11 (Target 11.7 — Sustainable Cities and Communities):** Proactively balances campus public spaces, prevents dangerous human density buildups, and maximizes equitable access to study facilities and sports infrastructure.
- **SDG 12 (Target 12.3 — Responsible Consumption and Production):** Minimizes dining hall food waste by predicting daily meal demand curves, helping cafeteria management tailor food procurement and prep quantities.

---

## 8. Authors & Academic Acknowledgments

Developed for the **Statistical Machine Learning** course at **Bennett University (The Times Group)**:
- **Aarsi Trivedi** (`S25CSEU1882`) — *Project Architecture, ML Pipeline, Web Dashboard & Report*
- **Arju** (`S25CSEU1868`) — *Statistical Rigor, Data Audit & Hypothesis Testing*
- **Angel** (`S25CSEU1873`) — *Facility Model Benchmarks, Validation & Documentation*
