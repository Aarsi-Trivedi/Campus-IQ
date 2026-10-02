# Campus IQ: An Interpretable Statistical Machine Learning System for Predicting Campus Resource Demand and Overcrowding

> **Core Principle:** *Campus Activity &rarr; Demand &rarr; Risk &rarr; Recommendation*

---

## 1. Project Overview & Scope
**Campus IQ** is an interpretable, classical/statistical machine learning campus intelligence system designed to analyze historical campus activity patterns and predict resource demand, overcrowding, and resource-related risks across interconnected campus facilities. 

Rather than treating facilities as isolated silos, Campus IQ investigates **Campus Demand Flow** across three primary zones:
1. **Sports Complex (Primary Focus):** Mitigates severe peak-hour overcrowding, estimates visitor counts and court loads, predicts overcrowding risk ($\ge 80\%$ capacity), and recommends lower-demand alternate time slots.
2. **Food Outlets (Cafeterias & Kiosks):** Forecasts meal and order demand, calculates food surplus and shortage risks, and optimizes daily kitchen preparation to reduce food waste (**SDG 12**).
3. **Library & Study Spaces:** Analyzes seat occupancy, identifies exam crunch periods, and reroutes students to underutilized satellite study zones (**SDG 11**).

### What Makes Campus IQ Different
- **Interconnected Campus Demand Flow:** Captures how academic dismissals, exams, and sporting events cause ripples across sports, dining, and study facilities over time.
- **Predictive Rather than Descriptive:** Estimates future demand and bottlenecks rather than merely showing current crowd counts.
- **Action-Oriented Recommendations:** Converts raw ML predictions into immediate operational recommendations (e.g. shift from 18:00 to 14:00 for 42% lower crowd and zero wait times).
- **Deliberate Classical ML Focus:** Strictly avoids opaque deep learning, neural networks, or LLMs in favor of transparent, interpretable statistical algorithms.
- **Explainability:** Employs feature importance rankings, regression coefficients, and instance-level factor attribution so facility managers understand *why* overcrowding is predicted.

---

## 2. Mathematical Formulations & Derived Metrics

### 1. Facility Load Percentages
$$\text{Sports Load (\%)} = \left(\frac{\text{Sports Visitors}}{\text{Capacity}_{\text{sports}}}\right) \times 100 \quad (\text{Capacity} = 200)$$
$$\text{Library Load (\%)} = \left(\frac{\text{Library Occupancy}}{\text{Capacity}_{\text{library}}}\right) \times 100 \quad (\text{Capacity} = 350)$$

### 2. Overcrowding Risk Flag
$$\text{Overcrowded Flag} = \begin{cases} 1 & \text{if Sports Load} \ge 80.0\% \\ 0 & \text{otherwise} \end{cases}$$

### 3. Food Waste & Shortage Metrics (SDG 12)
$$\text{Food Sold} = \min(\text{Prepared Quantity}, \text{Order Demand})$$
$$\text{Food Surplus Quantity} = \max(0, \text{Prepared Quantity} - \text{Food Sold})$$
$$\text{Food Surplus (\%)} = \left(\frac{\text{Food Surplus Quantity}}{\max(1, \text{Prepared Quantity})}\right) \times 100$$
$$\text{Food Shortage Quantity} = \max(0, \text{Order Demand} - \text{Prepared Quantity})$$

### 4. Campus Pulse Index (CPI)
A normalized composite indicator ($0$ to $100$) reflecting aggregate campus-wide operational pressure. Weights are mathematically and operationally justified by facility volatility and queuing severity:
$$\text{CPI} = w_{\text{sports}} \cdot \text{SportsLoad} + w_{\text{lib}} \cdot \text{LibraryLoad} + w_{\text{food}} \cdot \left(\frac{\text{Orders}}{300} \cdot 100\right) + w_{\text{workload}} \cdot \text{WorkloadScore}$$

- $w_{\text{sports}} = 0.35$: Sports complex represents the tightest physical safety and equipment bottleneck.
- $w_{\text{lib}} = 0.30$: Library represents core academic throughput with acute exam season surges.
- $w_{\text{food}} = 0.25$: High-frequency dining service with perishable food waste risks.
- $w_{\text{workload}} = 0.10$: Macro academic stressor and class release driving campus-wide movement.
$$\sum w_i = 0.35 + 0.30 + 0.25 + 0.10 = 1.00$$

---

## 3. Machine Learning Methods & Benchmarks

All models were evaluated on an isolated **chronological test split (20%, 100 observations)** to prevent temporal data leakage.

### Regression Tasks (Continuous Demand)
| Domain | Target | Candidate Model | MAE | RMSE | $R^2$ Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sports Complex** | Visitor Count | **Random Forest Regressor** | **6.05** | **7.36** | **0.9687** | **Selected Best** |
| Sports Complex | Visitor Count | Linear Regression | 6.07 | 7.77 | 0.9651 | Baseline |
| Sports Complex | Visitor Count | Gradient Boosting Regressor | 6.27 | 7.74 | 0.9654 | Candidate |
| **Food Outlets** | Order Demand | **Linear Regression** | **8.76** | **10.64** | **0.9703** | **Selected Best** |
| Food Outlets | Order Demand | Random Forest Regressor | 9.53 | 11.39 | 0.9660 | Candidate |
| Food Outlets | Order Demand | Gradient Boosting Regressor | 9.96 | 11.91 | 0.9628 | Candidate |
| **Library** | Seat Occupancy | **Linear Regression** | **7.84** | **10.12** | **0.9546** | **Selected Best** |

### Classification Tasks (Overcrowding & Risk)
| Domain | Target | Classifier | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sports Complex** | Overcrowding ($\ge 80\%$) | **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **Selected Best** |
| Sports Complex | Overcrowding ($\ge 80\%$) | Random Forest Classifier | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Candidate |
| Sports Complex | Overcrowding ($\ge 80\%$) | Gradient Boosting Classifier | 0.9600 | 1.0000 | 0.7895 | 0.8824 | Candidate |
| **Food Outlets** | Surplus / Shortage Risk | **Logistic Regression** | **0.6700** | **0.6812** | **0.6700** | **0.6505** | **Selected Best** |

### Unsupervised Clustering & PCA
- **K-Means Clustering:** Evaluated across $K=2 \dots 6$ using Silhouette Scores and Inertia Elbow methods. **Optimal $K=3$** identified:
  - **Cluster 0 — Campus Event & Dining Surge (34.0%):** Triggered by cultural fests/tournaments; high food orders ($212$ avg) and active circulation.
  - **Cluster 1 — Academic Intensive / Exam Pressure (8.4%):** Severe library bottleneck ($100.6\%$ load), subdued sports demand ($29.2\%$).
  - **Cluster 2 — Off-Peak / Balanced Circulation (57.6%):** Ample capacity across all facilities; ideal maintenance window.
- **PCA Dimensionality Reduction:** 2 Principal Components capture **$54.9\%$ of total multi-facility variance** ($\text{PC1} = 31.1\%, \text{PC2} = 23.8\%$).

---

## 4. Statistical Rigor & Hypothesis Testing

Conducted at $\alpha = 0.05$ using both parametric (Student's Two-Sample t-test) and non-parametric (Mann-Whitney U) tests with Cohen's $d$ effect sizes:

1. **Hypothesis 1 (Sports Exam Drop):**
   - *H1:* Sports Complex attendance drops significantly during Exam Weeks compared to Non-Exam Weeks.
   - *Result:* $t = -9.136$, $p = 1.21 \times 10^{-12}$, Mann-Whitney $U = 2571.5$ ($p = 3.74 \times 10^{-15}$), Cohen's $d = -1.161$ (Large effect).
   - *Decision:* **Reject $H_0$**. Confirms academic workload overrides recreational sports attendance.

2. **Hypothesis 2 (Library Exam Surge):**
   - *H2:* Library occupancy increases significantly during Exam Weeks.
   - *Result:* $t = 43.951$, $p = 3.93 \times 10^{-73}$, Mann-Whitney $U = 19169.5$ ($p = 1.58 \times 10^{-26}$), Cohen's $d = 3.321$ (Very Large effect).
   - *Decision:* **Reject $H_0$**. Average occupancy surges from $173$ to $352$ seats.

3. **Hypothesis 3 (Sports Event & Dining Spillover):**
   - *H3:* Food outlet demand increases significantly on Sports Event / Tournament days.
   - *Result:* $t = 4.563$, $p = 2.45 \times 10^{-5}$, Mann-Whitney $U = 16278.0$ ($p = 7.90 \times 10^{-7}$), Cohen's $d = 0.678$ (Medium effect).
   - *Decision:* **Reject $H_0$**. Validates cross-facility flow spillover.

4. **IQR Outlier Investigation (Section 7):**
   - Sports crowd upper threshold: $Q_3 + 1.5 \times \text{IQR} = 177.62$ visitors.
   - 48 crowd spikes identified.
   - *Methodological Rationale:* In accordance with Section 7, these observations were **investigated and retained** rather than deleted, because real-world tournaments and cultural fests represent genuine high-impact operational surges that the ML system must learn to anticipate.

---

## 5. SDG Alignment & Transparency (Sections 11 & 12)

- **SDG 11 (Sustainable Cities and Communities):** Supports equitable, data-driven access to shared campus physical infrastructure. Dynamically shifts student traffic to off-peak slots and satellite study lounges to prevent spatial congestion.
- **SDG 12 (Responsible Consumption and Production):** Prevents food waste in campus cafeterias by providing kitchen managers with demand forecasts and optimal prep batch recommendations within a $\pm 5\%$ safety margin.
- **Methodological Transparency (Section 11):** The initial 500-sample dataset is explicitly and honestly labeled as `CALIBRATED_SIMULATION_BENCHMARK`. The system includes interactive tools to upload and append real sensor/turnstile campus logs.

---

## 6. Project Architecture

```
campus_iq/
├── data/
│   ├── generate_dataset.py       # Calibrated simulation generator
│   ├── campus_iq_simulated.csv   # 500-record benchmark dataset
│   ├── train_dataset.csv         # 400 training records (isolated)
│   └── test_dataset.csv          # 100 testing records (isolated)
├── src/
│   ├── data_processing.py        # Cleaning, encoding, and chronological splitting
│   ├── statistical_analysis.py   # Descriptive stats, IQR outliers, t-tests, MWU, correlations
│   ├── cpi_engine.py             # Campus Pulse Index calculation & weighted scoring
│   ├── demand_flow.py            # Cross-facility transitions and lead-lag analysis
│   ├── recommender.py            # Actionable lower-demand slot recommendation engine
│   ├── explainability.py         # Instance-level feature attribution & factor deconstruction
│   └── models/
│       ├── sports_models.py      # Regressors & Classifiers for sports complex
│       ├── food_models.py        # Regressors & Classifiers for food outlets (SDG 12)
│       ├── library_models.py     # Regressors & Classifiers for study space demand
│       └── clustering_pca.py     # K-Means clustering (K=2..6) + PCA 2D projections
├── models_saved/                 # Serialized joblib models and pipeline_summary.json
├── web/
│   ├── server.py                 # REST API + Static HTTP server (Python standard library)
│   ├── index.html                # Interactive modern academic web dashboard
│   ├── css/style.css             # Glassmorphism dark-theme styling
│   └── js/
│       ├── app.js                # Frontend API controller & interactive scenario simulator
│       └── charts.js             # Dynamic Chart.js visualizations
├── run_pipeline.py               # Complete end-to-end ML pipeline runner
├── evaluate_all.py               # Comprehensive CLI statistical benchmark report
├── test_campus_iq.py             # Automated unit and system test suite
├── requirements.txt              # Core dependencies
└── README.md                     # System documentation & mathematical formulations
```

---

## 7. How to Run the Project

### 1. Run the Full ML Pipeline
To re-run data cleaning, statistical tests, model training, K-Means clustering, and artifact serialization:
```bash
python3 run_pipeline.py
```

### 2. View Terminal Evaluation Benchmark Report
To generate the formatted statistical and model evaluation report in your terminal:
```bash
python3 evaluate_all.py
```

### 3. Run Automated Tests
To run all 10 automated unit and integration tests:
```bash
python3 test_campus_iq.py
```

### 4. Launch the Interactive Web Dashboard
The web dashboard is served by Python's built-in HTTP server with JSON REST APIs:
```bash
python3 web/server.py 8080
```
Open your browser and navigate to:
**`http://localhost:8080`**

---

## 8. Dashboard Features Overview
- **Campus Pulse Index (CPI):** Real-time composite dial gauge ($0-100$) reflecting campus-wide strain.
- **Sports Complex Simulator:** Sliders for time, day, class dismissals, tournaments, and weather with instant visitor predictions and overcrowding risk badges.
- **Actionable Time Recommendations:** Displays the top 2-3 lower-demand alternative windows with exact percentage load reductions.
- **Explainability Deconstruction:** Horizontal waterfall chart explaining the positive/negative percentage pushes of each input factor.
- **Food Prep Optimizer (SDG 12):** Input planned meal batch $\rightarrow$ calculate estimated surplus/shortage and receive waste prevention guidance.
- **Study Space Availability:** Forecasts library capacity and suggests satellite quiet zones.
- **2D PCA Cluster Explorer:** Interactive scatter plot colored by K-Means cluster personas.
- **Statistical Rigor Panel:** Interactive tables for descriptive stats, Pearson correlation matrices, IQR outliers, and formal hypothesis test cards.
