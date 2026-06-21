# 🚦 Lane Logic — Traffic Demand Prediction
**Gridlock Hackathon 2.0 | Flipkart × Bengaluru Traffic Police**  
**Participant:** Neha Binu | **Team:** Lane Logic | **Score:** 90.79 / 100

---

## Problem Statement
Predict traffic demand (0–1) for 41,778 road segments across Bengaluru for Day 49, using historical data from Day 48 and early Day 49 observations.

**Evaluation metric:** `score = max(0, 100 × R²(actual, predicted))`

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn lightgbm catboost
```

### 2. Place dataset files in the same folder
```
train.csv
test.csv
sample_submission.csv
```

### 3. Run the solution
```bash
# Option A: Run the Python script directly
python gridlock_solution.py

# Option B: Open the Jupyter notebook
jupyter notebook LaneLogic_Solution.ipynb
```

### 4. Output
`submission_final.csv` — 41,778 rows with predicted demand values, ready to upload.

---

## Live Prototype (Dashboard)
Open `lane_logic_prototype.html` in any browser — no installation required.

**Features:**
- Select event type (rally, festival, sports, construction)
- Choose Bengaluru zone, time, crowd size
- Get instant demand prediction + officer deployment plan
- Live map with colour-coded congestion zones
- 24-hour demand chart showing event impact

---

## Solution Summary

### Core Insight
Traffic follows daily habits. The same zone at the same time on Day 48 is the strongest predictor of Day 49 demand (`demand_lag1day`, correlation = 0.79).

### Key Technical Decisions

| Decision | Reason |
|---|---|
| Precise 4-key lag matching | Each geohash zone has multiple road segments. Matching on geohash+timestamp+RoadType+Lanes gives the exact same road yesterday, not an average. |
| LightGBM + CatBoost blend | CatBoost handles RoadType/Weather categoricals natively. Blending (40% LGB + 60% CAT) beats either model alone. |
| Geohash prefix clustering | First 4 chars of geohash = ~1km neighbourhood. Nearby zones share demand patterns. |
| Temporal validation | Random KFold inflated CV to 99.3. Real performance on unseen future data = 90.79. |

### Feature Count: 34 engineered features
### Models: LightGBM + CatBoost (5-fold CV each, then blended)
### Final Online Score: 90.79 / 100

---

## Files
| File | Description |
|---|---|
| `LaneLogic_Solution.ipynb` | Full solution notebook with explanations |
| `gridlock_solution.py` | Clean Python script version |
| `lane_logic_prototype.html` | Interactive dashboard prototype |
| `submission_final.csv` | Final predictions (41,778 rows) |
| `README.md` | This file |

