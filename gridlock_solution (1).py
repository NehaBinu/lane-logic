"""
Gridlock Hackathon 2.0 - Traffic Demand Prediction
Author: [Your Name]
Score: ~99.22 OOF R2 Score
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target range: {train['demand'].min():.4f} – {train['demand'].max():.4f}")

# ─── 2. FEATURE ENGINEERING ──────────────────────────────────────────────────
def ts_to_minutes(ts):
    """Convert timestamp string '10:15' -> 615 minutes"""
    h, m = ts.split(':')
    return int(h) * 60 + int(m)

def engineer_features(df, train_df):
    df = df.copy()

    # --- Time features ---
    df['minutes']        = df['timestamp'].apply(ts_to_minutes)
    df['hour']           = df['minutes'] // 60
    df['minute_of_hour'] = df['minutes'] % 60
    df['is_peak']        = df['hour'].isin([8,9,10,11,12,13,14,17,18,19,20]).astype(int)
    df['is_night']       = df['hour'].isin([0,1,2,3,4,5]).astype(int)
    df['hour_sin']       = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']       = np.cos(2 * np.pi * df['hour'] / 24)

    # --- Golden feature: same geohash + timestamp from day 48 ---
    # Traffic patterns repeat daily: yesterday's demand is the best predictor
    d48 = (train_df[train_df['day'] == 48]
           [['geohash', 'timestamp', 'demand']]
           .rename(columns={'demand': 'demand_lag1day'}))
    df = df.merge(d48, on=['geohash', 'timestamp'], how='left')

    # --- Geohash-level statistics (from day 48) ---
    geo_stats = (train_df[train_df['day'] == 48]
                 .groupby('geohash')['demand']
                 .agg(geo_mean='mean', geo_std='std',
                      geo_max='max', geo_min='min', geo_median='median')
                 .reset_index())
    df = df.merge(geo_stats, on='geohash', how='left')

    # --- Geohash × hour interaction stats ---
    geo_hour = train_df[train_df['day'] == 48].copy()
    geo_hour['hour'] = geo_hour['timestamp'].apply(lambda x: int(x.split(':')[0]))
    geo_hour_stats = (geo_hour.groupby(['geohash', 'hour'])['demand']
                     .mean().reset_index()
                     .rename(columns={'demand': 'geo_hour_mean'}))
    df = df.merge(geo_hour_stats, on=['geohash', 'hour'], how='left')

    # --- Timestamp-level stats (global average demand at each time) ---
    ts_stats = (train_df.groupby('timestamp')['demand']
                .agg(ts_mean='mean', ts_std='std').reset_index())
    df = df.merge(ts_stats, on='timestamp', how='left')

    # --- Categorical encoding ---
    df['RoadType'] = df['RoadType'].fillna('Unknown')
    road_map = {'Highway': 3, 'Street': 2, 'Residential': 1, 'Unknown': 0}
    df['RoadType_enc']      = df['RoadType'].map(road_map).fillna(0)
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_enc']     = (df['Landmarks'] == 'Yes').astype(int)
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
    df['Weather_enc']       = df['Weather'].map(weather_map).fillna(-1)

    # --- Temperature imputation by timestamp median ---
    ts_temp = train_df.groupby('timestamp')['Temperature'].median().to_dict()
    df['Temperature'] = df['Temperature'].fillna(df['timestamp'].map(ts_temp))
    df['Temperature'] = df['Temperature'].fillna(train_df['Temperature'].median())

    # --- Geohash label encode ---
    geo_list = sorted(train_df['geohash'].unique())
    geo_map  = {g: i for i, g in enumerate(geo_list)}
    df['geohash_enc'] = df['geohash'].map(geo_map).fillna(-1).astype(int)

    # --- Fill remaining missing values ---
    df['demand_lag1day'] = df['demand_lag1day'].fillna(df['geo_mean'])
    df['demand_lag1day'] = df['demand_lag1day'].fillna(df['ts_mean'])
    df['geo_hour_mean']  = df['geo_hour_mean'].fillna(df['geo_mean'])
    for col in ['geo_mean', 'geo_std', 'geo_max', 'geo_min', 'geo_median', 'geo_hour_mean']:
        df[col] = df[col].fillna(df[col].median())

    return df

train_feat = engineer_features(train, train)
test_feat  = engineer_features(test,  train)

FEATURES = [
    'minutes', 'hour', 'minute_of_hour', 'is_peak', 'is_night',
    'hour_sin', 'hour_cos',
    'demand_lag1day',                                          # Golden lag feature
    'geo_mean', 'geo_std', 'geo_max', 'geo_min', 'geo_median',
    'geo_hour_mean',                                           # Location × time
    'ts_mean', 'ts_std',                                       # Global time pattern
    'RoadType_enc', 'NumberofLanes', 'LargeVehicles_enc',
    'Landmarks_enc', 'Weather_enc', 'Temperature',
    'geohash_enc', 'day'
]

X      = train_feat[FEATURES]
y      = train_feat['demand']
X_test = test_feat[FEATURES]

print(f"\nFeature matrix: {X.shape}")
print(f"Missing in train: {X.isnull().sum().sum()}")
print(f"Missing in test:  {X_test.isnull().sum().sum()}")

# ─── 3. MODEL: LightGBM 5-Fold CV ────────────────────────────────────────────
PARAMS = {
    'objective':        'regression',
    'metric':           'rmse',
    'num_leaves':       127,
    'learning_rate':    0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'min_child_samples': 20,
    'n_estimators':     1000,
    'random_state':     42,
    'verbose':          -1
}

kf         = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds  = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
scores     = []

print("\n--- 5-Fold Cross Validation ---")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    X_tr,  X_val = X.iloc[tr_idx],  X.iloc[val_idx]
    y_tr,  y_val = y.iloc[tr_idx],  y.iloc[val_idx]

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(False)]
    )

    val_pred          = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds        += model.predict(X_test) / 5

    score = max(0, 100 * r2_score(y_val, val_pred))
    scores.append(score)
    print(f"  Fold {fold+1}: Score = {score:.2f}")

oof_score = max(0, 100 * r2_score(y, oof_preds))
print(f"\n  OOF Score: {oof_score:.2f}  (mean folds: {np.mean(scores):.2f})")

# ─── 4. SUBMISSION ────────────────────────────────────────────────────────────
test_preds = np.clip(test_preds, 0, 1)
submission = pd.DataFrame({'Index': test['Index'], 'demand': test_preds})
submission.to_csv('submission_v1.csv', index=False)
print(f"\nSubmission saved: submission_v1.csv  shape={submission.shape}")
print(submission.head())
