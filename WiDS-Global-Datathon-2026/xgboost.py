import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# 1. Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 2. Feature Selection & Engineering
# We exclude IDs and target columns
features = [c for c in test.columns if c not in ['event_id']]


def add_features(df):
    # Time to impact estimate (Dist / Speed)
    df['est_time'] = df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'] + 1e-5)
    # Area per distance remaining
    df['density_threat'] = df['area_first_ha'] / (df['dist_min_ci_0_5h'] + 1e-5)
    return df


train = add_features(train)
test = add_features(test)
features += ['est_time', 'density_threat']

# 3. Training and Prediction Loop
horizons = [12, 24, 48, 72]
results = pd.DataFrame({'event_id': test['event_id']})

for h in horizons:
    # --- Labeling Logic ---
    # Label 1: Hit occurred within H hours
    # Label 0: Fire was observed for at least H hours and did NOT hit
    # NaN: Censored before H hours (we don't know if it hit or not)

    y = np.full(len(train), np.nan)
    y[(train['event'] == 1) & (train['time_to_hit_hours'] <= h)] = 1
    y[train['time_to_hit_hours'] > h] = 0


    train_h = train.copy()
    train_h['target'] = y
    train_h = train_h.dropna(subset=['target'])


    if train_h['target'].nunique() < 2:

        print(f"Skipping training for {h}h due to lack of class diversity.")
        results[f'prob_{h}h'] = results.iloc[:, -1]  # Copy previous horizon
        continue


    base_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)

    calibrated_model.fit(train_h[features], train_h['target'])

    results[f'prob_{h}h'] = calibrated_model.predict_proba(test[features])[:, 1]


results['prob_24h'] = np.maximum(results['prob_24h'], results['prob_12h'])
results['prob_48h'] = np.maximum(results['prob_48h'], results['prob_24h'])
results['prob_72h'] = np.maximum(results['prob_72h'], results['prob_48h'])

# 5. Save Output
results.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")