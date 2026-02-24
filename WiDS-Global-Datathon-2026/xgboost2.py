import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight

# 1. Load Data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# 2. Feature Engineering
def engineer_features(df):
    eps = 1e-6
    # ETA: How many hours until it hits?
    df['eta_hours'] = df['dist_min_ci_0_5h'] / (df['closing_speed_abs_m_per_h'] + eps)
    # Intensity: Growth x Speed
    df['fire_intensity'] = df['closing_speed_m_per_h'] * df['area_growth_rate_ha_per_h']
    # Threat: Size / Distance
    df['proximity_threat'] = df['log1p_area_first'] / (np.log1p(df['dist_min_ci_0_5h']) + eps)
    # Directional Alignment
    df['aligned_speed'] = df['closing_speed_m_per_h'] * df['alignment_cos']
    return df


train = engineer_features(train)
test = engineer_features(test)

# Define features for the model
features = [c for c in train.columns if c not in ['event_id', 'time_to_hit_hours', 'event', 'target']]

horizons = [12, 24, 48, 72]
submission = pd.DataFrame({'event_id': test['event_id']})
last_model = None

# 3. Training Loop for each Horizon
for h in horizons:
    # Labeling: 1 if hit by H, 0 if observed past H without hit, NaN otherwise
    y = np.full(len(train), np.nan)
    y[(train['event'] == 1) & (train['time_to_hit_hours'] <= h)] = 1
    y[train['time_to_hit_hours'] > h] = 0

    train_h = train.copy()
    train_h['target'] = y
    train_h = train_h.dropna(subset=['target'])

    if train_h['target'].nunique() >= 2:
        # Balance the classes (hits are often the minority)
        weights = compute_sample_weight(class_weight='balanced', y=train_h['target'])

        # Train calibrated Gradient Boosting
        base = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42)
        model = CalibratedClassifierCV(base, method='sigmoid', cv=5)
        model.fit(train_h[features], train_h['target'], sample_weight=weights)

        submission[f'prob_{h}h'] = model.predict_proba(test[features])[:, 1]
        last_model = model
    else:
        # Fallback for 72h (data often stops before 72h)
        prev_h = f'prob_{horizons[horizons.index(h) - 1]}h'
        submission[f'prob_{h}h'] = submission[prev_h] + (1 - submission[prev_h]) * 0.05

# 4. Enforce Physical Consistency
submission['prob_24h'] = np.maximum(submission['prob_24h'], submission['prob_12h'])
submission['prob_48h'] = np.maximum(submission['prob_48h'], submission['prob_24h'])
submission['prob_72h'] = np.maximum(submission['prob_72h'], submission['prob_48h'])

# Save to CSV
submission.to_csv('submission.csv', index=False)