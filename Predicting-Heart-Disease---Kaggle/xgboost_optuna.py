import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

# 1. Load data
# (Assuming data paths are correct for your environment)
train = pd.read_csv('data/train_encoded.csv')
test = pd.read_csv('data/test.csv')
features = [c for c in train.columns if c not in ['id', 'Heart Disease']]

X = train[features]
y = train['Heart Disease']
X_test = test[features]


# --- PHASE 1: OPTUNA HYPERPARAMETER TUNING ---
def objective(trial):
    param = {
        'n_estimators': 3000,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42,
        'eval_metric': 'logloss',
        # FIX 1: Move early_stopping_rounds to the constructor
        'early_stopping_rounds': 50
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X, y):
        # FIX 2: Correctly reference the data splits
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**param)

        # FIX 3: eval_set is still passed here, but early_stopping_rounds is NOT
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds = model.predict_proba(X_val)[:, 1]
        scores.append(log_loss(y_val, preds))

    return np.mean(scores)


print("Starting Optuna Study...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)

best_params = study.best_params
print(f"Best Parameters found: {best_params}")

# --- PHASE 2: FINAL K-FOLD WITH CALIBRATION ---
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
final_test_preds = np.zeros(len(test))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Initialize model with best params
    # We add early_stopping here too for the final training
    base_model = xgb.XGBClassifier(**best_params, n_estimators=2000,
                                   early_stopping_rounds=50, random_state=42)

    # FIX 4: Calibration & Early Stopping interaction
    # CalibratedClassifierCV fits the model internally. To use early stopping
    # effectively with calibration, we fit the model first, then use cv='prefit'.
    base_model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)

    cal_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    cal_model.fit(X_val_fold, y_val_fold)

    final_test_preds += cal_model.predict_proba(X_test)[:, 1] / n_splits
    print(f"Fold {fold + 1} complete.")

# 3. Create Submission
results = pd.DataFrame({'id': test['id'], 'Heart Disease': final_test_preds})
results.to_csv('submission.csv', index=False)
print("Submission saved!")