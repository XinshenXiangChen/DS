import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# 1. Load data
train = pd.read_csv('data/train_encoded.csv')
test = pd.read_csv('data/test.csv')

features = [c for c in train.columns if c not in ['id', 'Heart Disease']]


X = train[features]
y = train['Heart Disease']
X_test = test[features]


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


final_test_preds = np.zeros(len(test))

print(f"Starting {n_splits}-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]


    model = xgb.XGBClassifier(
        n_estimators=3000,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )


    cal_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    cal_model.fit(X_train_fold, y_train_fold)

    fold_preds = cal_model.predict_proba(X_test)[:, 1]
    final_test_preds += fold_preds / n_splits

    print(f"Fold {fold + 1} complete.")

results = pd.DataFrame({'id': test['id'], 'Heart Disease': final_test_preds})
results.to_csv('data/submission.csv', index=False)

print("\nOptimized submission saved!")