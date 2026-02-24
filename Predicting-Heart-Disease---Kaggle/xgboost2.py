import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV


train = pd.read_csv('data/train_encoded.csv')
test = pd.read_csv('data/test.csv')

features = [c for c in train.columns if c not in ['id', 'Heart Disease']]


base_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    verbosity=0,
    use_label_encoder=False,
    eval_metric='logloss'
)


cal_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
cal_model.fit(train[features], train['Heart Disease'])


results = pd.DataFrame({'id': test['id']})

results['Heart Disease'] = cal_model.predict_proba(test[features])[:, 1]

results.to_csv('data/submission.csv', index=False)

print("Submission file created successfully in data/submission.csv!")