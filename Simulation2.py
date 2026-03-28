import pandas as pd
import numpy as np
import joblib
import json
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn import set_config
from utils import preprocess_test, numerical_features, features_todrop,categorical_features, preprocessor

import utils
utils.preprocessor = joblib.load('models/Trees/preprocessor.pkl')



set_config(transform_output="default")

#Loading model
model = CatBoostClassifier()
model.load_model('models/Trees/catboost_tuned.cbm')

train_columns_before_impute = joblib.load('models/Trees/train_columns.pkl')
preprocessor= joblib.load('models/Trees/preprocessor.pkl')
means_smoothed= joblib.load('models/Trees/state_means.pkl')
global_default_mean = joblib.load('models/Trees/global_default_mean.pkl')
upperbounds= joblib.load('models/Trees/upperbounds.pkl')
lowerbounds= joblib.load('models/Trees/lowerbounds.pkl')

with open('models/Trees/features_todrop.json') as f:
    features_todrop = json.load(f)

with open('models/Trees/raw_features.json') as f:
    raw_features = json.load(f)

with open('models/config.json') as f:
    config = json.load(f)

with open('models/Trees/numerical_features.json') as f:
    numerical_features = json.load(f)

threshold   = config['threshold']
lgd         = config['lgd']
avg_intRate = config['average_IntRate']
avg_loan    = config['avg_loan']


def process_chunks(filepath, model, raw_features,
                   means_smoothed, global_default_mean,
                   upperbounds, lowerbounds, 
                   numerical_features, train_columns,
                   chunksize=10_000):
    all_proba = []
    all_true  = []
    all_X     = []

    model_features = model.feature_names_

    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        y_chunk = chunk['predictor']
        X_chunk = chunk[raw_features].copy()

        X_pre = preprocess_test(
            X_chunk,
            means_smoothed,
            global_default_mean,
            upperbounds,
            lowerbounds,
            numerical_features,
            train_columns
        )

        X_pre = X_pre[model_features]
        proba = model.predict_proba(X_pre)[:, 1]

        all_proba.append(proba)
        all_true.append(y_chunk)
        all_X.append(X_pre)

    return (np.concatenate(all_proba),
            pd.concat(all_true).reset_index(drop=True),
            pd.concat(all_X).reset_index(drop=True))

y_proba, y_holdout, X_hold_pre = process_chunks(
    'Data/loan_data_holdout.csv',
    model, raw_features,
    means_smoothed, global_default_mean,
    upperbounds, lowerbounds,
    numerical_features, train_columns_before_impute
)


print(f"Rows:          {len(y_holdout):,}")
print(f"Default rate:  {y_holdout.mean():.3f}")
print(f"Proba range:   {y_proba.min():.3f} - {y_proba.max():.3f}")
print(f"Flagged:       {(y_proba >= threshold).mean()*100:.1f}%")

y_pred = (y_proba >= threshold).astype(int)


def evaluate_policy(y_true, y_pred, y_proba, X_data, lgd, avg_loan ,avg_intRate, label):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    interest_rev = avg_loan * avg_intRate
    
    # Our financial metrics
    loss = (fn * avg_loan * lgd) + (fp * interest_rev)
    gain = (tn * interest_rev) + (tp * avg_loan * lgd)
    net = gain - loss
    
    # model metrics 
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    flagged_pct = (tp + fp) / len(y_true) * 100
    default_rate_approved = fn / (fn + tn)
    
    print(f"\n{'='*50}")
    print(f"Policy: {label}")
    print(f"{'='*50}")
    print(f"Net Value:              ${net:>15,.0f}")
    print(f"Expected Loss:          ${loss:>15,.0f}")
    print(f"Expected Gain:          ${gain:>15,.0f}")
    print(f"Recall (defaulters):    {recall:>15.3f}")
    print(f"Precision:              {precision:>15.3f}")
    print(f"Flagged:                {flagged_pct:>14.1f}%")
    print(f"Default rate approved:  {default_rate_approved:>15.3f}")
    
    return net

#A/B test
control_pred = (y_proba >= 0.5).astype(int)
treatment_pred = (y_proba >= threshold).astype(int)

net_control   = evaluate_policy(y_holdout, control_pred,   y_proba, 
                                 X_hold_pre, lgd,avg_loan,avg_intRate,'Control (0.5)')
net_treatment = evaluate_policy(y_holdout, treatment_pred, y_proba, 
                                 X_hold_pre, lgd, avg_loan,avg_intRate, f'Treatment ({threshold})')

print(f"\nNet Value Improvement:  ${net_treatment - net_control:,.0f}")
print(f"Lift:                   {net_treatment / net_control:.1f}x")