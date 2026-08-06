"""
06_train_model_v2.py
Improvements over the original 03_train_model.py:
1. 41 features instead of 26 (stochastic, ATR, CCI, Williams %R, skew/kurtosis, streaks, etc.)
2. Hyperparameter tuning via RandomizedSearchCV with time-series-safe CV
3. A soft-voting ensemble of the tuned models
4. Confidence-based filtering: instead of predicting every day, only act when the
   model's probability is above a threshold. This is the realistic way to improve
   usable accuracy on a noisy target - skip the coin-flip days.
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(SCRIPT_DIR, 'nifty_features_v2.csv'), parse_dates=['Date'])

import pickle
with open(os.path.join(SCRIPT_DIR, 'feature_cols_v2.pkl'), 'rb') as f:
    feature_cols = pickle.load(f)

X = df[feature_cols].values
y = df['Target'].values

split_idx = int(len(df) * 0.8)
X_train_full, X_test = X[:split_idx], X[split_idx:]
y_train_full, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {df['Date'].iloc[0].date()} to {df['Date'].iloc[split_idx-1].date()} ({split_idx} rows)")
print(f"Test:  {df['Date'].iloc[split_idx].date()} to {df['Date'].iloc[-1].date()} ({len(X_test)} rows)")
print(f"Features: {len(feature_cols)}\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

tscv = TimeSeriesSplit(n_splits=5)

# ---------------------------------------------------------------
# Hyperparameter search (time-series safe - no shuffling, no leakage)
# ---------------------------------------------------------------
print("=== Hyperparameter search (this takes a couple minutes) ===\n")

rf_param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 8],
    'min_samples_leaf': [10, 20, 30, 50],
    'max_features': ['sqrt', 'log2', 0.5],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_param_dist, n_iter=20, cv=tscv, scoring='accuracy', random_state=42, n_jobs=-1
)
rf_search.fit(X_train_scaled, y_train_full)
print("Best RandomForest params:", rf_search.best_params_)
print(f"Best RandomForest CV accuracy: {rf_search.best_score_:.4f}\n")

gb_param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [2, 3],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.7, 0.9],
    'min_samples_leaf': [20, 30],
}
gb_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_param_dist, n_iter=10, cv=tscv, scoring='accuracy', random_state=42, n_jobs=-1
)
gb_search.fit(X_train_scaled, y_train_full)
print("Best GradientBoosting params:", gb_search.best_params_)
print(f"Best GradientBoosting CV accuracy: {gb_search.best_score_:.4f}\n")

lr = LogisticRegression(max_iter=1000, C=0.05)
lr.fit(X_train_scaled, y_train_full)

best_rf = rf_search.best_estimator_
best_gb = gb_search.best_estimator_

# ---------------------------------------------------------------
# Soft-voting ensemble of the tuned models
# ---------------------------------------------------------------
ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', best_rf), ('gb', best_gb)],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train_full)

# ---------------------------------------------------------------
# Evaluate on held-out test set
# ---------------------------------------------------------------
print("=== Final test set evaluation ===")
baseline_acc = max(y_test.mean(), 1 - y_test.mean())
print(f"Naive baseline: {baseline_acc:.4f}\n")

final_models = {
    'LogisticRegression': lr,
    'RandomForest (tuned)': best_rf,
    'GradientBoosting (tuned)': best_gb,
    'Ensemble (soft voting)': ensemble,
}

for name, model in final_models.items():
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    print(f"{name:28s} test accuracy: {acc:.4f}")

# ---------------------------------------------------------------
# Confidence-based filtering: only predict when the model is
# actually confident. This is the realistic path to a usable
# accuracy number, at the cost of fewer trading days covered.
# ---------------------------------------------------------------
print("\n=== Confidence-filtered accuracy (Ensemble) ===")
print("Only counts predictions where the model's confidence exceeds the threshold.\n")

proba = ensemble.predict_proba(X_test_scaled)
confidence = proba.max(axis=1)
preds = ensemble.predict(X_test_scaled)

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = confidence >= threshold
    coverage = mask.sum() / len(mask)
    if mask.sum() > 0:
        acc = accuracy_score(y_test[mask], preds[mask])
        print(f"Confidence >= {threshold:.2f}: accuracy = {acc:.4f}  "
              f"(covers {coverage:.1%} of days, {mask.sum()} of {len(mask)})")
    else:
        print(f"Confidence >= {threshold:.2f}: no days meet this threshold")

# ---------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------
with open(os.path.join(SCRIPT_DIR, 'scaler_v2.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(SCRIPT_DIR, 'models_v2.pkl'), 'wb') as f:
    pickle.dump(final_models, f)
with open(os.path.join(SCRIPT_DIR, 'feature_cols_v2.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"\nSaved improved models to {SCRIPT_DIR}")
