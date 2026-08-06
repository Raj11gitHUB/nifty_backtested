import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv('D:/project/NIFTY/DATA/nifty_features.csv', parse_dates=['Date'])

# Use only stationary/normalized features (drop raw price-level SMAs, BB bands, Volume_SMA)
feature_cols = [
    'Return', 'LogReturn', 'Return_lag1', 'Return_lag2', 'Return_lag3', 'Return_lag5', 'Return_lag10',
    'Close_to_SMA_5', 'Close_to_SMA_10', 'Close_to_SMA_20', 'Close_to_SMA_50', 'SMA5_gt_SMA20',
    'Volatility_5', 'Volatility_10', 'Volatility_20',
    'HL_range', 'OC_range', 'Close_pos_in_range',
    'Volume_change', 'Volume_ratio',
    'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_position', 'DayOfWeek'
]

X = df[feature_cols].values
y = df['Target'].values
dates = df['Date'].values

# ---- Hold out the most recent 20% as a final untouched test set (chronological) ----
split_idx = int(len(df) * 0.8)
X_train_full, X_test = X[:split_idx], X[split_idx:]
y_train_full, y_test = y[:split_idx], y[split_idx:]
dates_test = dates[split_idx:]

print(f"Train period: {df['Date'].iloc[0].date()} to {df['Date'].iloc[split_idx-1].date()} ({split_idx} rows)")
print(f"Test period:  {df['Date'].iloc[split_idx].date()} to {df['Date'].iloc[-1].date()} ({len(X_test)} rows)")

# ---- Scale features (fit only on train) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# ---- Walk-forward CV on training set to pick model/hyperparams ----
tscv = TimeSeriesSplit(n_splits=5)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, C=0.1),
    'RandomForest': RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=20,
                                            random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.03,
                                                     subsample=0.8, random_state=42),
}

print("\n=== Walk-forward CV (on training set only) ===")
cv_results = {}
for name, model in models.items():
    accs = []
    for tr_idx, val_idx in tscv.split(X_train_scaled):
        model.fit(X_train_scaled[tr_idx], y_train_full[tr_idx])
        preds = model.predict(X_train_scaled[val_idx])
        accs.append(accuracy_score(y_train_full[val_idx], preds))
    cv_results[name] = accs
    print(f"{name}: mean acc = {np.mean(accs):.4f}  (folds: {[round(a,3) for a in accs]})")

# ---- Fit best-looking models on FULL train, evaluate on held-out test ----
print("\n=== Final evaluation on held-out (future) test set ===")
baseline_acc = max(y_test.mean(), 1 - y_test.mean())
print(f"Naive baseline (always predict majority class): {baseline_acc:.4f}\n")

results_summary = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train_full)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results_summary.append((name, acc))
    print(f"--- {name} ---")
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=['Down', 'Up']))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))
    print()

# ---- Feature importance from RandomForest ----
rf = models['RandomForest']
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n=== Top 15 feature importances (RandomForest) ===")
print(importances.head(15))

import pickle
with open('/home/claude/nifty/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('/home/claude/nifty/models.pkl', 'wb') as f:
    pickle.dump(models, f)
with open('/home/claude/nifty/feature_cols.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print("\nSaved scaler, models, and feature_cols to /home/claude/nifty/")
