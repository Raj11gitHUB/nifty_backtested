r"""
predict_next_day.py

Loads the trained models (scaler.pkl, models.pkl, feature_cols.pkl) and your
latest merged CSV to predict whether NIFTY will go UP or DOWN tomorrow,
based on today's (most recent) data.

Folder layout expected (adjust DATA_FILE below if yours differs):
    D:\project\NIFTY\
    ├── predict_next_day.py
    ├── scaler.pkl
    ├── models.pkl
    ├── feature_cols.pkl
    └── nifty_merged.csv        <- the RAW merged file from 01_load_data.py
                                    (not nifty_features.csv - we rebuild
                                    features fresh here so it's always current)

HOW TO KEEP THIS UP TO DATE:
Each day, after markets close, add the new day's OHLCV row to your CSVs
(or download an updated yearly CSV from NSE) and re-run 01_load_data.py,
then run this script. It always predicts for the day AFTER the last row.
"""

import pandas as pd
import numpy as np
import pickle
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'nifty_merged.csv')  # <-- adjust if needed

# ---------------------------------------------------------------
# 1. Load trained artifacts
# ---------------------------------------------------------------
with open(os.path.join(SCRIPT_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(SCRIPT_DIR, 'models.pkl'), 'rb') as f:
    models = pickle.load(f)
with open(os.path.join(SCRIPT_DIR, 'feature_cols.pkl'), 'rb') as f:
    feature_cols = pickle.load(f)

# ---------------------------------------------------------------
# 2. Load latest raw price data and rebuild the SAME features
#    used during training (must match 02_features.py exactly)
# ---------------------------------------------------------------
df = pd.read_csv(DATA_FILE, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

df['Return'] = df['Close'].pct_change()
df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))

for lag in [1, 2, 3, 5, 10]:
    df[f'Return_lag{lag}'] = df['Close'].pct_change(lag)

for w in [5, 10, 20, 50]:
    df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
    df[f'Close_to_SMA_{w}'] = df['Close'] / df[f'SMA_{w}'] - 1

df['SMA5_gt_SMA20'] = (df['SMA_5'] > df['SMA_20']).astype(int)

df['Volatility_5'] = df['Return'].rolling(5).std()
df['Volatility_10'] = df['Return'].rolling(10).std()
df['Volatility_20'] = df['Return'].rolling(20).std()

df['HL_range'] = (df['High'] - df['Low']) / df['Close']
df['OC_range'] = (df['Close'] - df['Open']) / df['Open']
df['Close_pos_in_range'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)

df['Volume_change'] = df['Volume'].pct_change()
df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_10']

delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / (avg_loss + 1e-9)
df['RSI_14'] = 100 - (100 / (1 + rs))

ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_diff'] = df['MACD'] - df['MACD_signal']

sma20 = df['Close'].rolling(20).mean()
std20 = df['Close'].rolling(20).std()
df['BB_upper'] = sma20 + 2 * std20
df['BB_lower'] = sma20 - 2 * std20
df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-9)

df['DayOfWeek'] = df['Date'].dt.dayofweek

# ---------------------------------------------------------------
# 3. Take the LAST row (most recent trading day) as input
# ---------------------------------------------------------------
latest_row = df.iloc[[-1]]
latest_date = latest_row['Date'].iloc[0].date()
latest_close = latest_row['Close'].iloc[0]

if latest_row[feature_cols].isna().any(axis=1).iloc[0]:
    raise ValueError(
        "Latest row has missing feature values (not enough history for rolling "
        "windows, e.g. SMA_50 needs 50+ days). Check your data."
    )

X_latest = latest_row[feature_cols].values
X_latest_scaled = scaler.transform(X_latest)

# ---------------------------------------------------------------
# 4. Predict with every model
# ---------------------------------------------------------------
print(f"Latest data available: {latest_date}  (Close = {latest_close:.2f})")
print(f"Predicting movement for the NEXT trading day\n")

for name, model in models.items():
    pred = model.predict(X_latest_scaled)[0]
    proba = model.predict_proba(X_latest_scaled)[0]
    direction = "UP" if pred == 1 else "DOWN"
    confidence = proba[pred]
    print(f"{name:20s} -> {direction:5s}  (confidence: {confidence:.1%}, "
          f"P[Up]={proba[1]:.1%}, P[Down]={proba[0]:.1%})")

print("\nReminder: these models test at ~54-55% accuracy, barely above the")
print("54% naive baseline of 'always predict Up'. Treat this as informational,")
print("not a trading signal — do not act on it with real money without much")
print("more validation, risk management, and realistic backtesting.")
