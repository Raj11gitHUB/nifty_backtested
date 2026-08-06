r"""
07_predict_next_day_v2.py

Same as before, but uses the improved v2 models (41 features, tuned
hyperparameters, ensemble) AND only gives a directional call when the
model is confident enough - otherwise it tells you honestly to sit out.

Folder layout expected: this script + scaler_v2.pkl + models_v2.pkl +
feature_cols_v2.pkl + nifty_merged.csv, all in the same folder.

CONFIDENCE_THRESHOLD below is set from backtested results: on the held-out
test set, when the ensemble's confidence was >= 0.55, it was right about
61% of the time (vs ~54% overall) - but only covered ~22% of days.
Change this threshold to trade off "how often it speaks" vs "how sure it is
when it does." Do not push it much past 0.60: past that point the sample of
historical high-confidence days gets too small to trust.
"""

import pandas as pd
import numpy as np
import pickle
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'nifty_merged.csv')
CONFIDENCE_THRESHOLD = 0.55

with open(os.path.join(SCRIPT_DIR, 'scaler_v2.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(SCRIPT_DIR, 'models_v2.pkl'), 'rb') as f:
    models = pickle.load(f)
with open(os.path.join(SCRIPT_DIR, 'feature_cols_v2.pkl'), 'rb') as f:
    feature_cols = pickle.load(f)

df = pd.read_csv(DATA_FILE, parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ---- Rebuild the exact same v2 feature set (must match 05_features_v2.py) ----
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

low14 = df['Low'].rolling(14).min()
high14 = df['High'].rolling(14).max()
df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-9)
df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()
df['Williams_R'] = -100 * (high14 - df['Close']) / (high14 - low14 + 1e-9)

for p in [5, 10, 20]:
    df[f'ROC_{p}'] = df['Close'].pct_change(p) * 100

prev_close = df['Close'].shift(1)
tr1 = df['High'] - df['Low']
tr2 = (df['High'] - prev_close).abs()
tr3 = (df['Low'] - prev_close).abs()
true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['ATR_14'] = true_range.rolling(14).mean()
df['ATR_pct'] = df['ATR_14'] / df['Close']

typical_price = (df['High'] + df['Low'] + df['Close']) / 3
tp_sma = typical_price.rolling(20).mean()
tp_mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
df['CCI_20'] = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-9)

df['Return_skew_10'] = df['Return'].rolling(10).skew()
df['Return_kurt_10'] = df['Return'].rolling(10).kurt()
df['Momentum_accel'] = df['Return'] - df['Return'].shift(1)
df['SMA20_slope'] = df['SMA_20'].pct_change(5)

up_day = (df['Return'] > 0).astype(int)
df['Streak'] = up_day.groupby((up_day != up_day.shift()).cumsum()).cumcount() + 1
df['Streak'] = df['Streak'] * np.where(up_day == 1, 1, -1)

df['Month'] = df['Date'].dt.month
df['DayOfMonth'] = df['Date'].dt.day

# ---- Predict on latest row ----
latest_row = df.iloc[[-1]]
latest_date = latest_row['Date'].iloc[0].date()
latest_close = latest_row['Close'].iloc[0]

if latest_row[feature_cols].isna().any(axis=1).iloc[0]:
    raise ValueError("Latest row has missing features - need more history for rolling windows.")

X_latest = latest_row[feature_cols].values
X_latest_scaled = scaler.transform(X_latest)

print(f"Latest data: {latest_date}  (Close = {latest_close:.2f})")
print(f"Predicting movement for the NEXT trading day\n")

ensemble = models['Ensemble (soft voting)']
proba = ensemble.predict_proba(X_latest_scaled)[0]
pred = int(proba[1] > proba[0])
confidence = proba[pred]
direction = "UP" if pred == 1 else "DOWN"

print(f"Ensemble prediction: {direction}  (P[Up]={proba[1]:.1%}, P[Down]={proba[0]:.1%})\n")

if confidence >= CONFIDENCE_THRESHOLD:
    print(f"Confidence {confidence:.1%} >= threshold {CONFIDENCE_THRESHOLD:.0%} -> "
          f"model is willing to call this one: {direction}")
    print("(Historically, calls at this confidence level were right ~61% of the time,")
    print(" but only made on ~22% of days - so wait for these moments rather than")
    print(" acting every day.)")
else:
    print(f"Confidence {confidence:.1%} < threshold {CONFIDENCE_THRESHOLD:.0%} -> "
          f"SIT OUT. Today looks like a coin flip, the model has no edge here.")

print("\nAll individual models for reference:")
for name, model in models.items():
    p = model.predict_proba(X_latest_scaled)[0]
    pr = int(p[1] > p[0])
    print(f"  {name:28s} -> {'UP' if pr==1 else 'DOWN':5s}  (P[Up]={p[1]:.1%})")

print("\nThis is a research tool, not financial advice. Even the confidence-filtered")
print("accuracy is based on a limited backtest and does not account for trading")
print("costs, slippage, or regime change. Do not trade real money on this alone.")
