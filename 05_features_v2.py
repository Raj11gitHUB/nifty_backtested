"""
02b_features_v2.py
Expanded feature set on top of the original 26 features:
- Stochastic Oscillator (%K, %D)
- Williams %R
- Rate of Change (ROC)
- Average True Range (ATR)
- Commodity Channel Index (CCI)
- Rolling skew/kurtosis of returns (distribution shape)
- Momentum acceleration (return of returns)
- Multi-day-ahead SMA crossovers
- Calendar features (month, day of month)
"""
import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'nifty_merged.csv'), parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ===== Original feature set =====
df['Return'] = df['Close'].pct_change()
df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

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

# ===== NEW features =====

# --- Stochastic Oscillator ---
low14 = df['Low'].rolling(14).min()
high14 = df['High'].rolling(14).max()
df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-9)
df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

# --- Williams %R ---
df['Williams_R'] = -100 * (high14 - df['Close']) / (high14 - low14 + 1e-9)

# --- Rate of Change (ROC) ---
for p in [5, 10, 20]:
    df[f'ROC_{p}'] = df['Close'].pct_change(p) * 100

# --- Average True Range (ATR, 14-day) ---
prev_close = df['Close'].shift(1)
tr1 = df['High'] - df['Low']
tr2 = (df['High'] - prev_close).abs()
tr3 = (df['Low'] - prev_close).abs()
true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['ATR_14'] = true_range.rolling(14).mean()
df['ATR_pct'] = df['ATR_14'] / df['Close']  # normalized, scale-free

# --- Commodity Channel Index (CCI, 20-day) ---
typical_price = (df['High'] + df['Low'] + df['Close']) / 3
tp_sma = typical_price.rolling(20).mean()
tp_mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
df['CCI_20'] = (typical_price - tp_sma) / (0.015 * tp_mad + 1e-9)

# --- Distribution shape of recent returns ---
df['Return_skew_10'] = df['Return'].rolling(10).skew()
df['Return_kurt_10'] = df['Return'].rolling(10).kurt()

# --- Momentum acceleration (is momentum speeding up or slowing down) ---
df['Momentum_accel'] = df['Return'] - df['Return'].shift(1)

# --- SMA slope (trend strength, not just position) ---
df['SMA20_slope'] = df['SMA_20'].pct_change(5)

# --- Consecutive up/down streak ---
up_day = (df['Return'] > 0).astype(int)
df['Streak'] = up_day.groupby((up_day != up_day.shift()).cumsum()).cumcount() + 1
df['Streak'] = df['Streak'] * np.where(up_day == 1, 1, -1)

# --- Calendar ---
df['Month'] = df['Date'].dt.month
df['DayOfMonth'] = df['Date'].dt.day

feature_cols = [
    'Return', 'LogReturn', 'Return_lag1', 'Return_lag2', 'Return_lag3', 'Return_lag5', 'Return_lag10',
    'Close_to_SMA_5', 'Close_to_SMA_10', 'Close_to_SMA_20', 'Close_to_SMA_50', 'SMA5_gt_SMA20',
    'Volatility_5', 'Volatility_10', 'Volatility_20',
    'HL_range', 'OC_range', 'Close_pos_in_range',
    'Volume_change', 'Volume_ratio',
    'RSI_14', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_position', 'DayOfWeek',
    'Stoch_K', 'Stoch_D', 'Williams_R', 'ROC_5', 'ROC_10', 'ROC_20',
    'ATR_pct', 'CCI_20', 'Return_skew_10', 'Return_kurt_10',
    'Momentum_accel', 'SMA20_slope', 'Streak', 'Month', 'DayOfMonth'
]

df_model = df.dropna(subset=feature_cols + ['Target']).reset_index(drop=True)
print("Final shape:", df_model.shape)
print(f"Total features: {len(feature_cols)} (was 26, now {len(feature_cols)})")

df_model.to_csv(os.path.join(SCRIPT_DIR, 'nifty_features_v2.csv'), index=False)

import pickle
with open(os.path.join(SCRIPT_DIR, 'feature_cols_v2.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)

print("Saved nifty_features_v2.csv and feature_cols_v2.pkl")
