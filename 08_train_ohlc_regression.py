r"""
08_train_ohlc_regression.py

Predicts next-day OPEN, HIGH, LOW, CLOSE (not just direction).

APPROACH:
Instead of predicting raw price levels directly (which is misleadingly easy -
tomorrow's close is almost always close to today's close, so a naive model
looks "accurate" while being useless), this predicts each price as a RETURN
relative to today's close:
    target_open  = (next_day_open  - today_close) / today_close
    target_high  = (next_day_high  - today_close) / today_close
    target_low   = (next_day_low   - today_close) / today_close
    target_close = (next_day_close - today_close) / today_close
Then reconstructs actual predicted prices as: today_close * (1 + predicted_return)

This is evaluated against two baselines:
    - "Naive same": predict tomorrow's OHLC all equal to today's close
    - "Naive drift": predict tomorrow's OHLC using the average historical
      Open/High/Low/Close-to-prior-close relationship (average gap/range)

Uses the same 41 engineered features as before (built fresh here so this
script is self-contained - only needs nifty_merged.csv).
"""
import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'nifty_merged.csv'), parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ===== Same 41 features as 05_features_v2.py =====
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

# ===== Regression targets: next-day O/H/L/C as % return vs today's close =====
df['target_open']  = (df['Open'].shift(-1)  - df['Close']) / df['Close']
df['target_high']  = (df['High'].shift(-1)  - df['Close']) / df['Close']
df['target_low']   = (df['Low'].shift(-1)   - df['Close']) / df['Close']
df['target_close'] = (df['Close'].shift(-1) - df['Close']) / df['Close']

target_cols = ['target_open', 'target_high', 'target_low', 'target_close']
df_model = df.dropna(subset=feature_cols + target_cols).reset_index(drop=True)

print(f"Final shape: {df_model.shape}")

X = df_model[feature_cols].values
Y = df_model[target_cols].values
close_today = df_model['Close'].values
dates = df_model['Date'].values

# Actual next-day OHLC (for evaluation, reconstructed from raw data - ground truth)
actual_next_open  = df_model['Open'].shift(-1).values   # note: last row will be NaN, dropped below anyway before use
actual_next_high  = df_model['High'].shift(-1).values
actual_next_low   = df_model['Low'].shift(-1).values
actual_next_close = df_model['Close'].shift(-1).values

# ---- Chronological 80/20 split ----
split_idx = int(len(df_model) * 0.8)

X_train, X_test = X[:split_idx], X[split_idx:]
Y_train, Y_test = Y[:split_idx], Y[split_idx:]
close_test = close_today[split_idx:]
dates_test = dates[split_idx:]

print(f"Train: {df_model['Date'].iloc[0].date()} to {df_model['Date'].iloc[split_idx-1].date()} ({split_idx} rows)")
print(f"Test:  {df_model['Date'].iloc[split_idx].date()} to {df_model['Date'].iloc[-1].date()} ({len(X_test)} rows)\n")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Train one regressor per target (Open, High, Low, Close) ----
model_type = 'gbr'  # 'ridge', 'rf', or 'gbr'

def make_model():
    if model_type == 'ridge':
        return Ridge(alpha=10.0)
    elif model_type == 'rf':
        return RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=30,
                                      random_state=42, n_jobs=-1)
    else:
        return GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.03,
                                          subsample=0.8, random_state=42)

regressors = {}
predicted_returns_test = {}

for i, tgt in enumerate(['open', 'high', 'low', 'close']):
    model = make_model()
    model.fit(X_train_scaled, Y_train[:, i])
    regressors[tgt] = model
    predicted_returns_test[tgt] = model.predict(X_test_scaled)

# ---- Reconstruct predicted prices from predicted returns ----
predicted_prices = {}
for tgt in ['open', 'high', 'low', 'close']:
    predicted_prices[tgt] = close_test * (1 + predicted_returns_test[tgt])

# Ensure High >= Low, and High >= max(Open,Close), Low <= min(Open,Close) - just for sanity/display
pred_open, pred_high, pred_low, pred_close = (
    predicted_prices['open'], predicted_prices['high'],
    predicted_prices['low'], predicted_prices['close']
)

# ---- Ground truth actual next-day prices for test period ----
actual_open_test  = actual_next_open[split_idx:]
actual_high_test  = actual_next_high[split_idx:]
actual_low_test   = actual_next_low[split_idx:]
actual_close_test = actual_next_close[split_idx:]

# Drop last row of test set (no ground truth available for the very last day)
valid = ~np.isnan(actual_close_test)
n_dropped = (~valid).sum()

def eval_target(name, pred, actual):
    pred, actual = pred[valid], actual[valid]
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    r2 = r2_score(actual, pred)
    print(f"{name:6s}  MAE={mae:8.2f} pts   RMSE={rmse:8.2f} pts   MAPE={mape:5.2f}%   R2={r2:6.3f}")
    return mae, rmse, mape, r2

print("=== Model performance (test set, reconstructed prices) ===")
eval_target('Open',  pred_open,  actual_open_test)
eval_target('High',  pred_high,  actual_high_test)
eval_target('Low',   pred_low,   actual_low_test)
eval_target('Close', pred_close, actual_close_test)

# ---- Naive baseline: predict tomorrow's OHLC all = today's close ----
print("\n=== Naive baseline (predict tomorrow's OHLC = today's close) ===")
naive_pred = close_test
eval_target('Open',  naive_pred, actual_open_test)
eval_target('High',  naive_pred, actual_high_test)
eval_target('Low',   naive_pred, actual_low_test)
eval_target('Close', naive_pred, actual_close_test)

print("\nNote: MAE/RMSE are in NIFTY index points. R2 close to 0 means the model")
print("adds little over just guessing 'tomorrow looks like today' - this is the")
print("honest, expected result for next-day price levels on a liquid index.")

# ---- Directional accuracy implied by the close prediction (bonus check) ----
pred_direction = (pred_close > close_test) & valid
actual_direction = (actual_close_test > close_test) & valid
dir_acc = (pred_direction[valid] == actual_direction[valid]).mean()
print(f"\nImplied next-day direction accuracy from Close regression: {dir_acc:.4f}")

# ---- Save everything ----
with open(os.path.join(SCRIPT_DIR, 'ohlc_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(SCRIPT_DIR, 'ohlc_regressors.pkl'), 'wb') as f:
    pickle.dump(regressors, f)
with open(os.path.join(SCRIPT_DIR, 'ohlc_feature_cols.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"\nSaved ohlc_scaler.pkl, ohlc_regressors.pkl, ohlc_feature_cols.pkl to {SCRIPT_DIR}")
