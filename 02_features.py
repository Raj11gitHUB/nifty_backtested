import pandas as pd
import numpy as np
import os
# Looks for the CSVs in a 'DATA' subfolder next to this script
# (i.e. D:\project\NIFTY\DATA if this script is in D:\project\NIFTY)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR)
 

df = pd.read_csv('D:/project/NIFTY/DATA/nifty_merged.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# ---- Basic returns ----
df['Return'] = df['Close'].pct_change()
df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))

# ---- Target: next day movement (1 = up, 0 = down/flat) ----
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# ---- Lagged returns (past N days) ----
for lag in [1, 2, 3, 5, 10]:
    df[f'Return_lag{lag}'] = df['Close'].pct_change(lag)

# ---- Moving averages & ratios ----
for w in [5, 10, 20, 50]:
    df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
    df[f'Close_to_SMA_{w}'] = df['Close'] / df[f'SMA_{w}'] - 1

df['SMA5_gt_SMA20'] = (df['SMA_5'] > df['SMA_20']).astype(int)

# ---- Volatility ----
df['Volatility_5'] = df['Return'].rolling(5).std()
df['Volatility_10'] = df['Return'].rolling(10).std()
df['Volatility_20'] = df['Return'].rolling(20).std()

# ---- Intraday range / candle features ----
df['HL_range'] = (df['High'] - df['Low']) / df['Close']
df['OC_range'] = (df['Close'] - df['Open']) / df['Open']
df['Close_pos_in_range'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-9)

# ---- Volume features ----
df['Volume_change'] = df['Volume'].pct_change()
df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_10']

# ---- RSI (14) ----
delta = df['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / (avg_loss + 1e-9)
df['RSI_14'] = 100 - (100 / (1 + rs))

# ---- MACD ----
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_diff'] = df['MACD'] - df['MACD_signal']

# ---- Bollinger Bands ----
sma20 = df['Close'].rolling(20).mean()
std20 = df['Close'].rolling(20).std()
df['BB_upper'] = sma20 + 2 * std20
df['BB_lower'] = sma20 - 2 * std20
df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-9)

# ---- Day of week (seasonality) ----
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Drop rows with NaNs from rolling windows, and the last row (no target)
feature_cols = [c for c in df.columns if c not in ['Date', 'Open', 'High', 'Low', 'Close',
                                                     'Volume', 'Turnover', 'Target']]
df_model = df.dropna(subset=feature_cols + ['Target']).reset_index(drop=True)

print("Final shape:", df_model.shape)
print("Feature columns:", feature_cols)
print("\nTarget balance:\n", df_model['Target'].value_counts(normalize=True))

df_model.to_csv('D:/project/NIFTY/DATA/nifty_features.csv', index=False)
