import pandas as pd
import glob
import os
 
# Looks for the CSVs in a 'DATA' subfolder next to this script
# (i.e. D:\project\NIFTY\DATA if this script is in D:\project\NIFTY)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'DATA')
 
files = sorted(glob.glob(os.path.join(DATA_DIR, 'NIFTY 50-*.csv')))
print(f"Looking in: {DATA_DIR}")
print(f"Found {len(files)} files")
 
files = sorted(glob.glob('D:project/NIFTY/DATA/NIFTY 50-*.csv'))
print(f"Found {len(files)} files")

dfs = []
for f in files:
    df = pd.read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data.columns = [c.strip() for c in data.columns]
print("Columns:", data.columns.tolist())

# Parse date and sort ascending (oldest -> newest)
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%Y')
data = data.sort_values('Date').drop_duplicates(subset='Date').reset_index(drop=True)

# Rename for convenience
data = data.rename(columns={
    'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close',
    'Shares Traded': 'Volume', 'Turnover (₹ Cr)': 'Turnover'
})

print(data.shape)
print(data.head())
print(data.tail())
print(data.isna().sum())

data.to_csv('D:/project/NIFTY/DATA/nifty_merged.csv', index=False)
print("Saved merged file.")
