import yfinance as yf
df = yf.download("MC.PA", start="2020-01-01", end="2025-10-01", auto_adjust=False)
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print(df.head(2))