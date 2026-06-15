# ============================================================
# Predict 2330.TW Close: sklearn AR(20) vs ARIMA
# Compact comparison. One-step-ahead, walk-forward, fair.
# ============================================================

import warnings; warnings.filterwarnings("ignore")
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA


def select_arima_order(series, p=range(4), d=(1,), q=range(4)):
    """Pick (p,d,q) minimising AIC on the training series (one-time)."""
    best_aic, best = np.inf, (1, 1, 0)
    for pp in p:
        for dd in d:
            for qq in q:
                try:
                    aic = ARIMA(series, order=(pp, dd, qq)).fit().aic
                    if aic < best_aic:
                        best_aic, best = aic, (pp, dd, qq)
                except Exception:
                    pass
    return best


def walk_forward_arima(train, test, order):
    """One-step-ahead forecasts; extend state with append(refit=False) -> fast."""
    res = ARIMA(np.asarray(train, float), order=order).fit()
    preds = []
    for t in range(len(test)):
        preds.append(res.forecast(1)[0])
        res = res.append([test.iloc[t]], refit=False)   # no full refit
    return pd.Series(preds, index=test.index)


def metrics(y_true, y_pred):
    return dict(RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
                MAE=mean_absolute_error(y_true, y_pred),
                R2=r2_score(y_true, y_pred))


# ---- Data ----
df = yf.download("2330.TW", period="1y", interval="1d", auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
close = df["Close"].dropna()

# ---- Prepare: 80/20 chronological split + AR(20) lag matrix ----
lag = 20
data = pd.DataFrame({"Close": close})
for i in range(1, lag + 1):
    data[f"lag_{i}"] = data["Close"].shift(i)
data.dropna(inplace=True)
cols = [f"lag_{i}" for i in range(1, lag + 1)]

split = int(len(data) * 0.8)
X_train, X_test = data[cols].iloc[:split], data[cols].iloc[split:]
y_train, y_test = data["Close"].iloc[:split], data["Close"].iloc[split:]
train_close, test_close = close.iloc[:int(len(close)*0.8)], close.iloc[int(len(close)*0.8):]

# ---- Model A: sklearn AR(20), one-step-ahead ----
lr = LinearRegression().fit(X_train, y_train)
lr_pred = pd.Series(lr.predict(X_test), index=y_test.index)

# ---- Model B: ARIMA, auto order, walk-forward one-step-ahead ----
order = select_arima_order(np.asarray(train_close, float))
arima_pred = walk_forward_arima(train_close, test_close, order)

# ---- Naive baseline (sanity floor) ----
naive_pred = test_close.shift(1); naive_pred.iloc[0] = train_close.iloc[-1]

# ---- Evaluate on common test days ----
idx = y_test.index.intersection(test_close.index)
results = pd.DataFrame({
    "Naive baseline":     metrics(test_close.loc[idx], naive_pred.loc[idx]),
    "sklearn AR(20)":     metrics(y_test.loc[idx], lr_pred.loc[idx]),
    f"ARIMA{order}":      metrics(test_close.loc[idx], arima_pred.loc[idx]),
}).T.round(3)
print(f"Auto-selected ARIMA order: {order}")
print(results.to_string())

# ---- Plot ----
plt.figure(figsize=(13, 6))
plt.plot(test_close.index, test_close.values, "k", lw=2, label="Actual")
plt.plot(lr_pred.index, lr_pred.values, alpha=.8, label="sklearn AR(20)")
plt.plot(arima_pred.index, arima_pred.values, alpha=.8, label=f"ARIMA{order}")
plt.plot(naive_pred.index, naive_pred.values, ":", alpha=.5, label="Naive")
plt.title("2330.TW one-step-ahead forecast"); plt.xlabel("Date"); plt.ylabel("Close")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("comparison.png", dpi=120)
print("Saved -> comparison.png")

# ---- Next-day forecast ----
next_x = data["Close"].iloc[-lag:].values[::-1].reshape(1, -1)
lr_next = float(lr.predict(next_x)[0])
arima_next = float(ARIMA(np.asarray(close, float), order=order).fit().forecast(1)[0])
print(f"\nLast close: {float(close.iloc[-1]):.2f}")
print(f"Next-day  AR(20): {lr_next:.2f}   ARIMA{order}: {arima_next:.2f}")
