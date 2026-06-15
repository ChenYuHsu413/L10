# ================================================================
# CRISP-DM: Forecasting 2330.TW Daily Close
# AR(20) Linear Regression vs auto-order ARIMA (walk-forward)
#   1.Business  2.Data Understanding  3.Data Prep
#   4.Modeling  5.Evaluation          6.Deployment
# ================================================================

import warnings; warnings.filterwarnings("ignore")
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def banner(t): print("\n" + "=" * 64 + f"\n{t}\n" + "=" * 64)

def metrics(yt, yp):
    return dict(RMSE=np.sqrt(mean_squared_error(yt, yp)),
                MAE=mean_absolute_error(yt, yp), R2=r2_score(yt, yp))

def select_arima_order(series, p=range(4), d=(1,), q=range(4)):
    """Pick (p,d,q) minimising AIC on the training series."""
    best_aic, best = np.inf, (1, 1, 0)
    for pp in p:
        for dd in d:
            for qq in q:
                try:
                    aic = ARIMA(series, order=(pp, dd, qq)).fit().aic
                    if aic < best_aic: best_aic, best = aic, (pp, dd, qq)
                except Exception: pass
    return best, best_aic

def walk_forward_arima(train, test, order):
    """One-step-ahead; extend with append(refit=False) -> fast, no 49 refits."""
    res = ARIMA(np.asarray(train, float), order=order).fit()
    preds = []
    for t in range(len(test)):
        preds.append(res.forecast(1)[0])
        res = res.append([test.iloc[t]], refit=False)
    return pd.Series(preds, index=test.index)


# ---------------- PHASE 1 — BUSINESS UNDERSTANDING ----------------
banner("PHASE 1 — BUSINESS UNDERSTANDING")
print("""Objective : forecast next-day Close of 2330.TW (TSMC).
Goal      : compare two univariate models on one-step-ahead accuracy.
Success   : a model is useful only if it beats a naive random-walk
            baseline (tomorrow = today) on RMSE/MAE on held-out data.
Models    : Naive | sklearn AR(20) | auto-order ARIMA (walk-forward).""")


# ---------------- PHASE 2 — DATA UNDERSTANDING -------------------
banner("PHASE 2 — DATA UNDERSTANDING")
df = yf.download("2330.TW", period="1y", interval="1d", auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
close = df["Close"].dropna()
returns = close.pct_change().dropna()

print(f"Rows {len(df)} | {df.index.min().date()} -> {df.index.max().date()} | "
      f"missing {int(df['Close'].isna().sum())}")
print("Close  mean/std : %.1f / %.1f" % (close.mean(), close.std()))
print("Return mean/std : %.3f%% / %.3f%%" % (returns.mean()*100, returns.std()*100))
print("ADF raw  Close : p=%.4f -> %s" %
      (adfuller(close)[1], "stationary" if adfuller(close)[1] < .05 else "NON-stationary"))
print("ADF diff Close : p=%.4f -> %s (justifies d=1)" %
      (adfuller(close.diff().dropna())[1],
       "stationary" if adfuller(close.diff().dropna())[1] < .05 else "non-stationary"))

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
ax[0, 0].plot(close); ax[0, 0].set_title("Close"); ax[0, 0].grid(True)
ax[0, 1].plot(returns, color="tab:orange"); ax[0, 1].set_title("Daily returns"); ax[0, 1].grid(True)
plot_acf(close.diff().dropna(), ax=ax[1, 0], lags=30); ax[1, 0].set_title("ACF diff (-> q)")
plot_pacf(close.diff().dropna(), ax=ax[1, 1], lags=30, method="ywm"); ax[1, 1].set_title("PACF diff (-> p)")
plt.tight_layout(); plt.savefig("phase2_eda.png", dpi=120); plt.close()
print("Saved -> phase2_eda.png")


# ---------------- PHASE 3 — DATA PREPARATION --------------------
banner("PHASE 3 — DATA PREPARATION")
split = int(len(close) * 0.8)
train_close, test_close = close.iloc[:split], close.iloc[split:]

lag = 20
data = pd.DataFrame({"Close": close})
for i in range(1, lag + 1):
    data[f"lag_{i}"] = data["Close"].shift(i)
data.dropna(inplace=True)
cols = [f"lag_{i}" for i in range(1, lag + 1)]
sl = int(len(data) * 0.8)
X_train, X_test = data[cols].iloc[:sl], data[cols].iloc[sl:]
y_train, y_test = data["Close"].iloc[:sl], data["Close"].iloc[sl:]
print(f"Train {len(train_close)} | Test {len(test_close)} | AR matrix {X_train.shape}")


# ---------------- PHASE 4 — MODELING ----------------------------
banner("PHASE 4 — MODELING")
naive_pred = test_close.shift(1); naive_pred.iloc[0] = train_close.iloc[-1]
lr = LinearRegression().fit(X_train, y_train)
lr_pred = pd.Series(lr.predict(X_test), index=y_test.index)
order, aic = select_arima_order(np.asarray(train_close, float))
arima_pred = walk_forward_arima(train_close, test_close, order)
print(f"Naive: fitted | AR(20): fitted | ARIMA auto-order={order} (AIC {aic:.1f})")


# ---------------- PHASE 5 — EVALUATION --------------------------
banner("PHASE 5 — EVALUATION")
idx = y_test.index.intersection(test_close.index)
results = pd.DataFrame({
    "Naive baseline": metrics(test_close.loc[idx], naive_pred.loc[idx]),
    "sklearn AR(20)": metrics(y_test.loc[idx], lr_pred.loc[idx]),
    f"ARIMA{order}":  metrics(test_close.loc[idx], arima_pred.loc[idx]),
}).T.round(3)
print(f"One-step-ahead accuracy ({len(idx)} common days):")
print(results.to_string())
best = results["RMSE"].idxmin()
beats = "YES" if results.loc[best, "RMSE"] < results.loc["Naive baseline", "RMSE"] else "NO"
print(f"\nBest by RMSE: {best} | beats naive baseline: {beats}")

resid = test_close.loc[idx] - arima_pred.loc[idx]
fig, a = plt.subplots(1, 2, figsize=(13, 4))
a[0].plot(resid); a[0].axhline(0, color="r", ls="--"); a[0].set_title("ARIMA residuals"); a[0].grid(True)
a[1].hist(resid, bins=15, color="tab:green", alpha=.8); a[1].set_title("Residual distribution")
plt.tight_layout(); plt.savefig("phase5_residuals.png", dpi=120); plt.close()

plt.figure(figsize=(13, 6))
plt.plot(test_close.index, test_close.values, "k", lw=2, label="Actual")
plt.plot(lr_pred.index, lr_pred.values, alpha=.8, label="sklearn AR(20)")
plt.plot(arima_pred.index, arima_pred.values, alpha=.8, label=f"ARIMA{order}")
plt.plot(naive_pred.index, naive_pred.values, ":", alpha=.5, label="Naive")
plt.title("2330.TW one-step-ahead forecast"); plt.xlabel("Date"); plt.ylabel("Close")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("phase5_comparison.png", dpi=120); plt.close()
print("Saved -> phase5_residuals.png, phase5_comparison.png")


# ---------------- PHASE 6 — DEPLOYMENT --------------------------
banner("PHASE 6 — DEPLOYMENT")
last = float(close.iloc[-1])
lr_next = float(lr.predict(data["Close"].iloc[-lag:].values[::-1].reshape(1, -1))[0])
arima_next = float(ARIMA(np.asarray(close, float), order=order).fit().forecast(1)[0])
print(f"Last close ({close.index[-1].date()}): {last:.2f}")
print(f"Next-day  Naive: {last:.2f} | AR(20): {lr_next:.2f} ({lr_next/last-1:+.2%}) | "
      f"ARIMA{order}: {arima_next:.2f} ({arima_next/last-1:+.2%})")
results.to_csv("phase6_model_metrics.csv")
print("Saved -> phase6_model_metrics.csv")
print("Note: series is near random-walk; retrain daily, monitor RMSE drift.")
