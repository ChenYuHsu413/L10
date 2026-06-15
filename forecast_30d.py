# ================================================================
# 2330.TW 未來 30 交易日預測 + 喇叭圖 (fan chart)
# ARIMA 信賴區間扇形 + AR(20) 遞迴預測對照線
# ================================================================

import warnings; warnings.filterwarnings("ignore")
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

HORIZON = 30   # 預測未來交易日數


def select_arima_order(series, p=range(4), d=(1,), q=range(4)):
    best_aic, best = np.inf, (1, 1, 0)
    for pp in p:
        for dd in d:
            for qq in q:
                try:
                    aic = ARIMA(series, order=(pp, dd, qq)).fit().aic
                    if aic < best_aic: best_aic, best = aic, (pp, dd, qq)
                except Exception: pass
    return best


# ---- 資料 ----
df = yf.download("2330.TW", period="1y", interval="1d", auto_adjust=True)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
close = df["Close"].dropna()

# 未來交易日索引(跳過週末)
future_idx = pd.bdate_range(close.index[-1] + pd.Timedelta(days=1), periods=HORIZON)

# ---- ARIMA: 全資料配適 -> 未來 30 步預測 + 信賴區間 ----
order = select_arima_order(np.asarray(close, float))
res = ARIMA(np.asarray(close, float), order=order).fit()
fc = res.get_forecast(steps=HORIZON)
mean = pd.Series(fc.predicted_mean, index=future_idx)
ci95 = fc.conf_int(alpha=0.05)   # 95% 帶
ci80 = fc.conf_int(alpha=0.20)   # 80% 帶
lo95, hi95 = ci95[:, 0], ci95[:, 1]
lo80, hi80 = ci80[:, 0], ci80[:, 1]

# ---- AR(20) 遞迴多步預測(對照線,無信賴區間)----
lag = 20
window = list(close.iloc[-lag:].values)
lr = LinearRegression().fit(
    np.column_stack([close.shift(i) for i in range(1, lag + 1)]).astype(float)[lag:],
    close.values[lag:])
ar_future = []
for _ in range(HORIZON):
    x = np.array(window[-lag:][::-1]).reshape(1, -1)   # lag_1..lag_20 (新->舊)
    yhat = float(lr.predict(x)[0])
    ar_future.append(yhat)
    window.append(yhat)
ar_future = pd.Series(ar_future, index=future_idx)

# ---- 喇叭圖 ----
hist = close.iloc[-60:]   # 只畫最近 60 天歷史,讓喇叭看得清楚
plt.figure(figsize=(13, 6))
plt.plot(hist.index, hist.values, color="black", lw=1.6, label="歷史收盤")
plt.plot(mean.index, mean.values, color="tab:blue", lw=2, label=f"ARIMA{order} 預測均值")
plt.fill_between(future_idx, lo95, hi95, color="tab:blue", alpha=0.15, label="95% 信賴區間")
plt.fill_between(future_idx, lo80, hi80, color="tab:blue", alpha=0.30, label="80% 信賴區間")
plt.plot(ar_future.index, ar_future.values, color="tab:orange", ls="--", lw=1.6,
         label="AR(20) 遞迴預測")
plt.axvline(close.index[-1], color="grey", ls=":", lw=1)
plt.title(f"2330.TW 未來 {HORIZON} 交易日預測 — 喇叭圖 (fan chart)")
plt.xlabel("日期"); plt.ylabel("收盤價")
plt.legend(loc="upper left"); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("forecast_30d.png", dpi=120)
print(f"自動選階 ARIMA order = {order}")
print(f"最後收盤 ({close.index[-1].date()}): {float(close.iloc[-1]):.2f}")
print("已存圖 -> forecast_30d.png")

# ---- 輸出預測表 ----
out = pd.DataFrame({
    "ARIMA_mean": mean, "lo80": lo80, "hi80": hi80,
    "lo95": lo95, "hi95": hi95, "AR20": ar_future.values}).round(2)
out.index.name = "date"
out.to_csv("forecast_30d.csv")
print("已存表 -> forecast_30d.csv")
print(out.head(10).to_string())
