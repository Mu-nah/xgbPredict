import os
import time
import requests
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# ── ENV ──
load_dotenv()
TD_KEYS   = [k.strip() for k in os.getenv("TD_API_KEYS", "").split(",") if k.strip()]
NEWS_KEYS = [k.strip() for k in os.getenv("NEWS_API_KEYS", "").split(",") if k.strip()]
WEBHOOK_URL = "http://localhost:5678/webhook-test/receive-post"

# ── API fetch with key‑rotation ──
def fetch(symbol: str, interval: str, limit: int = 120):
    url = "https://api.twelvedata.com/time_series"
    for key in TD_KEYS:
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": interval, "outputsize": limit, "apikey": key}, timeout=15)
            if r.status_code != 200:
                continue
            j = r.json()
            if "values" not in j:
                continue
            df = pd.DataFrame(j["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            return df.astype(float).sort_index()
        except:
            time.sleep(1)
    return pd.DataFrame()

# ── TA Features ──
def add_ta(df: pd.DataFrame):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["rsi"] = RSIIndicator(df["close"], window=7).rsi()
    df["macd"] = MACD(df["close"]).macd_diff()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], window=3).average_true_range()
    df["hour"] = df.index.hour
    df["dow"]  = df.index.dayofweek
    return df.dropna()

# ── Sentiment ──
vader = SentimentIntensityAnalyzer()

def sentiment_score(sym: str):
    for key in NEWS_KEYS:
        try:
            r = requests.get("https://newsapi.org/v2/everything", params={"q": sym.split("/")[0], "language": "en", "pageSize": 10, "apiKey": key}, timeout=8).json()
            if "articles" in r:
                vals = [vader.polarity_scores(a["title"])['compound'] for a in r["articles"]]
                if vals:
                    return float(np.mean(vals))
        except:
            continue
    return 0.0

# ── Predict Function ──
def predict(symbol: str):
    df1h = add_ta(fetch(symbol, "1h"))
    df2h = add_ta(fetch(symbol, "2h"))
    df30 = add_ta(fetch(symbol, "30min", 60))

    if df1h.empty or df2h.empty or df30.empty:
        return None

    df1h, df2h = df1h.align(df2h, join="inner", axis=0)
    df1h, df2h = df1h.iloc[:-1], df2h.iloc[:-1]

    y = (df1h["close"].shift(-1) > df1h["close"]).astype(int)[:-1]

    X = pd.concat([
        df1h[["ret","rsi","macd","atr","hour","dow"]].add_suffix("_1h"),
        df2h[["ret","rsi","macd","atr","hour","dow"]].add_suffix("_2h")
    ], axis=1)

    X, y = X.align(y, join="inner", axis=0)

    if len(X) < 20:
        return None

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))

    x_pred = pd.concat([
        df1h[["ret","rsi","macd","atr","hour","dow"]].iloc[[-1]].add_suffix("_1h"),
        df2h[["ret","rsi","macd","atr","hour","dow"]].iloc[[-1]].add_suffix("_2h")
    ], axis=1)

    prob = model.predict_proba(x_pred)[0,1]
    decision = "Buy 📈" if prob > 0.5 else "Sell 📉"
    price = float(df30["close"].iloc[-1])

    return {
        "symbol": symbol,
        "live_price": round(price,2),
        "prediction": decision,
        "confidence": round(prob*100,2),
        "predicted_price": round(price*(1+prob if prob>0.5 else 1-prob),2),
        "accuracy": round(acc*100,2),
        "sentiment": round(sentiment_score(symbol),3)
    }

# ── n8n push ──
def push_n8n(payload):
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        print(f"n8n status {r.status_code}")
    except Exception as e:
        print(f"n8n push failed: {e}")

# ── Run without Streamlit ──
if __name__ == "__main__":
    btc = predict("BTC/USD")
    xau = predict("XAU/USD")

    if btc and xau:
        push_n8n({"predictions":[btc,xau]})
    else:
        print("Prediction failed for one or more symbols.")
