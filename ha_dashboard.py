import pandas as pd
import numpy as np
import datetime as dt
import json
import streamlit as st
from kiteconnect import KiteConnect
import time
from nse500_symbols import nse500_symbols

# -------------------------
# KiteConnect setup
# -------------------------
KITE_API_KEY = "z9rful06a9890v8m"
#KITE_API_SECRET = "YOUR_API_SECRET"
KITE_ACCESS_TOKEN = "X78rnH2NAuTJvEfblvtVShawi4ygf2W9"

kite = KiteConnect(api_key=KITE_API_KEY)
kite.set_access_token(KITE_ACCESS_TOKEN)

SIGNAL_STORE_FILE = "/app/live_ha_signals.json"  # Docker friendly
signal_store = []

# -------------------------
# Indicator functions
# -------------------------
def heikin_ashi(df):
    ha = df.copy()
    ha["HA_Close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha["HA_Open"] = (df["open"].shift(1) + df["close"].shift(1)) / 2
    ha["HA_High"] = ha[["HA_Open","HA_Close", "high"]].max(axis=1)
    ha["HA_Low"] = ha[["HA_Open","HA_Close", "low"]].min(axis=1)
    ha.fillna(method="bfill", inplace=True)
    return ha

def EMA(df, period=50):
    return df["close"].ewm(span=period, adjust=False).mean()

def RSI(df, period=14):
    delta = df["close"].diff()
    gain = np.where(delta>0, delta, 0)
    loss = np.where(delta<0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain/avg_loss
    return 100-(100/(1+rs))

def ADX(df, period=14):
    df["TR"] = np.maximum.reduce([
        df["high"]-df["low"],
        abs(df["high"]-df["close"].shift()),
        abs(df["low"]-df["close"].shift())
    ])
    df["+DM"] = np.where(df["high"].diff()>df["low"].diff(), np.maximum(df["high"].diff(),0), 0)
    df["-DM"] = np.where(df["low"].diff()>df["high"].diff(), np.maximum(df["low"].diff(),0),0)
    TR_sma = df["TR"].rolling(period).mean()
    plus_di = 100*(df["+DM"].rolling(period).mean()/TR_sma)
    minus_di = 100*(df["-DM"].rolling(period).mean()/TR_sma)
    dx = 100*abs(plus_di-minus_di)/(plus_di+minus_di)
    return dx.rolling(period).mean()

def strong_ha_bullish(row):
    return row["HA_Open"]==row["HA_Low"] and row["HA_Close"]>row["HA_Open"]

def strong_ha_bearish(row):
    return row["HA_Open"]==row["HA_High"] and row["HA_Close"]<row["HA_Open"]

# -------------------------
# Fetch OHLC from Kite
# -------------------------
def get_ohlc(kite_symbol, interval, from_date, to_date):
    data = kite.historical_data(kite_symbol, from_date, to_date, interval)
    df = pd.DataFrame(data)
    df.rename(columns={"date":"Datetime"}, inplace=True)
    df.set_index("Datetime", inplace=True)
    return df

# -------------------------
# Scan single symbol
# -------------------------
def scan_symbol(symbol):
    try:
        today = dt.date.today()
        wk_ago = today - dt.timedelta(days=365*3)
        df_weekly = get_ohlc(f"NSE:{symbol}", "week", wk_ago, today)
        df_daily = get_ohlc(f"NSE:{symbol}", "day", wk_ago, today)
        df_hourly = get_ohlc(f"NSE:{symbol}", "hour", today-dt.timedelta(days=30), today)

        ha_w = heikin_ashi(df_weekly)
        ha_d = heikin_ashi(df_daily)
        ha_h = heikin_ashi(df_hourly)

        ha_d["EMA50"] = EMA(ha_d)
        ha_d["EMA50_slope"] = ha_d["EMA50"].diff()
        ha_h["EMA50"] = EMA(ha_h)
        ha_h["RSI"] = RSI(ha_h)
        ha_h["ADX"] = ADX(ha_h)

        signals = []

        week_bull = strong_ha_bullish(ha_w.iloc[-1])
        week_bear = strong_ha_bearish(ha_w.iloc[-1])
        daily_bull = strong_ha_bullish(ha_d.iloc[-1]) or ha_d["HA_Close"].iloc[-1]>=ha_d["HA_Open"].iloc[-1]
        daily_bear = strong_ha_bearish(ha_d.iloc[-1]) or ha_d["HA_Close"].iloc[-1]<=ha_d["HA_Open"].iloc[-1]
        daily_ema_rising = ha_d["EMA50_slope"].iloc[-1]>0

        price = ha_h["close"].iloc[-1]
        ema50 = ha_h["EMA50"].iloc[-1]
        rsi = ha_h["RSI"].iloc[-1]
        adx = ha_h["ADX"].iloc[-1]

        bullish_rsi = (40<=rsi<=60) or (rsi>60)
        bearish_rsi = (40<=rsi<=60) or (rsi<40)
        support = abs(price-ema50)<=0.5*ema50/100
        resistance = abs(price-ema50)<=0.5*ema50/100

        if week_bull and daily_bull and daily_ema_rising and support and price>ema50 and bullish_rsi and adx>adx:
            signals.append({"symbol":symbol,"signal":"Bullish","time":dt.datetime.now().isoformat()})
        if week_bear and daily_bear and daily_ema_rising and resistance and price<ema50 and bearish_rsi and adx>adx:
            signals.append({"symbol":symbol,"signal":"Bearish","time":dt.datetime.now().isoformat()})

        return signals
    except Exception as e:
        print(f"Error {symbol}: {e}")
        return []

# -------------------------
# Auto NSE500 scan
# -------------------------
st.title("📊 AUTO NSE500 Heikin-Ashi Scanner")
if st.button("Run Full Auto Scan"):
    global signal_store
    for sym in nse500_symbols:
        sigs = scan_symbol(sym)
        signal_store.extend(sigs)
        st.write(f"{sym}: {len(sigs)} signals")
        time.sleep(0.5)  # Rate-limit safe

    with open(SIGNAL_STORE_FILE,"w") as f:
        json.dump(signal_store,f,indent=4)

    st.success(f"Scan completed! Signals saved in {SIGNAL_STORE_FILE}")
    if signal_store:
        st.dataframe(pd.DataFrame(signal_store))
