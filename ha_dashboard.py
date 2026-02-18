import streamlit as st
import pandas as pd
import datetime
import smtplib
import os
import json
import time
from email.mime.text import MIMEText
from kiteconnect import KiteConnect
from concurrent.futures import ThreadPoolExecutor, as_completed
from streamlit_autorefresh import st_autorefresh

# ===== SAFE PLOTLY IMPORT =====
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False


# ================= CONFIG =================

API_KEY = "z9rful06a9890v8m"
ACCESS_TOKEN = "60PRJS0GYlhAs05Ki8Hx68JtvxQF79Is"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_FROM = "sambhajimene@gmail.com"
EMAIL_PASSWORD = "jgebigpsoeqqwrfa"
EMAIL_TO = ["sambhajimene@gmail.com"]

BODY_THRESHOLD = 0.2
MAX_WORKERS = 8

START_DAILY = datetime.date.today() - datetime.timedelta(days=120)
START_WEEKLY = datetime.date.today() - datetime.timedelta(days=365)
START_HOURLY = datetime.date.today() - datetime.timedelta(days=10)
END_DATE = datetime.date.today()

SIGNAL_STORE_FILE = "last_signals.json"

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)


# ================= HEIKIN ASHI =================
def calculate_heikin_ashi(df):

    ha = df.copy()

    ha["HA_Close"] = (df["open"] + df["high"] +
                      df["low"] + df["close"]) / 4

    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2]

    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha["HA_Close"].iloc[i-1]) / 2)

    ha["HA_Open"] = ha_open
    ha["HA_High"] = ha[["high", "HA_Open", "HA_Close"]].max(axis=1)
    ha["HA_Low"] = ha[["low", "HA_Open", "HA_Close"]].min(axis=1)

    return ha


# ================= EMAIL =================
def send_email(subject, body):

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(EMAIL_TO)

    try:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        return str(e)


def send_signal_alert(bullish, bearish):

    body = "🔥 HA Signals\n\n"

    if bullish:
        body += "🟢 Bullish:\n" + "\n".join(bullish) + "\n\n"

    if bearish:
        body += "🔴 Bearish:\n" + "\n".join(bearish)

    return send_email("🔥 HA Scanner Alert", body)


# ================= MEMORY =================
def load_previous_signals():

    if os.path.exists(SIGNAL_STORE_FILE):
        try:
            with open(SIGNAL_STORE_FILE, "r") as f:
                data = json.load(f)
            return data
        except:
            pass

    return {"bullish": [], "bearish": []}


def save_signals(bullish, bearish):
    with open(SIGNAL_STORE_FILE, "w") as f:
        json.dump({"bullish": bullish, "bearish": bearish}, f)


# ================= SINGLE STOCK SCAN =================
def scan_symbol(row):

    symbol = row["tradingsymbol"]
    token = row["instrument_token"]

    try:

        # ================= DAILY =================
        daily = pd.DataFrame(
            kite.historical_data(token, START_DAILY, END_DATE, "day")
        )
        if len(daily) < 60:
            return None

        daily["EMA50"] = daily["close"].ewm(span=50).mean()
        ha_daily = calculate_heikin_ashi(daily)
        d = ha_daily.iloc[-1]

        body = abs(d["HA_Close"] - d["HA_Open"])
        full_range = d["HA_High"] - d["HA_Low"]

        if full_range == 0:
            return None

        body_ratio = body / full_range
        upper_wick = d["HA_High"] - max(d["HA_Open"], d["HA_Close"])
        lower_wick = min(d["HA_Open"], d["HA_Close"]) - d["HA_Low"]

        price_above = d["close"] > d["EMA50"]
        price_below = d["close"] < d["EMA50"]

        # OLD STRICT LOGIC
        daily_neutral_bull = (
            body_ratio < BODY_THRESHOLD and
            upper_wick > 0 and
            lower_wick > 0 and
            price_above
        )

        daily_strong_bull = (
            d["HA_Close"] > d["HA_Open"] and
            lower_wick <= (body * 0.1) and
            body_ratio > 0.5 and
            price_above
        )

        daily_neutral_bear = (
            body_ratio < BODY_THRESHOLD and
            upper_wick > 0 and
            lower_wick > 0 and
            price_below
        )

        daily_strong_bear = (
            d["HA_Close"] < d["HA_Open"] and
            upper_wick <= (body * 0.1) and
            body_ratio > 0.5 and
            price_below
        )

        # ================= WEEKLY =================
        weekly = pd.DataFrame(
            kite.historical_data(token, START_WEEKLY, END_DATE, "week")
        )
        if len(weekly) < 10:
            return None

        ha_weekly = calculate_heikin_ashi(weekly)
        w = ha_weekly.iloc[-1]

        weekly_body = w["HA_Close"] - w["HA_Open"]
        weekly_upper = w["HA_High"] - max(w["HA_Open"], w["HA_Close"])
        weekly_lower = min(w["HA_Open"], w["HA_Close"]) - w["HA_Low"]

        weekly_bull = weekly_body > 0 and weekly_lower <= (weekly_body * 0.1)
        weekly_bear = weekly_body < 0 and weekly_upper <= (abs(weekly_body) * 0.1)

        # ================= HOURLY =================
        hourly = pd.DataFrame(
            kite.historical_data(token, START_HOURLY, END_DATE, "60minute")
        )
        if len(hourly) < 20:
            return None

        hourly["EMA50"] = hourly["close"].ewm(span=50).mean()
        ha_hourly = calculate_heikin_ashi(hourly)
        h = ha_hourly.iloc[-1]

        ema = hourly["EMA50"].iloc[-1]
        last = hourly.iloc[-1]

        body_h = abs(h["HA_Close"] - h["HA_Open"])
        range_h = h["HA_High"] - h["HA_Low"]

        if range_h == 0:
            return None

        ratio_h = body_h / range_h

        hourly_bull = (
            h["HA_Close"] > h["HA_Open"] and
            ratio_h > 0.5
        )

        hourly_bear = (
            h["HA_Close"] < h["HA_Open"] and
            ratio_h > 0.5
        )

        # EMA reactions
        ema_support = last["low"] <= ema * 1.002 and last["close"] > ema
        failed_breakdown = last["low"] < ema and last["close"] > ema

        ema_resistance = last["high"] >= ema * 0.998 and last["close"] < ema
        failed_breakout = last["high"] > ema and last["close"] < ema

        time.sleep(0.05)

        # ================= FINAL CONDITIONS =================

        if weekly_bull and (daily_neutral_bull or daily_strong_bull) \
                and hourly_bull and (ema_support or failed_breakdown):
            return ("bullish", symbol)

        if weekly_bear and (daily_neutral_bear or daily_strong_bear) \
                and hourly_bear and (ema_resistance or failed_breakout):
            return ("bearish", symbol)

    except:
        return None

    return None


# ================= PARALLEL SCAN =================
def scan_market():

    instruments = kite.instruments("NSE")
    df = pd.DataFrame(instruments)

    nse500 = pd.read_csv(
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    )["Symbol"].tolist()

    df = df[
        (df["segment"] == "NSE") &
        (df["instrument_type"] == "EQ") &
        (df["tradingsymbol"].isin(nse500))
    ]

    bullish = []
    bearish = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [executor.submit(scan_symbol, row)
                   for _, row in df.iterrows()]

        for f in as_completed(futures):
            res = f.result()
            if res:
                side, sym = res
                if side == "bullish":
                    bullish.append(sym)
                else:
                    bearish.append(sym)

    return bullish, bearish


# ================= UI =================

st.set_page_config(layout="wide")
st.title("🔥 HA Strict Hybrid Scanner")

auto_mode = st.sidebar.checkbox("Auto Scan Hourly")

if auto_mode:
    st_autorefresh(interval=3600000, limit=None)

run_scan = st.button("🚀 Run Scan")

if run_scan or auto_mode:

    with st.spinner("Scanning market..."):
        bullish, bearish = scan_market()

    prev = load_previous_signals()

    new_bull = list(set(bullish) - set(prev["bullish"]))
    new_bear = list(set(bearish) - set(prev["bearish"]))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 Bullish")
        st.write(bullish)

    with col2:
        st.subheader("🔴 Bearish")
        st.write(bearish)

    if new_bull or new_bear:
        send_signal_alert(new_bull, new_bear)
        st.success("📧 Alert Sent")

    save_signals(bullish, bearish)
