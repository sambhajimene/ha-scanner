import streamlit as st
import pandas as pd
import datetime
import smtplib
import os
import json
from email.mime.text import MIMEText
from kiteconnect import KiteConnect
from streamlit_autorefresh import st_autorefresh

# ================= CONFIG =================

API_KEY = "z9rful06a9890v8m"
ACCESS_TOKEN = "5Fywm7ZbZ7PuK4MVWTYSXsmtQSlwLmzy"

# ================== Email Config ==================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_FROM = "sambhajimene@gmail.com"
EMAIL_PASSWORD = "jgebigpsoeqqwrfa"
EMAIL_TO = ["sambhajimene@gmail.com"]

BODY_THRESHOLD = 0.2

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

    ha["HA_Close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4

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


def send_signal_alert(symbols, signal_type="Bullish"):
    body = f"🔥 New {signal_type} HA Multi-Timeframe Signals:\n\n"
    for s in symbols:
        body += f"{s}\n"

    return send_email(f"🔥 {signal_type} HA Scanner Alert", body)


# ================= SIGNAL MEMORY =================
def load_previous_signals():
    if os.path.exists(SIGNAL_STORE_FILE):
        with open(SIGNAL_STORE_FILE, "r") as f:
            return json.load(f)
    return {"bullish": [], "bearish": []}


def save_signals(bullish, bearish):
    with open(SIGNAL_STORE_FILE, "w") as f:
        json.dump({"bullish": bullish, "bearish": bearish}, f)


# ================= SCANNER =================
def scan_market():

    instruments = kite.instruments("NSE")
    df_instruments = pd.DataFrame(instruments)

    nse500_symbols = pd.read_csv(
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    )["Symbol"].tolist()

    df_nse500 = df_instruments[
        (df_instruments["segment"] == "NSE") &
        (df_instruments["instrument_type"] == "EQ") &
        (df_instruments["tradingsymbol"].isin(nse500_symbols))
    ]

    bullish_results = []
    bearish_results = []

    for _, row in df_nse500.iterrows():

        symbol = row["tradingsymbol"]
        token = row["instrument_token"]

        try:
            # ================= DAILY =================
            daily = pd.DataFrame(
                kite.historical_data(token, START_DAILY, END_DATE, "day")
            )
            if len(daily) < 60:
                continue

            daily["EMA50"] = daily["close"].ewm(span=50).mean()
            ha_daily = calculate_heikin_ashi(daily)
            d = ha_daily.iloc[-1]

            body = abs(d["HA_Close"] - d["HA_Open"])
            full_range = d["HA_High"] - d["HA_Low"]
            if full_range == 0:
                continue

            body_ratio = body / full_range
            upper_wick = d["HA_High"] - max(d["HA_Open"], d["HA_Close"])
            lower_wick = min(d["HA_Open"], d["HA_Close"]) - d["HA_Low"]

            price_above_ema = d["close"] > d["EMA50"]
            price_below_ema = d["close"] < d["EMA50"]

            daily_neutral_bull = (
                body_ratio < BODY_THRESHOLD and
                upper_wick > 0 and
                lower_wick > 0 and
                price_above_ema
            )

            daily_strong_bull = (
                d["HA_Close"] > d["HA_Open"] and
                lower_wick <= (body * 0.1) and
                body_ratio > 0.5 and
                price_above_ema
            )

            daily_neutral_bear = (
                body_ratio < BODY_THRESHOLD and
                upper_wick > 0 and
                lower_wick > 0 and
                price_below_ema
            )

            daily_strong_bear = (
                d["HA_Close"] < d["HA_Open"] and
                upper_wick <= (body * 0.1) and
                body_ratio > 0.5 and
                price_below_ema
            )

            # ================= WEEKLY =================
            weekly = pd.DataFrame(
                kite.historical_data(token, START_WEEKLY, END_DATE, "week")
            )
            if len(weekly) < 10:
                continue

            ha_weekly = calculate_heikin_ashi(weekly)
            w = ha_weekly.iloc[-1]

            weekly_body = w["HA_Close"] - w["HA_Open"]
            weekly_upper_wick = w["HA_High"] - max(w["HA_Open"], w["HA_Close"])
            weekly_lower_wick = min(w["HA_Open"], w["HA_Close"]) - w["HA_Low"]

            weekly_strong_bull = (
                weekly_body > 0 and
                weekly_lower_wick <= (weekly_body * 0.1)
            )

            weekly_strong_bear = (
                weekly_body < 0 and
                weekly_upper_wick <= (abs(weekly_body) * 0.1)
            )

            # ================= HOURLY =================
            hourly = pd.DataFrame(
                kite.historical_data(token, START_HOURLY, END_DATE, "60minute")
            )
            if len(hourly) < 20:
                continue

            hourly["EMA50"] = hourly["close"].ewm(span=50).mean()

            ha_hourly = calculate_heikin_ashi(hourly)
            h = ha_hourly.iloc[-1]

            ema = hourly["EMA50"].iloc[-1]
            last = hourly.iloc[-1]

            hourly_body = h["HA_Close"] - h["HA_Open"]
            hourly_range = h["HA_High"] - h["HA_Low"]
            hourly_lower_wick = min(h["HA_Open"], h["HA_Close"]) - h["HA_Low"]
            hourly_upper_wick = h["HA_High"] - max(h["HA_Open"], h["HA_Close"])

            if hourly_range == 0:
                continue

            # ===== Bullish =====
            hourly_strong_bull = (
                hourly_body > 0 and
                hourly_lower_wick <= (hourly_body * 0.1) and
                (hourly_body / hourly_range) > 0.5
            )

            ema_support = (
                last["low"] <= ema * 1.002 and
                last["close"] > ema
            )

            failed_breakdown = (
                last["low"] < ema and
                last["close"] > ema
            )

            # ===== Bearish =====
            hourly_strong_bear = (
                hourly_body < 0 and
                hourly_upper_wick <= (abs(hourly_body) * 0.1) and
                (abs(hourly_body) / hourly_range) > 0.5
            )

            ema_resistance = (
                last["high"] >= ema * 0.998 and
                last["close"] < ema
            )

            failed_breakout = (
                last["high"] > ema and
                last["close"] < ema
            )

            # ===== Final Conditions =====

            if weekly_strong_bull and (daily_neutral_bull or daily_strong_bull) and hourly_strong_bull and (ema_support or failed_breakdown):
                bullish_results.append(symbol)

            if weekly_strong_bear and (daily_neutral_bear or daily_strong_bear) and hourly_strong_bear and (ema_resistance or failed_breakout):
                bearish_results.append(symbol)

        except:
            continue

    return bullish_results, bearish_results


# ================= STREAMLIT UI =================

st.set_page_config(layout="wide")
st.title("🔥 HA Scanner (Bullish + Bearish)")

auto_mode = st.sidebar.checkbox("Enable Auto Scan Every Hour")

if auto_mode:
    st_autorefresh(interval=3600000, limit=None, key="refresh")

run_scan = st.button("🚀 Run Scan Now")

if run_scan or auto_mode:

    with st.spinner("Scanning NSE500... Please wait..."):
        bullish, bearish = scan_market()

    prev = load_previous_signals()

    new_bull = list(set(bullish) - set(prev["bullish"]))
    new_bear = list(set(bearish) - set(prev["bearish"]))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 Bullish Signals")
        st.write(bullish)

    with col2:
        st.subheader("🔴 Bearish Signals")
        st.write(bearish)

    if new_bull:
        send_signal_alert(new_bull, "Bullish")

    if new_bear:
        send_signal_alert(new_bear, "Bearish")

    save_signals(bullish, bearish)

st.caption("Weekly Strong + Daily EMA Trend + Hourly Strong + EMA Reaction")
