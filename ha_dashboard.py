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
ACCESS_TOKEN = "IInPxgBD32aAoK3WdfiWXipAyXZzhjYJ"

# ================== Email Config ==================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_FROM = "sambhajimene@gmail.com"
EMAIL_PASSWORD = "jgebigpsoeqqwrfa"
EMAIL_TO = ["sambhajimene@gmail.com"]

BODY_THRESHOLD = 0.2
SLEEP_INTERVAL = 0.2

START_DAILY = datetime.date.today() - datetime.timedelta(days=120)
START_WEEKLY = datetime.date.today() - datetime.timedelta(days=365)
START_HOURLY = datetime.date.today() - datetime.timedelta(days=10)
END_DATE = datetime.date.today()

SIGNAL_STORE_FILE = "last_signals.json"

# ==========================================

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
# =================================================


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


def send_test_email():
    return send_email(
        "✅ HA Scanner Test Email",
        "Your HA Scanner email configuration is working properly."
    )


def send_signal_alert(symbols):
    body = "🔥 New HA Multi-Timeframe Signals Found:\n\n"
    for s in symbols:
        body += f"{s}\n"

    return send_email("🔥 HA Scanner Alert", body)
# =================================================


# ================= SIGNAL MEMORY =================
def load_previous_signals():
    if os.path.exists(SIGNAL_STORE_FILE):
        with open(SIGNAL_STORE_FILE, "r") as f:
            return json.load(f)
    return []


def save_signals(signals):
    with open(SIGNAL_STORE_FILE, "w") as f:
        json.dump(signals, f)
# =================================================


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

    results = []

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

            daily_neutral = (
                body_ratio < BODY_THRESHOLD and
                upper_wick > 0 and
                lower_wick > 0 and
                price_above_ema
            )

            daily_strong = (
                d["HA_Close"] > d["HA_Open"] and
                lower_wick <= (body * 0.1) and
                body_ratio > 0.5 and
                price_above_ema
            )

            if not (daily_neutral or daily_strong):
                continue

            # ================= WEEKLY =================
            weekly = pd.DataFrame(
                kite.historical_data(token, START_WEEKLY, END_DATE, "week")
            )
            if len(weekly) < 10:
                continue

            ha_weekly = calculate_heikin_ashi(weekly)
            w = ha_weekly.iloc[-1]

            weekly_body = w["HA_Close"] - w["HA_Open"]
            weekly_lower_wick = min(w["HA_Open"], w["HA_Close"]) - w["HA_Low"]

            weekly_strong = (
                weekly_body > 0 and
                weekly_lower_wick <= (weekly_body * 0.1)
            )

            if not weekly_strong:
                continue

            # ================= HOURLY =================
            hourly = pd.DataFrame(
                kite.historical_data(token, START_HOURLY, END_DATE, "60minute")
            )
            if len(hourly) < 20:
                continue

            ha_hourly = calculate_heikin_ashi(hourly)
            h = ha_hourly.iloc[-1]

            hourly_body = h["HA_Close"] - h["HA_Open"]
            hourly_lower_wick = min(h["HA_Open"], h["HA_Close"]) - h["HA_Low"]
            hourly_range = h["HA_High"] - h["HA_Low"]

            if hourly_range == 0:
                continue

            hourly_strong = (
                hourly_body > 0 and
                hourly_lower_wick <= (hourly_body * 0.1) and
                (hourly_body / hourly_range) > 0.5
            )

            if hourly_strong:
                results.append(symbol)

        except:
            continue

    return results
# =================================================


# ================= STREAMLIT UI =================

st.set_page_config(layout="wide")
st.title("🔥 HA Scanner")

st.sidebar.title("⚙ Settings")
auto_mode = st.sidebar.checkbox("Enable Auto Scan Every Hour")

if auto_mode:
    st_autorefresh(interval=3600000, limit=None, key="refresh")

col1, col2 = st.columns(2)

run_scan = col1.button("🚀 Run Scan Now")
test_mail = col2.button("📧 Send Test Mail")

if test_mail:
    result = send_test_email()
    if result == True:
        st.success("Test Email Sent Successfully!")
    else:
        st.error(result)

if run_scan or auto_mode:

    with st.spinner("Scanning NSE500... Please wait..."):
        current_signals = scan_market()

    previous_signals = load_previous_signals()
    new_signals = list(set(current_signals) - set(previous_signals))

    if current_signals:
        st.success(f"Found {len(current_signals)} Signals")
        st.write(current_signals)
    else:
        st.warning("No setups found.")

    if new_signals:
        email_status = send_signal_alert(new_signals)
        if email_status == True:
            st.success(f"📬 Email Alert Sent for {len(new_signals)} New Signals")
        else:
            st.error(email_status)

    save_signals(current_signals)

st.markdown("---")
st.caption("Weekly Strong + Daily (Neutral or Strong) Above EMA50 + Hourly Strong")

