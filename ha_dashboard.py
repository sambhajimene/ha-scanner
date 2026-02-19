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
ACCESS_TOKEN = "3d6BWxuk9IuP4fvIAo3q06IE4EGavzCs"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_FROM = "sambhajimene@gmail.com"
EMAIL_PASSWORD = "jgebigpsoeqqwrfa"
EMAIL_TO = ["sambhajimene@gmail.com"]

BODY_THRESHOLD = 0.2
MAX_WORKERS = 30          # ✅ FIX 1: Increased from 8 → 30 (I/O bound, safe for Zerodha)
MIN_VOLUME = 100_000      # ✅ FIX 2: Pre-filter threshold

START_DAILY  = datetime.date.today() - datetime.timedelta(days=120)
START_WEEKLY = datetime.date.today() - datetime.timedelta(days=365)
START_HOURLY = datetime.date.today() - datetime.timedelta(days=10)
END_DATE     = datetime.date.today()

SIGNAL_STORE_FILE = "last_signals.json"

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)


# ================= HEIKIN ASHI =================
def calculate_heikin_ashi(df):
    ha = df.copy()
    ha["HA_Close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i - 1] + ha["HA_Close"].iloc[i - 1]) / 2)
    ha["HA_Open"]  = ha_open
    ha["HA_High"]  = ha[["high", "HA_Open", "HA_Close"]].max(axis=1)
    ha["HA_Low"]   = ha[["low",  "HA_Open", "HA_Close"]].min(axis=1)
    return ha


# ================= EMAIL =================
def send_email(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = EMAIL_FROM
    msg["To"]      = ", ".join(EMAIL_TO)
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
                return json.load(f)
        except:
            pass
    return {"bullish": [], "bearish": []}


def save_signals(bullish, bearish):
    with open(SIGNAL_STORE_FILE, "w") as f:
        json.dump({"bullish": bullish, "bearish": bearish}, f)


# ================= PRE-FILTER BY VOLUME =================
def pre_filter_by_volume(df):
    """
    ✅ FIX 3: Single kite.quote() call for all 500 stocks at once.
    Drops low-volume / illiquid stocks before making 3x historical API calls.
    Typically cuts scan list by 30–40%.
    """
    symbols = df["tradingsymbol"].tolist()

    # Zerodha quote API accepts max ~500 instruments per call
    nse_symbols = ["NSE:" + s for s in symbols]
    try:
        quotes = kite.quote(nse_symbols)
    except Exception as e:
        st.warning(f"Volume pre-filter failed ({e}), scanning all symbols.")
        return df

    active_symbols = set()
    for sym in symbols:
        q = quotes.get("NSE:" + sym, {})
        vol = q.get("volume", 0)
        if vol >= MIN_VOLUME:
            active_symbols.add(sym)

    filtered = df[df["tradingsymbol"].isin(active_symbols)].reset_index(drop=True)
    return filtered


# ================= SINGLE STOCK SCAN =================
def scan_symbol(row):
    symbol = row["tradingsymbol"]
    token  = row["instrument_token"]

    try:
        # ── DAILY ──────────────────────────────────────────────────────────
        daily = pd.DataFrame(kite.historical_data(token, START_DAILY, END_DATE, "day"))
        if len(daily) < 60:
            return None

        daily["EMA50"] = daily["close"].ewm(span=50).mean()
        ha_daily = calculate_heikin_ashi(daily)
        d = ha_daily.iloc[-1]

        body       = abs(d["HA_Close"] - d["HA_Open"])
        full_range = d["HA_High"] - d["HA_Low"]
        if full_range == 0:
            return None

        body_ratio  = body / full_range
        upper_wick  = d["HA_High"] - max(d["HA_Open"], d["HA_Close"])
        lower_wick  = min(d["HA_Open"], d["HA_Close"]) - d["HA_Low"]
        price_above = d["close"] > d["EMA50"]
        price_below = d["close"] < d["EMA50"]

        daily_neutral_bull = (body_ratio < BODY_THRESHOLD and upper_wick > 0
                              and lower_wick > 0 and price_above)
        daily_strong_bull  = (d["HA_Close"] > d["HA_Open"]
                              and lower_wick <= body * 0.1
                              and body_ratio > 0.5 and price_above)
        daily_neutral_bear = (body_ratio < BODY_THRESHOLD and upper_wick > 0
                              and lower_wick > 0 and price_below)
        daily_strong_bear  = (d["HA_Close"] < d["HA_Open"]
                              and upper_wick <= body * 0.1
                              and body_ratio > 0.5 and price_below)

        # Early exit — skip weekly + hourly API calls if daily doesn't qualify
        if not (daily_neutral_bull or daily_strong_bull or
                daily_neutral_bear or daily_strong_bear):
            return None

        # ── WEEKLY ─────────────────────────────────────────────────────────
        weekly = pd.DataFrame(kite.historical_data(token, START_WEEKLY, END_DATE, "week"))
        if len(weekly) < 10:
            return None

        ha_weekly    = calculate_heikin_ashi(weekly)
        w            = ha_weekly.iloc[-1]
        weekly_body  = w["HA_Close"] - w["HA_Open"]
        weekly_upper = w["HA_High"] - max(w["HA_Open"], w["HA_Close"])
        weekly_lower = min(w["HA_Open"], w["HA_Close"]) - w["HA_Low"]

        weekly_bull  = weekly_body > 0 and weekly_lower <= weekly_body * 0.1
        weekly_bear  = weekly_body < 0 and weekly_upper <= abs(weekly_body) * 0.1

        # Early exit — skip hourly if weekly doesn't qualify
        if not (weekly_bull or weekly_bear):
            return None

        # ── HOURLY ─────────────────────────────────────────────────────────
        hourly = pd.DataFrame(kite.historical_data(token, START_HOURLY, END_DATE, "60minute"))
        if len(hourly) < 20:
            return None

        hourly["EMA50"] = hourly["close"].ewm(span=50).mean()
        ha_hourly = calculate_heikin_ashi(hourly)
        h         = ha_hourly.iloc[-1]
        ema       = hourly["EMA50"].iloc[-1]
        last      = hourly.iloc[-1]

        body_h  = abs(h["HA_Close"] - h["HA_Open"])
        range_h = h["HA_High"] - h["HA_Low"]
        if range_h == 0:
            return None

        ratio_h      = body_h / range_h
        hourly_bull  = h["HA_Close"] > h["HA_Open"] and ratio_h > 0.5
        hourly_bear  = h["HA_Close"] < h["HA_Open"] and ratio_h > 0.5

        ema_support       = last["low"]  <= ema * 1.002 and last["close"] > ema
        failed_breakdown  = last["low"]  <  ema         and last["close"] > ema
        ema_resistance    = last["high"] >= ema * 0.998 and last["close"] < ema
        failed_breakout   = last["high"] >  ema         and last["close"] < ema

        # ✅ FIX 4: Removed time.sleep() — was adding 25+ sec of dead time
        # time.sleep(0.05)   ← REMOVED

        # ── FINAL SIGNAL ───────────────────────────────────────────────────
        if weekly_bull and (daily_neutral_bull or daily_strong_bull) \
                and hourly_bull and (ema_support or failed_breakdown):
            return ("bullish", symbol)

        if weekly_bear and (daily_neutral_bear or daily_strong_bear) \
                and hourly_bear and (ema_resistance or failed_breakout):
            return ("bearish", symbol)

    except:
        return None

    return None


# ================= PARALLEL SCAN WITH PROGRESS BAR =================
def scan_market(progress_bar, status_text, stats_cols):
    instruments = kite.instruments("NSE")
    df = pd.DataFrame(instruments)

    nse500 = pd.read_csv(
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    )["Symbol"].tolist()

    df = df[
        (df["segment"] == "NSE") &
        (df["instrument_type"] == "EQ") &
        (df["tradingsymbol"].isin(nse500))
    ].reset_index(drop=True)

    # ── STEP 1: Volume pre-filter ─────────────────────────────────────────
    status_text.markdown("**Step 1/2 — Volume pre-filter** (single API call)…")
    before = len(df)
    df = pre_filter_by_volume(df)
    after  = len(df)
    status_text.markdown(
        f"**Step 1/2 — Done ✅** &nbsp; {before} → **{after}** stocks after volume filter "
        f"({before - after} removed)"
    )

    bullish  = []
    bearish  = []
    skipped  = 0
    total    = len(df)

    # ── STEP 2: Parallel historical scan ─────────────────────────────────
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(scan_symbol, row): row["tradingsymbol"]
            for _, row in df.iterrows()
        }

        done = 0
        for f in as_completed(futures):
            res = f.result()
            done += 1

            if res is None:
                skipped += 1
            else:
                side, sym = res
                if side == "bullish":
                    bullish.append(sym)
                else:
                    bearish.append(sym)

            # ✅ FIX 5: Live progress bar + counters
            pct = done / total
            progress_bar.progress(pct)
            status_text.markdown(
                f"**Step 2/2 — Scanning** &nbsp; `{done}/{total}` stocks scanned &nbsp;|&nbsp; "
                f"🟢 Bullish: **{len(bullish)}** &nbsp;|&nbsp; "
                f"🔴 Bearish: **{len(bearish)}** &nbsp;|&nbsp; "
                f"⏭ Skipped: **{skipped}**"
            )

            # Live update signal columns every 25 stocks
            if done % 25 == 0 or done == total:
                with stats_cols[0]:
                    st.markdown("### 🟢 Bullish")
                    st.write(bullish if bullish else "_None yet_")
                with stats_cols[1]:
                    st.markdown("### 🔴 Bearish")
                    st.write(bearish if bearish else "_None yet_")

    return bullish, bearish


# ================= UI =================

st.set_page_config(page_title="HA Strict Hybrid Scanner", layout="wide")

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2rem; margin-bottom:0'>
    🔥 HA Strict Hybrid Scanner
</h1>
<p style='color:gray; margin-top:4px'>
    Nifty 500 · Daily + Weekly + Hourly · EMA50 · Volume Pre-filter
</p>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    auto_mode   = st.checkbox("🔄 Auto Scan (hourly)", value=False)
    min_vol_ui  = st.number_input("Min Volume Filter", value=MIN_VOLUME, step=50_000)
    workers_ui  = st.number_input("Parallel Workers",  value=MAX_WORKERS, step=5, min_value=5, max_value=50)
    st.divider()
    st.caption(f"Workers: {workers_ui} · Min Vol: {min_vol_ui:,}")

MIN_VOLUME  = min_vol_ui
MAX_WORKERS = workers_ui

if auto_mode:
    st_autorefresh(interval=3_600_000, limit=None)

# ── Scan button ───────────────────────────────────────────────────────────
run_scan = st.button("🚀 Run Scan", type="primary", use_container_width=False)

if run_scan or auto_mode:

    st.markdown("---")

    # Persistent placeholders so they update in place
    progress_bar = st.progress(0)
    status_text  = st.empty()

    # Live signal columns
    stats_cols = st.columns(2)
    with stats_cols[0]:
        st.markdown("### 🟢 Bullish")
        st.write("_Scanning…_")
    with stats_cols[1]:
        st.markdown("### 🔴 Bearish")
        st.write("_Scanning…_")

    start_time           = time.time()
    bullish, bearish     = scan_market(progress_bar, status_text, stats_cols)
    elapsed              = time.time() - start_time

    # ── Final results ─────────────────────────────────────────────────────
    progress_bar.progress(1.0)
    status_text.success(f"✅ Scan complete in {elapsed:.1f}s — "
                        f"{len(bullish)} Bullish · {len(bearish)} Bearish")

    # Final signal display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🟢 Bullish Signals")
        if bullish:
            for sym in sorted(bullish):
                st.markdown(f"- `{sym}`")
        else:
            st.info("No bullish signals found.")

    with col2:
        st.subheader("🔴 Bearish Signals")
        if bearish:
            for sym in sorted(bearish):
                st.markdown(f"- `{sym}`")
        else:
            st.info("No bearish signals found.")

    # ── New signal email alert ────────────────────────────────────────────
    prev     = load_previous_signals()
    new_bull = list(set(bullish) - set(prev["bullish"]))
    new_bear = list(set(bearish) - set(prev["bearish"]))

    if new_bull or new_bear:
        result = send_signal_alert(new_bull, new_bear)
        if result is True:
            st.success(f"📧 Alert sent — {len(new_bull)} new bullish, {len(new_bear)} new bearish")
        else:
            st.error(f"📧 Email failed: {result}")
    else:
        st.info("No new signals since last scan — email not sent.")

    save_signals(bullish, bearish)

    # ── Summary metrics ───────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🟢 Bullish",     len(bullish))
    m2.metric("🔴 Bearish",     len(bearish))
    m3.metric("🆕 New Signals", len(new_bull) + len(new_bear))
    m4.metric("⏱ Scan Time",   f"{elapsed:.1f}s")
