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

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except:
    PLOTLY_AVAILABLE = False


# ================= CONFIG =================

API_KEY      = "z9rful06a9890v8m"
ACCESS_TOKEN = "3d6BWxuk9IuP4fvIAo3q06IE4EGavzCs"

SMTP_SERVER    = "smtp.gmail.com"
SMTP_PORT      = 465
EMAIL_FROM     = "sambhajimene@gmail.com"
EMAIL_PASSWORD = "jgebigpsoeqqwrfa"
EMAIL_TO       = ["sambhajimene@gmail.com"]

BODY_THRESHOLD = 0.2
MAX_WORKERS    = 30
MIN_VOLUME     = 100_000

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
    ha["HA_Open"] = ha_open
    ha["HA_High"] = ha[["high", "HA_Open", "HA_Close"]].max(axis=1)
    ha["HA_Low"]  = ha[["low",  "HA_Open", "HA_Close"]].min(axis=1)
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
def pre_filter_by_volume(df, status_text):
    symbols     = df["tradingsymbol"].tolist()
    nse_symbols = ["NSE:" + s for s in symbols]
    all_quotes  = {}

    # Chunk into 500 per call (Zerodha limit)
    for i in range(0, len(nse_symbols), 500):
        chunk = nse_symbols[i:i + 500]
        try:
            q = kite.quote(chunk)
            all_quotes.update(q)
        except Exception as e:
            st.warning(f"Quote chunk failed: {e}")

    active_symbols = set()
    for sym in symbols:
        q   = all_quotes.get("NSE:" + sym, {})
        vol = q.get("volume", 0)
        if vol >= MIN_VOLUME:
            active_symbols.add(sym)

    return df[df["tradingsymbol"].isin(active_symbols)].reset_index(drop=True)


# ================= SINGLE STOCK SCAN =================
def scan_symbol(row, debug_mode=False):
    symbol = row["tradingsymbol"]
    token  = row["instrument_token"]
    log    = []

    try:
        # ── DAILY ──────────────────────────────────────────────────────────
        daily = pd.DataFrame(kite.historical_data(token, START_DAILY, END_DATE, "day"))
        if len(daily) < 60:
            return None, f"{symbol}: insufficient daily data ({len(daily)} bars)"

        daily["EMA50"] = daily["close"].ewm(span=50).mean()
        ha_daily       = calculate_heikin_ashi(daily)
        d              = ha_daily.iloc[-1]

        body       = abs(d["HA_Close"] - d["HA_Open"])
        full_range = d["HA_High"] - d["HA_Low"]
        if full_range == 0:
            return None, f"{symbol}: zero daily range"

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

        daily_bull_ok = daily_neutral_bull or daily_strong_bull
        daily_bear_ok = daily_neutral_bear or daily_strong_bear

        if debug_mode:
            log.append(
                f"{symbol} | Daily: body_ratio={body_ratio:.2f}, price_above={price_above}, "
                f"price_below={price_below}, neutral_bull={daily_neutral_bull}, "
                f"strong_bull={daily_strong_bull}, neutral_bear={daily_neutral_bear}, "
                f"strong_bear={daily_strong_bear}"
            )

        if not (daily_bull_ok or daily_bear_ok):
            return None, "\n".join(log) if log else f"{symbol}: failed daily filter"

        # ── WEEKLY ─────────────────────────────────────────────────────────
        weekly = pd.DataFrame(kite.historical_data(token, START_WEEKLY, END_DATE, "week"))
        if len(weekly) < 10:
            return None, f"{symbol}: insufficient weekly data ({len(weekly)} bars)"

        ha_weekly    = calculate_heikin_ashi(weekly)
        w            = ha_weekly.iloc[-1]
        weekly_body  = w["HA_Close"] - w["HA_Open"]
        weekly_upper = w["HA_High"] - max(w["HA_Open"], w["HA_Close"])
        weekly_lower = min(w["HA_Open"], w["HA_Close"]) - w["HA_Low"]

        weekly_bull = weekly_body > 0 and weekly_lower <= weekly_body * 0.1
        weekly_bear = weekly_body < 0 and weekly_upper <= abs(weekly_body) * 0.1

        if debug_mode:
            log.append(
                f"{symbol} | Weekly: body={weekly_body:.2f}, upper={weekly_upper:.2f}, "
                f"lower={weekly_lower:.2f}, bull={weekly_bull}, bear={weekly_bear}"
            )

        if not (weekly_bull or weekly_bear):
            return None, "\n".join(log) if log else f"{symbol}: failed weekly filter"

        # ── HOURLY ─────────────────────────────────────────────────────────
        hourly = pd.DataFrame(kite.historical_data(token, START_HOURLY, END_DATE, "60minute"))
        if len(hourly) < 20:
            return None, f"{symbol}: insufficient hourly data ({len(hourly)} bars)"

        hourly["EMA50"] = hourly["close"].ewm(span=50).mean()
        ha_hourly       = calculate_heikin_ashi(hourly)
        h               = ha_hourly.iloc[-1]
        ema             = hourly["EMA50"].iloc[-1]
        last            = hourly.iloc[-1]

        body_h  = abs(h["HA_Close"] - h["HA_Open"])
        range_h = h["HA_High"] - h["HA_Low"]
        if range_h == 0:
            return None, f"{symbol}: zero hourly range"

        ratio_h          = body_h / range_h
        hourly_bull      = h["HA_Close"] > h["HA_Open"] and ratio_h > 0.5
        hourly_bear      = h["HA_Close"] < h["HA_Open"] and ratio_h > 0.5
        ema_support      = last["low"]  <= ema * 1.002 and last["close"] > ema
        failed_breakdown = last["low"]  <  ema         and last["close"] > ema
        ema_resistance   = last["high"] >= ema * 0.998 and last["close"] < ema
        failed_breakout  = last["high"] >  ema         and last["close"] < ema

        if debug_mode:
            log.append(
                f"{symbol} | Hourly: ratio={ratio_h:.2f}, bull={hourly_bull}, bear={hourly_bear}, "
                f"ema={ema:.2f}, close={last['close']:.2f}, "
                f"ema_support={ema_support}, failed_breakdown={failed_breakdown}, "
                f"ema_resistance={ema_resistance}, failed_breakout={failed_breakout}"
            )

        # ── FINAL SIGNAL ───────────────────────────────────────────────────
        if weekly_bull and daily_bull_ok and hourly_bull and (ema_support or failed_breakdown):
            log.append(f"✅ {symbol} → BULLISH")
            return ("bullish", symbol), "\n".join(log)

        if weekly_bear and daily_bear_ok and hourly_bear and (ema_resistance or failed_breakout):
            log.append(f"✅ {symbol} → BEARISH")
            return ("bearish", symbol), "\n".join(log)

        log.append(f"{symbol}: passed daily+weekly but FAILED final combined check")
        return None, "\n".join(log)

    except Exception as e:
        return None, f"{symbol}: ERROR — {e}"


# ================= PARALLEL SCAN =================
def scan_market(progress_bar, status_text, bull_placeholder, bear_placeholder, debug_mode):
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

    total_before = len(df)

    # ── Step 1: Volume pre-filter ──────────────────────────────────────────
    status_text.info(f"Step 1/2 — Volume pre-filter across {total_before} stocks…")
    df    = pre_filter_by_volume(df, status_text)
    total = len(df)

    if total == 0:
        st.error(
            "❌ All stocks removed by volume filter. "
            "Lower the 'Min Volume Filter' in the sidebar and try again."
        )
        return [], [], []

    status_text.success(
        f"Step 1/2 ✅ — {total_before} → {total} stocks after volume filter "
        f"({total_before - total} removed)"
    )

    bullish   = []
    bearish   = []
    skipped   = 0
    done      = 0
    debug_log = []

    # ── Step 2: Parallel historical scan ──────────────────────────────────
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(scan_symbol, row, debug_mode): row["tradingsymbol"]
            for _, row in df.iterrows()
        }

        for f in as_completed(futures):
            result, log_line = f.result()
            done += 1

            if log_line:
                debug_log.append(log_line)

            if result is None:
                skipped += 1
            else:
                side, sym = result
                if side == "bullish":
                    bullish.append(sym)
                else:
                    bearish.append(sym)

            progress_bar.progress(done / total)
            status_text.info(
                f"Step 2/2 — `{done}/{total}` scanned &nbsp;|&nbsp; "
                f"🟢 **{len(bullish)}** bullish &nbsp;|&nbsp; "
                f"🔴 **{len(bearish)}** bearish &nbsp;|&nbsp; "
                f"⏭ {skipped} skipped"
            )

            # ✅ KEY FIX: update SAME st.empty() placeholders — never create new columns
            bull_placeholder.write(sorted(bullish) if bullish else ["_None yet_"])
            bear_placeholder.write(sorted(bearish) if bearish else ["_None yet_"])

    return bullish, bearish, debug_log


# ================= UI =================

st.set_page_config(page_title="HA Strict Hybrid Scanner", layout="wide")

st.markdown("""
<h1 style='font-size:2rem; margin-bottom:0'>🔥 HA Strict Hybrid Scanner</h1>
<p style='color:gray; margin-top:4px'>
    Nifty 500 · Daily + Weekly + Hourly · EMA50 · Volume Pre-filter
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    MIN_VOLUME  = st.number_input("Min Volume Filter", value=MIN_VOLUME,  step=50_000, min_value=0)
    MAX_WORKERS = st.number_input("Parallel Workers",  value=MAX_WORKERS, step=5, min_value=1, max_value=50)
    debug_mode  = st.checkbox("🐛 Debug Mode (show scan log)", value=False)
    auto_mode   = st.checkbox("🔄 Auto Scan (hourly)", value=False)
    st.divider()
    st.caption(f"Workers: {MAX_WORKERS} · Min Vol: {MIN_VOLUME:,}")

if auto_mode:
    st_autorefresh(interval=3_600_000, limit=None)

run_scan = st.button("🚀 Run Scan", type="primary")

if run_scan or auto_mode:

    st.markdown("---")
    progress_bar = st.progress(0)
    status_text  = st.empty()

    # ✅ KEY FIX: columns created ONCE here, placeholders updated inside loop
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🟢 Bullish (live)")
        bull_placeholder = st.empty()
        bull_placeholder.write(["_Scanning…_"])
    with col2:
        st.markdown("### 🔴 Bearish (live)")
        bear_placeholder = st.empty()
        bear_placeholder.write(["_Scanning…_"])

    start_time             = time.time()
    bullish, bearish, debug_log = scan_market(
        progress_bar, status_text,
        bull_placeholder, bear_placeholder,
        debug_mode
    )
    elapsed = time.time() - start_time

    progress_bar.progress(1.0)
    status_text.success(
        f"✅ Scan complete in {elapsed:.1f}s — "
        f"{len(bullish)} Bullish · {len(bearish)} Bearish"
    )

    # ── Final results ─────────────────────────────────────────────────────
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🟢 Final Bullish Signals")
        if bullish:
            for sym in sorted(bullish):
                st.markdown(f"- `{sym}`")
        else:
            st.info("No bullish signals found.")
    with c2:
        st.subheader("🔴 Final Bearish Signals")
        if bearish:
            for sym in sorted(bearish):
                st.markdown(f"- `{sym}`")
        else:
            st.info("No bearish signals found.")

    # ── Metrics ───────────────────────────────────────────────────────────
    st.divider()
    prev     = load_previous_signals()
    new_bull = list(set(bullish) - set(prev["bullish"]))
    new_bear = list(set(bearish) - set(prev["bearish"]))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🟢 Bullish",     len(bullish))
    m2.metric("🔴 Bearish",     len(bearish))
    m3.metric("🆕 New Signals", len(new_bull) + len(new_bear))
    m4.metric("⏱ Scan Time",   f"{elapsed:.1f}s")

    # ── Email ─────────────────────────────────────────────────────────────
    if new_bull or new_bear:
        email_result = send_signal_alert(new_bull, new_bear)
        if email_result is True:
            st.success(f"📧 Alert sent — {len(new_bull)} new bullish, {len(new_bear)} new bearish")
        else:
            st.error(f"📧 Email failed: {email_result}")
    else:
        st.info("No new signals since last scan — email not sent.")

    save_signals(bullish, bearish)

    # ── Debug log ─────────────────────────────────────────────────────────
    if debug_mode and debug_log:
        st.divider()
        st.subheader("🐛 Debug Log")

        # Most useful: stocks that passed daily+weekly (reached final check)
        final_stage = [l for l in debug_log if "Hourly" in l or "✅" in l or "FAILED final" in l]
        with st.expander(f"📋 Stocks reaching final stage ({len(final_stage)} entries)", expanded=True):
            st.text("\n\n".join(final_stage[:300]))

        with st.expander("📋 Full scan log (all stocks, first 500 entries)"):
            st.text("\n\n".join(debug_log[:500]))
