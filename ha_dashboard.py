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
    from plotly.subplots import make_subplots
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

SIGNAL_STORE_FILE   = "last_signals.json"
SIGNAL_HISTORY_FILE = "signal_history.json"

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


# ================= SIGNAL HISTORY PERSISTENCE =================
def load_signal_history():
    if os.path.exists(SIGNAL_HISTORY_FILE):
        try:
            with open(SIGNAL_HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return []


def save_signal_history(history):
    with open(SIGNAL_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def append_signals_to_history(bullish, bearish):
    """Append today's new signals to persistent history with entry price."""
    history = load_signal_history()
    today   = str(datetime.date.today())

    # Avoid duplicate entries for same date
    existing = {(e["date"], e["symbol"]) for e in history}

    new_entries = []
    all_signals = [("bullish", s) for s in bullish] + [("bearish", s) for s in bearish]

    for side, sym in all_signals:
        if (today, sym) in existing:
            continue
        try:
            quote       = kite.quote(f"NSE:{sym}")
            entry_price = quote.get(f"NSE:{sym}", {}).get("last_price", None)
        except:
            entry_price = None

        new_entries.append({
            "date":        today,
            "symbol":      sym,
            "signal":      side,
            "entry_price": entry_price,
        })

    history.extend(new_entries)
    save_signal_history(history)
    return new_entries


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


# ================= BACKTEST ENGINE =================
def backtest_symbol(token, symbol, side, signal_date_str, entry_price, hold_days=(5, 10, 20)):
    """
    Given a signal (side, entry_price, date), fetch subsequent daily bars
    and calculate return at each hold horizon.
    Returns dict with pct returns at each horizon.
    """
    try:
        signal_date = datetime.date.fromisoformat(signal_date_str)
        fetch_start = signal_date
        fetch_end   = signal_date + datetime.timedelta(days=max(hold_days) + 10)
        fetch_end   = min(fetch_end, datetime.date.today())

        bars = pd.DataFrame(
            kite.historical_data(token, fetch_start, fetch_end, "day")
        )
        if bars.empty or entry_price is None or entry_price == 0:
            return None

        bars["date"] = pd.to_datetime(bars["date"]).dt.date
        bars = bars[bars["date"] >= signal_date].reset_index(drop=True)

        result = {
            "date":         signal_date_str,
            "symbol":       symbol,
            "signal":       side,
            "entry_price":  round(entry_price, 2),
        }

        for d in hold_days:
            if len(bars) > d:
                exit_price  = bars.iloc[d]["close"]
                multiplier  = 1 if side == "bullish" else -1
                pct         = multiplier * (exit_price - entry_price) / entry_price * 100
                result[f"return_{d}d"] = round(pct, 2)
                result[f"exit_{d}d"]   = round(exit_price, 2)
            else:
                result[f"return_{d}d"] = None
                result[f"exit_{d}d"]   = None

        # Max adverse / Max favourable excursion
        if len(bars) > 1:
            future_closes = bars["close"].iloc[1:max(hold_days)+1]
            if side == "bullish":
                result["max_gain"] = round((future_closes.max() - entry_price) / entry_price * 100, 2)
                result["max_loss"] = round((future_closes.min() - entry_price) / entry_price * 100, 2)
            else:
                result["max_gain"] = round((entry_price - future_closes.min()) / entry_price * 100, 2)
                result["max_loss"] = round((entry_price - future_closes.max()) / entry_price * 100, 2)
        else:
            result["max_gain"] = None
            result["max_loss"] = None

        return result
    except Exception as e:
        return None


def run_backtest(history, instrument_df, hold_days=(5, 10, 20), progress_bar=None, status_text=None):
    """Run backtest across all historical signals."""
    if not history:
        return pd.DataFrame()

    # Build token lookup
    token_map = dict(zip(instrument_df["tradingsymbol"], instrument_df["instrument_token"]))

    results  = []
    total    = len(history)
    done     = 0

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {}
        for entry in history:
            sym    = entry["symbol"]
            token  = token_map.get(sym)
            if token is None:
                continue
            f = executor.submit(
                backtest_symbol,
                token, sym,
                entry["signal"],
                entry["date"],
                entry["entry_price"],
                hold_days
            )
            futures[f] = sym

        for f in as_completed(futures):
            res = f.result()
            done += 1
            if res:
                results.append(res)
            if progress_bar:
                progress_bar.progress(done / total)
            if status_text:
                status_text.info(f"Backtesting `{done}/{total}` signals…")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df


# ================= PRE-FILTER BY VOLUME =================
def pre_filter_by_volume(df, status_text):
    symbols     = df["tradingsymbol"].tolist()
    nse_symbols = ["NSE:" + s for s in symbols]
    all_quotes  = {}

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
    status_text.info(f"Step 1/2 — Volume pre-filter across {total_before} stocks…")
    df    = pre_filter_by_volume(df, status_text)
    total = len(df)

    if total == 0:
        st.error("❌ All stocks removed by volume filter. Lower the 'Min Volume Filter' and try again.")
        return [], [], [], pd.DataFrame()

    status_text.success(
        f"Step 1/2 ✅ — {total_before} → {total} stocks after volume filter "
        f"({total_before - total} removed)"
    )

    bullish   = []
    bearish   = []
    skipped   = 0
    done      = 0
    debug_log = []

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

            bull_placeholder.write(sorted(bullish) if bullish else ["_None yet_"])
            bear_placeholder.write(sorted(bearish) if bearish else ["_None yet_"])

    return bullish, bearish, debug_log, df


# ================= BACKTEST UI =================
def render_backtest_tab(instrument_df):
    st.subheader("📊 Signal History & Backtest")

    history = load_signal_history()

    if not history:
        st.info(
            "No signal history yet. Run a scan first — each scan auto-saves signals "
            "with entry prices for future backtesting."
        )
        return

    # ── Summary stats ─────────────────────────────────────────────────────
    hist_df = pd.DataFrame(history)
    total_signals = len(hist_df)
    bull_count    = (hist_df["signal"] == "bullish").sum()
    bear_count    = (hist_df["signal"] == "bearish").sum()
    date_range    = f"{hist_df['date'].min()} → {hist_df['date'].max()}"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📋 Total Signals",  total_signals)
    m2.metric("🟢 Bullish",        bull_count)
    m3.metric("🔴 Bearish",        bear_count)
    m4.metric("📅 Date Range",     date_range)

    st.divider()

    # ── Controls ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        hold_days_input = st.multiselect(
            "Hold horizons (days)",
            options=[3, 5, 10, 15, 20, 30],
            default=[5, 10, 20]
        )
    with c2:
        filter_signal = st.selectbox("Filter by signal", ["All", "Bullish", "Bearish"])
    with c3:
        run_bt = st.button("▶️ Run Backtest", type="primary")

    if not hold_days_input:
        st.warning("Select at least one hold horizon.")
        return

    hold_days = tuple(sorted(hold_days_input))

    # ── Raw history table ─────────────────────────────────────────────────
    with st.expander("📋 Raw Signal History", expanded=False):
        display_hist = hist_df.copy()
        if filter_signal != "All":
            display_hist = display_hist[display_hist["signal"] == filter_signal.lower()]
        display_hist = display_hist.sort_values("date", ascending=False)
        st.dataframe(display_hist, use_container_width=True)

        csv = display_hist.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History CSV", csv, "signal_history.csv", "text/csv")

    # ── Backtest ──────────────────────────────────────────────────────────
    if run_bt:
        bt_history = history
        if filter_signal != "All":
            bt_history = [h for h in history if h["signal"] == filter_signal.lower()]

        if not bt_history:
            st.warning("No signals match the filter.")
            return

        bt_progress = st.progress(0)
        bt_status   = st.empty()

        with st.spinner(f"Running backtest on {len(bt_history)} signals…"):
            bt_df = run_backtest(
                bt_history, instrument_df,
                hold_days=hold_days,
                progress_bar=bt_progress,
                status_text=bt_status
            )

        bt_progress.progress(1.0)
        bt_status.success(f"✅ Backtest complete — {len(bt_df)} results")

        if bt_df.empty:
            st.error("Backtest returned no results. Check instrument data or signal history.")
            return

        # ── Performance summary ───────────────────────────────────────────
        st.divider()
        st.subheader("📈 Performance Summary")

        summary_rows = []
        for d in hold_days:
            col = f"return_{d}d"
            if col not in bt_df.columns:
                continue
            series   = bt_df[col].dropna()
            wins     = (series > 0).sum()
            losses   = (series <= 0).sum()
            total    = len(series)
            win_rate = wins / total * 100 if total > 0 else 0
            summary_rows.append({
                "Hold Period":   f"{d} days",
                "Signals":       total,
                "Win Rate %":    round(win_rate, 1),
                "Avg Return %":  round(series.mean(), 2),
                "Best %":        round(series.max(), 2),
                "Worst %":       round(series.min(), 2),
                "Median %":      round(series.median(), 2),
            })

        summary_df = pd.DataFrame(summary_rows)

        # Color the summary table
        def color_return(val):
            if isinstance(val, float):
                color = "green" if val > 0 else "red"
                return f"color: {color}"
            return ""

        st.dataframe(
            summary_df.style.applymap(color_return, subset=["Avg Return %", "Best %", "Worst %", "Median %"]),
            use_container_width=True
        )

        # ── Detailed results table ────────────────────────────────────────
        st.divider()
        st.subheader("📋 Detailed Backtest Results")

        return_cols  = [f"return_{d}d" for d in hold_days if f"return_{d}d" in bt_df.columns]
        display_cols = ["date", "symbol", "signal", "entry_price", "max_gain", "max_loss"] + return_cols
        display_cols = [c for c in display_cols if c in bt_df.columns]
        disp_df      = bt_df[display_cols].copy()

        def color_cell(val):
            if isinstance(val, (int, float)):
                return f"color: {'green' if val > 0 else 'red'}"
            return ""

        styled = disp_df.style.applymap(color_cell, subset=return_cols + ["max_gain", "max_loss"])
        st.dataframe(styled, use_container_width=True)

        csv_bt = bt_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Backtest CSV", csv_bt, "backtest_results.csv", "text/csv")

        # ── Charts ────────────────────────────────────────────────────────
        if PLOTLY_AVAILABLE:
            st.divider()
            st.subheader("📉 Charts")

            tab_equity, tab_dist, tab_heatmap = st.tabs([
                "Equity Curve", "Return Distribution", "Win Rate Heatmap"
            ])

            primary_d   = hold_days[0]
            primary_col = f"return_{primary_d}d"

            # ── Equity curve (cumulative avg return) ──────────────────────
            with tab_equity:
                if primary_col in bt_df.columns:
                    eq = (
                        bt_df[["date", primary_col]]
                        .dropna()
                        .sort_values("date")
                        .copy()
                    )
                    eq["cumulative"] = eq[primary_col].cumsum()

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=eq["date"], y=eq["cumulative"],
                        mode="lines+markers",
                        name=f"Cumulative Return ({primary_d}d)",
                        line=dict(color="royalblue", width=2),
                        fill="tozeroy",
                        fillcolor="rgba(65,105,225,0.1)"
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        title=f"Cumulative Signal Returns ({primary_d}-day hold)",
                        xaxis_title="Signal Date",
                        yaxis_title="Cumulative Return %",
                        height=400,
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # ── Return distribution ───────────────────────────────────────
            with tab_dist:
                fig2 = make_subplots(
                    rows=1, cols=len(hold_days),
                    subplot_titles=[f"{d}d returns" for d in hold_days]
                )
                for i, d in enumerate(hold_days, 1):
                    col = f"return_{d}d"
                    if col not in bt_df.columns:
                        continue
                    series = bt_df[col].dropna()
                    colors = ["green" if v > 0 else "red" for v in series]
                    fig2.add_trace(
                        go.Histogram(
                            x=series, name=f"{d}d",
                            marker_color=colors,
                            nbinsx=30, opacity=0.75
                        ),
                        row=1, col=i
                    )
                fig2.update_layout(
                    title="Return Distribution by Hold Period",
                    height=400, template="plotly_dark", showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)

            # ── Win rate heatmap by month ─────────────────────────────────
            with tab_heatmap:
                primary_col = f"return_{primary_d}d"
                if primary_col in bt_df.columns:
                    hm = bt_df[["date", "signal", primary_col]].dropna().copy()
                    hm["month"]   = pd.to_datetime(hm["date"]).dt.to_period("M").astype(str)
                    hm["signal_type"] = hm["signal"].str.capitalize()
                    hm["win"]     = (hm[primary_col] > 0).astype(int)

                    pivot = hm.pivot_table(
                        index="signal_type", columns="month",
                        values="win", aggfunc="mean"
                    ) * 100

                    if not pivot.empty:
                        fig3 = go.Figure(go.Heatmap(
                            z=pivot.values,
                            x=pivot.columns.tolist(),
                            y=pivot.index.tolist(),
                            colorscale="RdYlGn",
                            zmin=0, zmax=100,
                            text=pivot.values.round(0),
                            texttemplate="%{text}%",
                            colorbar=dict(title="Win Rate %")
                        ))
                        fig3.update_layout(
                            title=f"Monthly Win Rate Heatmap ({primary_d}d hold)",
                            height=300, template="plotly_dark"
                        )
                        st.plotly_chart(fig3, use_container_width=True)

        # ── Best & worst trades ───────────────────────────────────────────
        st.divider()
        col_best, col_worst = st.columns(2)
        primary_col = f"return_{hold_days[0]}d"

        if primary_col in bt_df.columns:
            sorted_bt = bt_df.dropna(subset=[primary_col]).sort_values(primary_col, ascending=False)
            with col_best:
                st.markdown("#### 🏆 Top 10 Trades")
                top10 = sorted_bt.head(10)[["date", "symbol", "signal", primary_col, "max_gain"]]
                st.dataframe(top10, use_container_width=True)
            with col_worst:
                st.markdown("#### 💀 Bottom 10 Trades")
                bot10 = sorted_bt.tail(10)[["date", "symbol", "signal", primary_col, "max_loss"]]
                st.dataframe(bot10, use_container_width=True)

    # ── Delete history ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚠️ Danger Zone"):
        if st.button("🗑️ Clear All Signal History", type="secondary"):
            save_signal_history([])
            st.success("Signal history cleared.")
            st.rerun()


# ================= UI =================

st.set_page_config(page_title="HA Strict Hybrid Scanner", layout="wide")

st.markdown("""
<h1 style='font-size:2rem; margin-bottom:0'>🔥 HA Strict Hybrid Scanner</h1>
<p style='color:gray; margin-top:4px'>
    Nifty 500 · Daily + Weekly + Hourly · EMA50 · Volume Pre-filter · Backtest
</p>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    MIN_VOLUME  = st.number_input("Min Volume Filter", value=MIN_VOLUME,  step=50_000, min_value=0)
    MAX_WORKERS = st.number_input("Parallel Workers",  value=MAX_WORKERS, step=5, min_value=1, max_value=50)
    debug_mode  = st.checkbox("🐛 Debug Mode", value=False)
    auto_mode   = st.checkbox("🔄 Auto Scan (hourly)", value=False)
    st.divider()
    st.caption(f"Workers: {MAX_WORKERS} · Min Vol: {MIN_VOLUME:,}")

if auto_mode:
    st_autorefresh(interval=3_600_000, limit=None)

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_scanner, tab_backtest = st.tabs(["🔍 Scanner", "📊 Signal History & Backtest"])

# ── Shared instrument_df (loaded once, reused in backtest) ────────────────
@st.cache_data(ttl=3600)
def get_instrument_df():
    instruments = kite.instruments("NSE")
    df = pd.DataFrame(instruments)
    nse500 = pd.read_csv(
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    )["Symbol"].tolist()
    return df[
        (df["segment"] == "NSE") &
        (df["instrument_type"] == "EQ") &
        (df["tradingsymbol"].isin(nse500))
    ].reset_index(drop=True)

try:
    instrument_df = get_instrument_df()
except Exception as e:
    instrument_df = pd.DataFrame()
    st.sidebar.warning(f"Instrument load failed: {e}")

# ── Scanner tab ───────────────────────────────────────────────────────────
with tab_scanner:
    run_scan = st.button("🚀 Run Scan", type="primary")

    if run_scan or auto_mode:
        st.markdown("---")
        progress_bar = st.progress(0)
        status_text  = st.empty()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🟢 Bullish (live)")
            bull_placeholder = st.empty()
            bull_placeholder.write(["_Scanning…_"])
        with col2:
            st.markdown("### 🔴 Bearish (live)")
            bear_placeholder = st.empty()
            bear_placeholder.write(["_Scanning…_"])

        start_time = time.time()
        bullish, bearish, debug_log, scanned_df = scan_market(
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

        st.divider()
        prev     = load_previous_signals()
        new_bull = list(set(bullish) - set(prev["bullish"]))
        new_bear = list(set(bearish) - set(prev["bearish"]))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🟢 Bullish",     len(bullish))
        m2.metric("🔴 Bearish",     len(bearish))
        m3.metric("🆕 New Signals", len(new_bull) + len(new_bear))
        m4.metric("⏱ Scan Time",   f"{elapsed:.1f}s")

        # ── Auto-save signals to history with entry prices ─────────────────
        if bullish or bearish:
            saved = append_signals_to_history(bullish, bearish)
            if saved:
                st.success(f"📝 {len(saved)} new signal(s) saved to history for backtesting.")

        if new_bull or new_bear:
            email_result = send_signal_alert(new_bull, new_bear)
            if email_result is True:
                st.success(f"📧 Alert sent — {len(new_bull)} new bullish, {len(new_bear)} new bearish")
            else:
                st.error(f"📧 Email failed: {email_result}")
        else:
            st.info("No new signals since last scan — email not sent.")

        save_signals(bullish, bearish)

        if debug_mode and debug_log:
            st.divider()
            st.subheader("🐛 Debug Log")
            final_stage = [l for l in debug_log if "Hourly" in l or "✅" in l or "FAILED final" in l]
            with st.expander(f"Stocks reaching final stage ({len(final_stage)} entries)", expanded=True):
                st.text("\n\n".join(final_stage[:300]))
            with st.expander("Full scan log"):
                st.text("\n\n".join(debug_log[:500]))

# ── Backtest tab ──────────────────────────────────────────────────────────
with tab_backtest:
    render_backtest_tab(instrument_df)
