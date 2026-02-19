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

TODAY        = datetime.date.today()
START_DAILY  = TODAY - datetime.timedelta(days=120)
START_WEEKLY = TODAY - datetime.timedelta(days=365)
START_HOURLY = TODAY - datetime.timedelta(days=10)
END_DATE     = TODAY

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


# ================= SIGNAL HISTORY (live scan saves) =================
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
    history  = load_signal_history()
    today    = str(TODAY)
    existing = {(e["date"], e["symbol"]) for e in history}
    new_entries = []
    for side, sym in [("bullish", s) for s in bullish] + [("bearish", s) for s in bearish]:
        if (today, sym) in existing:
            continue
        try:
            quote       = kite.quote(f"NSE:{sym}")
            entry_price = quote.get(f"NSE:{sym}", {}).get("last_price", None)
        except:
            entry_price = None
        new_entries.append({"date": today, "symbol": sym, "signal": side, "entry_price": entry_price})
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


# =================================================================
# HISTORICAL BAR-BY-BAR BACKTEST ENGINE
# Replays the DAILY HA signal logic on every past daily bar in the
# backtest window, records entry price (next open), then measures
# returns at each hold horizon.
#
# NOTE: Weekly & hourly checks need historical weekly/hourly data too.
# For simplicity and speed this engine uses DAILY-only signal replay
# (which is the primary trend filter). Hourly EMA entry is simulated
# using the daily close vs daily EMA50.
# =================================================================

def _check_daily_signal(ha_row, ema50_val, close_val, body_threshold=0.2):
    """
    Apply daily HA signal logic to a single bar.
    Returns 'bullish', 'bearish', or None.
    """
    d = ha_row
    body       = abs(d["HA_Close"] - d["HA_Open"])
    full_range = d["HA_High"] - d["HA_Low"]
    if full_range == 0:
        return None

    body_ratio  = body / full_range
    upper_wick  = d["HA_High"] - max(d["HA_Open"], d["HA_Close"])
    lower_wick  = min(d["HA_Open"], d["HA_Close"]) - d["HA_Low"]
    price_above = close_val > ema50_val
    price_below = close_val < ema50_val

    daily_neutral_bull = (body_ratio < body_threshold and upper_wick > 0
                          and lower_wick > 0 and price_above)
    daily_strong_bull  = (d["HA_Close"] > d["HA_Open"]
                          and lower_wick <= body * 0.1
                          and body_ratio > 0.5 and price_above)
    daily_neutral_bear = (body_ratio < body_threshold and upper_wick > 0
                          and lower_wick > 0 and price_below)
    daily_strong_bear  = (d["HA_Close"] < d["HA_Open"]
                          and upper_wick <= body * 0.1
                          and body_ratio > 0.5 and price_below)

    if daily_neutral_bull or daily_strong_bull:
        return "bullish"
    if daily_neutral_bear or daily_strong_bear:
        return "bearish"
    return None


def _check_weekly_signal(ha_weekly_row):
    """Apply weekly HA signal logic."""
    w            = ha_weekly_row
    weekly_body  = w["HA_Close"] - w["HA_Open"]
    weekly_upper = w["HA_High"] - max(w["HA_Open"], w["HA_Close"])
    weekly_lower = min(w["HA_Open"], w["HA_Close"]) - w["HA_Low"]

    if weekly_body > 0 and weekly_lower <= weekly_body * 0.1:
        return "bullish"
    if weekly_body < 0 and weekly_upper <= abs(weekly_body) * 0.1:
        return "bearish"
    return None


def historical_backtest_symbol(token, symbol, bt_start, bt_end, hold_days):
    """
    For a single stock:
    1. Fetch enough historical daily + weekly bars (extra warmup for EMA50 + HA)
    2. Walk forward bar-by-bar from bt_start to bt_end
    3. On each bar, check if the HA daily+weekly signal fires
    4. Record entry (next bar open) and returns at each hold horizon
    5. Avoid duplicate signals — skip if same-direction signal already open
    """
    try:
        # Need 60 extra days warmup for EMA50 + HA calculation
        warmup_start = bt_start - datetime.timedelta(days=200)

        daily = pd.DataFrame(kite.historical_data(token, warmup_start, bt_end, "day"))
        if len(daily) < 70:
            return []

        weekly_start = bt_start - datetime.timedelta(days=400)
        weekly = pd.DataFrame(kite.historical_data(token, weekly_start, bt_end, "week"))
        if len(weekly) < 15:
            return []

        # Prepare daily
        daily["date"]  = pd.to_datetime(daily["date"]).dt.date
        daily["EMA50"] = daily["close"].ewm(span=50).mean()
        ha_daily       = calculate_heikin_ashi(daily)
        ha_daily["date"]  = daily["date"].values
        ha_daily["close"] = daily["close"].values
        ha_daily["open"]  = daily["open"].values
        ha_daily["EMA50"] = daily["EMA50"].values

        # Prepare weekly — build a date→weekly_signal lookup
        weekly["date"] = pd.to_datetime(weekly["date"]).dt.date
        ha_weekly      = calculate_heikin_ashi(weekly)
        ha_weekly["date"] = weekly["date"].values

        # Build a daily-date → most recent weekly HA signal map
        # Weekly bar date = week start; we map forward to all days in that week
        weekly_signal_by_date = {}
        for i, wrow in ha_weekly.iterrows():
            sig = _check_weekly_signal(wrow)
            weekly_signal_by_date[wrow["date"]] = sig

        def get_weekly_signal_for_date(d):
            # find the most recent weekly bar date <= d
            candidates = [k for k in weekly_signal_by_date if k <= d]
            if not candidates:
                return None
            return weekly_signal_by_date[max(candidates)]

        # Walk forward through daily bars in backtest window
        signals_found = []
        last_signal   = None   # prevent consecutive same-direction signals

        bt_start_dt = bt_start
        bt_end_dt   = bt_end

        for i, row in ha_daily.iterrows():
            bar_date = row["date"]
            if bar_date < bt_start_dt or bar_date > bt_end_dt:
                continue
            if i + 1 >= len(ha_daily):
                continue  # need next bar for entry price

            daily_sig  = _check_daily_signal(row, row["EMA50"], row["close"])
            weekly_sig = get_weekly_signal_for_date(bar_date)

            # Both must agree
            if daily_sig is None or weekly_sig is None:
                last_signal = None
                continue
            if daily_sig != weekly_sig:
                last_signal = None
                continue

            signal = daily_sig

            # Avoid repeating same signal back-to-back (cooldown)
            if signal == last_signal:
                continue

            # Entry = next bar's open (realistic — can't trade on same bar's close)
            next_bar    = ha_daily.iloc[i + 1]
            entry_price = next_bar["open"]
            entry_date  = next_bar["date"]
            if entry_price == 0:
                continue

            result = {
                "signal_date": str(bar_date),
                "entry_date":  str(entry_date),
                "symbol":      symbol,
                "signal":      signal,
                "entry_price": round(entry_price, 2),
            }

            # Calculate returns at each hold horizon
            for d in hold_days:
                exit_idx = i + 1 + d
                if exit_idx < len(ha_daily):
                    exit_price = ha_daily.iloc[exit_idx]["close"]
                    mult       = 1 if signal == "bullish" else -1
                    pct        = mult * (exit_price - entry_price) / entry_price * 100
                    result[f"return_{d}d"] = round(pct, 2)
                    result[f"exit_{d}d"]   = round(exit_price, 2)
                else:
                    result[f"return_{d}d"] = None
                    result[f"exit_{d}d"]   = None

            # Max gain / max loss over the longest hold window
            max_d    = max(hold_days)
            end_idx  = min(i + 1 + max_d, len(ha_daily))
            future   = ha_daily.iloc[i + 1:end_idx]["close"]
            if len(future) > 0:
                if signal == "bullish":
                    result["max_gain"] = round((future.max() - entry_price) / entry_price * 100, 2)
                    result["max_loss"] = round((future.min() - entry_price) / entry_price * 100, 2)
                else:
                    result["max_gain"] = round((entry_price - future.min()) / entry_price * 100, 2)
                    result["max_loss"] = round((entry_price - future.max()) / entry_price * 100, 2)
            else:
                result["max_gain"] = None
                result["max_loss"] = None

            signals_found.append(result)
            last_signal = signal

        return signals_found

    except Exception as e:
        return []


def run_historical_backtest(instrument_df, bt_start, bt_end, hold_days,
                             progress_bar=None, status_text=None):
    """Run historical backtest across all Nifty500 stocks in parallel."""
    token_map = dict(zip(instrument_df["tradingsymbol"], instrument_df["instrument_token"]))
    all_rows  = []
    total     = len(instrument_df)
    done      = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                historical_backtest_symbol,
                token_map[row["tradingsymbol"]],
                row["tradingsymbol"],
                bt_start, bt_end, hold_days
            ): row["tradingsymbol"]
            for _, row in instrument_df.iterrows()
            if row["tradingsymbol"] in token_map
        }

        for f in as_completed(futures):
            rows = f.result()
            done += 1
            if rows:
                all_rows.extend(rows)
            if progress_bar:
                progress_bar.progress(done / total)
            if status_text:
                sym = futures[f]
                status_text.info(
                    f"Backtesting `{done}/{total}` stocks… "
                    f"· {len(all_rows)} signals found so far"
                )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values("signal_date", ascending=False).reset_index(drop=True)
    return df


# ================= BACKTEST RESULTS UI =================
def render_backtest_results(bt_df, hold_days):
    """Shared UI to display backtest results — used by both tabs."""
    if bt_df.empty:
        st.warning("No signals found in this period. Try a wider date range or relax filters.")
        return

    return_cols = [f"return_{d}d" for d in hold_days if f"return_{d}d" in bt_df.columns]

    # ── Summary metrics ───────────────────────────────────────────────────
    st.subheader("📈 Performance Summary")
    summary_rows = []
    for d in hold_days:
        col = f"return_{d}d"
        if col not in bt_df.columns:
            continue
        series   = bt_df[col].dropna()
        wins     = (series > 0).sum()
        total    = len(series)
        win_rate = wins / total * 100 if total > 0 else 0
        summary_rows.append({
            "Hold Period":  f"{d} days",
            "# Signals":    total,
            "Win Rate %":   round(win_rate, 1),
            "Avg Return %": round(series.mean(), 2),
            "Median %":     round(series.median(), 2),
            "Best %":       round(series.max(), 2),
            "Worst %":      round(series.min(), 2),
            "Std Dev %":    round(series.std(), 2),
        })

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        def color_val(val):
            if isinstance(val, (int, float)):
                return f"color: {'green' if val > 0 else ('red' if val < 0 else 'gray')}"
            return ""
        st.dataframe(
            sdf.style.applymap(color_val, subset=["Avg Return %", "Median %", "Best %", "Worst %"]),
            use_container_width=True
        )

    # Signal direction breakdown
    bull_bt = bt_df[bt_df["signal"] == "bullish"]
    bear_bt = bt_df[bt_df["signal"] == "bearish"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📋 Total Signals", len(bt_df))
    m2.metric("🟢 Bullish",       len(bull_bt))
    m3.metric("🔴 Bearish",       len(bear_bt))
    primary_col = f"return_{hold_days[0]}d"
    if primary_col in bt_df.columns:
        overall_wr = (bt_df[primary_col].dropna() > 0).mean() * 100
        m4.metric(f"🎯 Win Rate ({hold_days[0]}d)", f"{overall_wr:.1f}%")

    # ── Charts ────────────────────────────────────────────────────────────
    if PLOTLY_AVAILABLE and return_cols:
        st.divider()
        tab_eq, tab_dist, tab_heatmap, tab_sym = st.tabs([
            "📈 Equity Curve", "📊 Distribution", "🗓 Monthly Heatmap", "🏆 By Symbol"
        ])

        primary_col = f"return_{hold_days[0]}d"
        primary_d   = hold_days[0]

        # Equity curve
        with tab_eq:
            if primary_col in bt_df.columns:
                eq = bt_df[["signal_date", "signal", primary_col]].dropna().sort_values("signal_date")

                fig = go.Figure()
                for sig, color in [("bullish", "limegreen"), ("bearish", "tomato")]:
                    sub = eq[eq["signal"] == sig].copy()
                    sub["cumret"] = sub[primary_col].cumsum()
                    fig.add_trace(go.Scatter(
                        x=sub["signal_date"], y=sub["cumret"],
                        name=sig.capitalize(), mode="lines",
                        line=dict(color=color, width=2)
                    ))

                all_eq = eq.copy()
                all_eq["cumret"] = all_eq[primary_col].cumsum()
                fig.add_trace(go.Scatter(
                    x=all_eq["signal_date"], y=all_eq["cumret"],
                    name="Combined", mode="lines",
                    line=dict(color="royalblue", width=2, dash="dot")
                ))

                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title=f"Cumulative Returns — {primary_d}-day hold",
                    xaxis_title="Signal Date", yaxis_title="Cumulative Return %",
                    height=420, template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Distribution
        with tab_dist:
            ncols = len(hold_days)
            fig2  = make_subplots(rows=1, cols=ncols,
                                   subplot_titles=[f"{d}-day hold" for d in hold_days])
            for i, d in enumerate(hold_days, 1):
                col = f"return_{d}d"
                if col not in bt_df.columns:
                    continue
                vals   = bt_df[col].dropna()
                colors = ["green" if v > 0 else "red" for v in vals]
                fig2.add_trace(go.Histogram(
                    x=vals, name=f"{d}d",
                    marker_color=colors, nbinsx=40, opacity=0.75
                ), row=1, col=i)
                fig2.add_vline(x=0, line_dash="dash", line_color="white",
                               row=1, col=i)
            fig2.update_layout(
                title="Return Distribution", height=400,
                template="plotly_dark", showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Monthly heatmap
        with tab_heatmap:
            if primary_col in bt_df.columns:
                hm = bt_df[["signal_date", "signal", primary_col]].dropna().copy()
                hm["month"] = pd.to_datetime(hm["signal_date"]).dt.to_period("M").astype(str)
                hm["win"]   = (hm[primary_col] > 0).astype(int)
                pivot = hm.pivot_table(
                    index="signal", columns="month",
                    values="win", aggfunc="mean"
                ) * 100

                if not pivot.empty:
                    fig3 = go.Figure(go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns.tolist(),
                        y=[s.capitalize() for s in pivot.index.tolist()],
                        colorscale="RdYlGn", zmin=0, zmax=100,
                        text=pivot.values.round(0),
                        texttemplate="%{text}%",
                        colorbar=dict(title="Win Rate %")
                    ))
                    fig3.update_layout(
                        title=f"Monthly Win Rate ({primary_d}-day hold)",
                        height=300, template="plotly_dark"
                    )
                    st.plotly_chart(fig3, use_container_width=True)

        # By symbol
        with tab_sym:
            if primary_col in bt_df.columns:
                sym_stats = (
                    bt_df.groupby("symbol")[primary_col]
                    .agg(Signals="count", WinRate=lambda x: (x > 0).mean() * 100,
                         AvgReturn="mean")
                    .round(2)
                    .sort_values("AvgReturn", ascending=False)
                    .reset_index()
                )
                sym_stats.columns = ["Symbol", "Signals", "Win Rate %", "Avg Return %"]
                top20  = sym_stats.head(20)
                bot20  = sym_stats.tail(20)
                c_sym1, c_sym2 = st.columns(2)
                with c_sym1:
                    st.markdown("#### 🏆 Top 20 Symbols")
                    st.dataframe(top20, use_container_width=True)
                with c_sym2:
                    st.markdown("#### 💀 Bottom 20 Symbols")
                    st.dataframe(bot20, use_container_width=True)

    # ── Detailed trades table ─────────────────────────────────────────────
    st.divider()
    st.subheader("📋 All Trades")
    display_cols = (["signal_date", "entry_date", "symbol", "signal", "entry_price",
                      "max_gain", "max_loss"] + return_cols)
    display_cols = [c for c in display_cols if c in bt_df.columns]

    def color_cell(val):
        if isinstance(val, (int, float)):
            return f"color: {'green' if val > 0 else 'red'}"
        return ""

    styled_bt = bt_df[display_cols].style.applymap(
        color_cell, subset=[c for c in return_cols + ["max_gain", "max_loss"]
                            if c in bt_df.columns]
    )
    st.dataframe(styled_bt, use_container_width=True)

    csv = bt_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "backtest_results.csv", "text/csv")

    # ── Top / bottom trades ───────────────────────────────────────────────
    st.divider()
    primary_col = f"return_{hold_days[0]}d"
    if primary_col in bt_df.columns:
        sorted_bt = bt_df.dropna(subset=[primary_col]).sort_values(primary_col, ascending=False)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🏆 Top 10 Trades")
            st.dataframe(
                sorted_bt.head(10)[["signal_date", "symbol", "signal", primary_col, "max_gain"]],
                use_container_width=True
            )
        with c2:
            st.markdown("#### 💀 Bottom 10 Trades")
            st.dataframe(
                sorted_bt.tail(10)[["signal_date", "symbol", "signal", primary_col, "max_loss"]],
                use_container_width=True
            )


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


# ================= SINGLE STOCK LIVE SCAN =================
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
            log.append(f"{symbol} | Daily: body_ratio={body_ratio:.2f}, price_above={price_above}, "
                       f"neutral_bull={daily_neutral_bull}, strong_bull={daily_strong_bull}, "
                       f"neutral_bear={daily_neutral_bear}, strong_bear={daily_strong_bear}")

        if not (daily_bull_ok or daily_bear_ok):
            return None, "\n".join(log) if log else f"{symbol}: failed daily"

        weekly = pd.DataFrame(kite.historical_data(token, START_WEEKLY, END_DATE, "week"))
        if len(weekly) < 10:
            return None, f"{symbol}: insufficient weekly data"
        ha_weekly    = calculate_heikin_ashi(weekly)
        w            = ha_weekly.iloc[-1]
        weekly_body  = w["HA_Close"] - w["HA_Open"]
        weekly_upper = w["HA_High"] - max(w["HA_Open"], w["HA_Close"])
        weekly_lower = min(w["HA_Open"], w["HA_Close"]) - w["HA_Low"]
        weekly_bull  = weekly_body > 0 and weekly_lower <= weekly_body * 0.1
        weekly_bear  = weekly_body < 0 and weekly_upper <= abs(weekly_body) * 0.1

        if debug_mode:
            log.append(f"{symbol} | Weekly: body={weekly_body:.2f}, bull={weekly_bull}, bear={weekly_bear}")

        if not (weekly_bull or weekly_bear):
            return None, "\n".join(log) if log else f"{symbol}: failed weekly"

        hourly = pd.DataFrame(kite.historical_data(token, START_HOURLY, END_DATE, "60minute"))
        if len(hourly) < 20:
            return None, f"{symbol}: insufficient hourly data"
        hourly["EMA50"] = hourly["close"].ewm(span=50).mean()
        ha_hourly       = calculate_heikin_ashi(hourly)
        h               = ha_hourly.iloc[-1]
        ema             = hourly["EMA50"].iloc[-1]
        last            = hourly.iloc[-1]
        body_h          = abs(h["HA_Close"] - h["HA_Open"])
        range_h         = h["HA_High"] - h["HA_Low"]
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
            log.append(f"{symbol} | Hourly: ratio={ratio_h:.2f}, bull={hourly_bull}, bear={hourly_bear}, "
                       f"ema_support={ema_support}, failed_breakdown={failed_breakdown}")

        if weekly_bull and daily_bull_ok and hourly_bull and (ema_support or failed_breakdown):
            log.append(f"✅ {symbol} → BULLISH")
            return ("bullish", symbol), "\n".join(log)
        if weekly_bear and daily_bear_ok and hourly_bear and (ema_resistance or failed_breakout):
            log.append(f"✅ {symbol} → BEARISH")
            return ("bearish", symbol), "\n".join(log)

        log.append(f"{symbol}: passed daily+weekly but FAILED final check")
        return None, "\n".join(log)
    except Exception as e:
        return None, f"{symbol}: ERROR — {e}"


# ================= PARALLEL LIVE SCAN =================
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
        st.error("❌ All stocks removed by volume filter. Lower the threshold and try again.")
        return [], [], [], pd.DataFrame()

    status_text.success(f"Step 1/2 ✅ — {total_before} → {total} stocks after volume filter")

    bullish = []; bearish = []; skipped = 0; done = 0; debug_log = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scan_symbol, row, debug_mode): row["tradingsymbol"]
                   for _, row in df.iterrows()}
        for f in as_completed(futures):
            result, log_line = f.result()
            done += 1
            if log_line:
                debug_log.append(log_line)
            if result is None:
                skipped += 1
            else:
                side, sym = result
                if side == "bullish": bullish.append(sym)
                else: bearish.append(sym)
            progress_bar.progress(done / total)
            status_text.info(
                f"Step 2/2 — `{done}/{total}` scanned | "
                f"🟢 **{len(bullish)}** | 🔴 **{len(bearish)}** | ⏭ {skipped} skipped"
            )
            bull_placeholder.write(sorted(bullish) if bullish else ["_None yet_"])
            bear_placeholder.write(sorted(bearish) if bearish else ["_None yet_"])

    return bullish, bearish, debug_log, df


# ================= UI =================

st.set_page_config(page_title="HA Strict Hybrid Scanner", layout="wide")
st.markdown("""
<h1 style='font-size:2rem; margin-bottom:0'>🔥 HA Strict Hybrid Scanner</h1>
<p style='color:gray; margin-top:4px'>
    Nifty 500 · Daily + Weekly + Hourly · EMA50 · Volume Pre-filter · Historical Backtest
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
tab_scanner, tab_historical_bt, tab_live_history = st.tabs([
    "🔍 Live Scanner",
    "📊 Historical Backtest (last N months)",
    "📋 Live Scan Signal History"
])


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


# ═══════════════════════════════════════════════════════════
# TAB 1 — LIVE SCANNER
# ═══════════════════════════════════════════════════════════
with tab_scanner:
    run_scan = st.button("🚀 Run Live Scan", type="primary")

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
            progress_bar, status_text, bull_placeholder, bear_placeholder, debug_mode
        )
        elapsed = time.time() - start_time

        progress_bar.progress(1.0)
        status_text.success(f"✅ Scan complete in {elapsed:.1f}s — {len(bullish)} Bullish · {len(bearish)} Bearish")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🟢 Bullish Signals")
            [st.markdown(f"- `{s}`") for s in sorted(bullish)] if bullish else st.info("None found.")
        with c2:
            st.subheader("🔴 Bearish Signals")
            [st.markdown(f"- `{s}`") for s in sorted(bearish)] if bearish else st.info("None found.")

        st.divider()
        prev     = load_previous_signals()
        new_bull = list(set(bullish) - set(prev["bullish"]))
        new_bear = list(set(bearish) - set(prev["bearish"]))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🟢 Bullish", len(bullish))
        m2.metric("🔴 Bearish", len(bearish))
        m3.metric("🆕 New", len(new_bull) + len(new_bear))
        m4.metric("⏱ Time", f"{elapsed:.1f}s")

        if bullish or bearish:
            saved = append_signals_to_history(bullish, bearish)
            if saved:
                st.success(f"📝 {len(saved)} signal(s) saved to history.")

        if new_bull or new_bear:
            r = send_signal_alert(new_bull, new_bear)
            st.success("📧 Alert sent.") if r is True else st.error(f"📧 Email failed: {r}")
        else:
            st.info("No new signals since last scan.")

        save_signals(bullish, bearish)

        if debug_mode and debug_log:
            st.divider()
            st.subheader("🐛 Debug Log")
            final = [l for l in debug_log if "Hourly" in l or "✅" in l or "FAILED" in l]
            with st.expander(f"Final stage ({len(final)} stocks)", expanded=True):
                st.text("\n\n".join(final[:300]))
            with st.expander("Full log"):
                st.text("\n\n".join(debug_log[:500]))


# ═══════════════════════════════════════════════════════════
# TAB 2 — HISTORICAL BACKTEST (bar-by-bar replay)
# ═══════════════════════════════════════════════════════════
with tab_historical_bt:
    st.subheader("📊 Historical Bar-by-Bar Backtest")
    st.markdown(
        "Replays the **Daily + Weekly HA signal logic** bar-by-bar across every stock in Nifty 500. "
        "Entry is the **next bar's open** (no look-ahead bias). "
        "Returns measured at each hold horizon."
    )

    if instrument_df.empty:
        st.error("Instrument data not loaded. Check API connection.")
    else:
        # ── Controls ──────────────────────────────────────────────────────
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bt_period = st.selectbox(
                "Backtest period",
                ["Last 1 month", "Last 2 months", "Last 3 months", "Custom range"],
                index=0
            )
        with bc2:
            hold_days_sel = st.multiselect(
                "Hold horizons (days)",
                options=[3, 5, 10, 15, 20, 30],
                default=[5, 10, 20]
            )
        with bc3:
            filter_sig_bt = st.selectbox("Signal type", ["All", "Bullish only", "Bearish only"])

        # Date range
        if bt_period == "Last 1 month":
            bt_start = TODAY - datetime.timedelta(days=30)
            bt_end   = TODAY
        elif bt_period == "Last 2 months":
            bt_start = TODAY - datetime.timedelta(days=60)
            bt_end   = TODAY
        elif bt_period == "Last 3 months":
            bt_start = TODAY - datetime.timedelta(days=90)
            bt_end   = TODAY
        else:
            dc1, dc2 = st.columns(2)
            bt_start = dc1.date_input("From", value=TODAY - datetime.timedelta(days=30),
                                       max_value=TODAY - datetime.timedelta(days=1))
            bt_end   = dc2.date_input("To",   value=TODAY, max_value=TODAY)

        st.caption(f"📅 Backtest window: **{bt_start}** → **{bt_end}** "
                   f"({(bt_end - bt_start).days} calendar days)")

        if not hold_days_sel:
            st.warning("Select at least one hold horizon.")
        else:
            hold_days_tuple = tuple(sorted(hold_days_sel))

            # Optional: filter to specific symbols
            with st.expander("🔎 Filter to specific symbols (optional)"):
                all_syms     = sorted(instrument_df["tradingsymbol"].tolist())
                selected_sym = st.multiselect("Leave empty to scan all Nifty 500", all_syms)

            run_bt = st.button("▶️ Run Historical Backtest", type="primary")

            if run_bt:
                bt_instr = instrument_df.copy()
                if selected_sym:
                    bt_instr = bt_instr[bt_instr["tradingsymbol"].isin(selected_sym)]

                n_stocks = len(bt_instr)
                st.info(
                    f"Running backtest on **{n_stocks} stocks** · "
                    f"window **{bt_start} → {bt_end}** · "
                    f"hold horizons: {hold_days_tuple}"
                )

                bt_progress = st.progress(0)
                bt_status   = st.empty()

                start_bt = time.time()
                bt_df = run_historical_backtest(
                    bt_instr, bt_start, bt_end,
                    hold_days_tuple, bt_progress, bt_status
                )
                elapsed_bt = time.time() - start_bt

                bt_progress.progress(1.0)
                bt_status.success(
                    f"✅ Done in {elapsed_bt:.1f}s — "
                    f"{len(bt_df)} signals found across {n_stocks} stocks"
                )

                # Apply signal filter
                if filter_sig_bt == "Bullish only":
                    bt_df = bt_df[bt_df["signal"] == "bullish"]
                elif filter_sig_bt == "Bearish only":
                    bt_df = bt_df[bt_df["signal"] == "bearish"]

                st.divider()
                render_backtest_results(bt_df, hold_days_tuple)


# ═══════════════════════════════════════════════════════════
# TAB 3 — LIVE SCAN SIGNAL HISTORY
# ═══════════════════════════════════════════════════════════
with tab_live_history:
    st.subheader("📋 Live Scan Signal History")
    st.caption("Signals saved automatically each time you run a live scan.")

    history = load_signal_history()
    if not history:
        st.info("No history yet — run a live scan first.")
    else:
        hist_df = pd.DataFrame(history)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Signals",  len(hist_df))
        m2.metric("🟢 Bullish",     (hist_df["signal"] == "bullish").sum())
        m3.metric("🔴 Bearish",     (hist_df["signal"] == "bearish").sum())
        m4.metric("📅 Since",       hist_df["date"].min())

        st.divider()
        filt = st.selectbox("Filter", ["All", "Bullish", "Bearish"], key="hist_filt")
        disp = hist_df.copy()
        if filt != "All":
            disp = disp[disp["signal"] == filt.lower()]
        disp = disp.sort_values("date", ascending=False)
        st.dataframe(disp, use_container_width=True)

        csv = disp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "live_signal_history.csv", "text/csv")

        st.divider()
        with st.expander("⚠️ Danger Zone"):
            if st.button("🗑️ Clear All Live Signal History"):
                save_signal_history([])
                st.success("Cleared.")
                st.rerun()
