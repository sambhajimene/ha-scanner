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

MAX_WORKERS         = 8      # Keep low — Zerodha rate limit ~3 req/sec, 3 calls/stock
MIN_VOLUME          = 100_000
RETRY_DELAY         = 1.5    # seconds to wait on rate limit before retry
EMA_NEAR_PCT        = 0.03   # Daily HA close must be within 3% of EMA50 ("near EMA")

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

def _check_daily_signal(ha_row, ema50_val):
    """
    Daily HA signal — exact same logic as live scanner.
    BULLISH: (strong bull OR neutral) AND HA_Close > EMA50
    BEARISH: (strong bear OR neutral) AND HA_Close < EMA50
    """
    strong_bull = is_strong_bull_ha(ha_row)
    strong_bear = is_strong_bear_ha(ha_row)
    neutral     = is_neutral_ha(ha_row)

    above_ema = ha_row["HA_Close"] > ema50_val
    below_ema = ha_row["HA_Close"] < ema50_val

    if (strong_bull or neutral) and above_ema:
        return "bullish"
    if (strong_bear or neutral) and below_ema:
        return "bearish"
    return None


def _check_weekly_signal(ha_weekly_row):
    """
    Weekly HA signal — exact same logic as live scanner.
    BULLISH: HA Open == HA Low AND HA Close > HA Open
    BEARISH: HA Open == HA High AND HA Close < HA Open
    """
    if is_strong_bull_ha(ha_weekly_row):
        return "bullish"
    if is_strong_bear_ha(ha_weekly_row):
        return "bearish"
    return None


def historical_backtest_symbol(token, symbol, bt_start, bt_end, hold_days):
    """
    CORRECT BACKTEST LOGIC:
    ─────────────────────────────────────────────────────────────
    SIGNAL  : Daily HA + Weekly HA agree (same as live scanner)
    ENTRY   : Signal candle ke baad PEHLA hourly HA candle ka OPEN
    EXIT    : Hourly HA bars walk karo bar-by-bar:
                Bullish  → exit jab hourly HA_Close < Hourly EMA50
                Bearish  → exit jab hourly HA_Close > Hourly EMA50
    ─────────────────────────────────────────────────────────────
    """
    try:
        # ── Fetch data ────────────────────────────────────────────────────

        warmup_days  = 200   # EMA50 needs 50 bars warmup + buffer
        warmup_start = bt_start - datetime.timedelta(days=warmup_days)
        hourly_start = bt_start - datetime.timedelta(days=15)  # small warmup for hourly EMA50

        # Daily bars (signal detection)
        daily_raw = fetch_with_retry(token, warmup_start, bt_end, "day")
        if daily_raw.empty or len(daily_raw) < 70:
            return []

        # Weekly bars (trend confirmation)
        weekly_raw = fetch_with_retry(token, bt_start - datetime.timedelta(days=400), bt_end, "week")
        if weekly_raw.empty or len(weekly_raw) < 15:
            return []

        # Hourly bars (entry + exit timing)
        # Fetch from bt_start - 15 days so EMA50 is warmed up by bt_start
        hourly_raw = fetch_with_retry(token, hourly_start, bt_end, "60minute")
        if hourly_raw.empty or len(hourly_raw) < 20:
            return []

        # ── Prepare DAILY ─────────────────────────────────────────────────
        daily_raw["date"]  = pd.to_datetime(daily_raw["date"]).dt.date
        daily_raw["EMA50"] = daily_raw["close"].ewm(span=50).mean()
        ha_daily           = calculate_heikin_ashi(daily_raw)
        ha_daily["date"]   = daily_raw["date"].values
        ha_daily["EMA50"]  = daily_raw["EMA50"].values
        ha_daily           = ha_daily.reset_index(drop=True)

        # ── Prepare WEEKLY signal lookup ──────────────────────────────────
        weekly_raw["date"] = pd.to_datetime(weekly_raw["date"]).dt.date
        ha_weekly          = calculate_heikin_ashi(weekly_raw)
        ha_weekly["date"]  = weekly_raw["date"].values

        weekly_sig_map = {}
        for _, wrow in ha_weekly.iterrows():
            weekly_sig_map[wrow["date"]] = _check_weekly_signal(wrow)

        def get_weekly_sig(d):
            candidates = [k for k in weekly_sig_map if k <= d]
            return weekly_sig_map[max(candidates)] if candidates else None

        # ── Prepare HOURLY ────────────────────────────────────────────────
        hourly_raw["datetime"] = pd.to_datetime(hourly_raw["date"])
        hourly_raw["date"]     = hourly_raw["datetime"].dt.date
        hourly_raw["EMA50"]    = hourly_raw["close"].ewm(span=50).mean()
        ha_hourly              = calculate_heikin_ashi(hourly_raw)
        ha_hourly["datetime"]  = hourly_raw["datetime"].values
        ha_hourly["date"]      = hourly_raw["date"].values
        ha_hourly["EMA50"]     = hourly_raw["EMA50"].values
        ha_hourly              = ha_hourly.reset_index(drop=True)

        # Build date → list of hourly bar indices  (for fast lookup)
        date_to_hourly_idx = {}
        for idx, row in ha_hourly.iterrows():
            d = row["date"]
            date_to_hourly_idx.setdefault(d, []).append(idx)

        # ── Walk forward daily bars (signal detection) ────────────────────
        signals_found = []
        last_signal   = None  # cooldown: skip consecutive same-direction signals

        for i, drow in ha_daily.iterrows():
            bar_date = drow["date"]
            if bar_date < bt_start or bar_date > bt_end:
                continue

            # Check daily + weekly agreement
            daily_sig  = _check_daily_signal(drow, drow["EMA50"])
            weekly_sig = get_weekly_sig(bar_date)

            if daily_sig is None or weekly_sig is None or daily_sig != weekly_sig:
                last_signal = None
                continue

            signal = daily_sig
            if signal == last_signal:
                continue   # cooldown

            # ── Find first hourly bar AFTER this daily signal candle ──────
            # Signal candle closes at end of bar_date.
            # Entry = OPEN of first hourly candle on the NEXT trading date.

            # Get all hourly bar indices after bar_date
            future_hourly = [
                idx for d, idxs in date_to_hourly_idx.items()
                if d > bar_date
                for idx in idxs
            ]
            if not future_hourly:
                continue

            first_hourly_idx = min(future_hourly)  # earliest bar after signal
            first_hourly_bar = ha_hourly.iloc[first_hourly_idx]

            entry_price    = float(first_hourly_bar["open"])
            entry_datetime = first_hourly_bar["datetime"]
            entry_date     = first_hourly_bar["date"]

            if entry_price == 0:
                continue

            # ── Walk hourly bars to find EXIT ─────────────────────────────
            # Bullish  → exit when hourly HA_Close crosses BELOW EMA50
            # Bearish  → exit when hourly HA_Close crosses ABOVE EMA50
            exit_price    = None
            exit_datetime = None
            exit_bars     = 0   # how many hourly bars held
            max_gain_pct  = 0.0
            max_loss_pct  = 0.0

            for j in range(first_hourly_idx + 1, len(ha_hourly)):
                hbar      = ha_hourly.iloc[j]
                ha_close  = float(hbar["HA_Close"])
                ema_val   = float(hbar["EMA50"])
                bar_close = float(hbar["close"])
                exit_bars += 1

                # Track max gain / max loss (using actual close, not HA)
                pct_now = (bar_close - entry_price) / entry_price * 100
                if signal == "bullish":
                    max_gain_pct = max(max_gain_pct,  pct_now)
                    max_loss_pct = min(max_loss_pct,  pct_now)
                else:
                    max_gain_pct = max(max_gain_pct, -pct_now)
                    max_loss_pct = min(max_loss_pct, -pct_now)

                # Exit condition
                exit_triggered = (
                    (signal == "bullish" and ha_close < ema_val) or
                    (signal == "bearish" and ha_close > ema_val)
                )

                if exit_triggered:
                    exit_price    = bar_close  # exit at actual close of exit bar
                    exit_datetime = hbar["datetime"]
                    break

            # If no exit found before data ends → mark as open trade
            if exit_price is None:
                last_hourly      = ha_hourly.iloc[-1]
                exit_price       = float(last_hourly["close"])
                exit_datetime    = last_hourly["datetime"]
                trade_still_open = True
            else:
                trade_still_open = False

            # ── Calculate P&L ─────────────────────────────────────────────
            mult       = 1 if signal == "bullish" else -1
            return_pct = mult * (exit_price - entry_price) / entry_price * 100

            signals_found.append({
                "signal_date":    str(bar_date),
                "entry_datetime": str(entry_datetime),
                "exit_datetime":  str(exit_datetime),
                "symbol":         symbol,
                "signal":         signal,
                "entry_price":    round(entry_price, 2),
                "exit_price":     round(exit_price, 2),
                "return_%":       round(return_pct, 2),
                "max_gain_%":     round(max_gain_pct, 2),
                "max_loss_%":     round(max_loss_pct, 2),
                "hold_hours":     exit_bars,
                "trade_open":     trade_still_open,  # True = no exit signal yet
            })

            last_signal = signal

        return signals_found

    except Exception as e:
        return []


def run_historical_backtest(instrument_df, bt_start, bt_end,
                             progress_bar=None, status_text=None):
    """Run historical backtest across all stocks in parallel."""
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
                bt_start, bt_end,
                (5, 10, 20)
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
                status_text.info(
                    f"Backtesting `{done}/{total}` stocks · "
                    f"{len(all_rows)} signals found"
                )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values("signal_date", ascending=False).reset_index(drop=True)
    return df


# ================= BACKTEST RESULTS UI =================
def render_backtest_results(bt_df):
    """
    Display backtest results.
    Columns expected:
      signal_date, entry_datetime, exit_datetime, symbol, signal,
      entry_price, exit_price, return_%, max_gain_%, max_loss_%,
      hold_hours, trade_open
    """
    if bt_df.empty:
        st.warning("No signals found in this period. Try a wider date range or relax filters.")
        return

    ret_col = "return_%"

    # ── Summary metrics ───────────────────────────────────────────────────
    closed  = bt_df[~bt_df["trade_open"]] if "trade_open" in bt_df.columns else bt_df
    open_tr = bt_df[bt_df["trade_open"]]  if "trade_open" in bt_df.columns else pd.DataFrame()

    bull_bt = bt_df[bt_df["signal"] == "bullish"]
    bear_bt = bt_df[bt_df["signal"] == "bearish"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📋 Total Signals",  len(bt_df))
    m2.metric("🟢 Bullish",        len(bull_bt))
    m3.metric("🔴 Bearish",        len(bear_bt))
    m4.metric("✅ Closed Trades",  len(closed))
    m5.metric("⏳ Still Open",     len(open_tr))

    if ret_col in closed.columns and len(closed) > 0:
        series   = closed[ret_col].dropna()
        wins     = (series > 0).sum()
        win_rate = wins / len(series) * 100 if len(series) > 0 else 0

        st.divider()
        st.subheader("📈 Performance Summary (Closed Trades Only)")
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Win Rate",   f"{win_rate:.1f}%")
        s2.metric("Avg Return", f"{series.mean():.2f}%")
        s3.metric("Median",     f"{series.median():.2f}%")
        s4.metric("Best Trade", f"{series.max():.2f}%")
        s5.metric("Worst Trade",f"{series.min():.2f}%")
        s6.metric("Avg Hold",   f"{closed['hold_hours'].mean():.0f}h" if "hold_hours" in closed.columns else "N/A")

        # Bull vs Bear breakdown
        for side, sub, color in [("Bullish", bull_bt, "🟢"), ("Bearish", bear_bt, "🔴")]:
            sub_closed = sub[~sub["trade_open"]] if "trade_open" in sub.columns else sub
            if len(sub_closed) == 0:
                continue
            s = sub_closed[ret_col].dropna()
            wr = (s > 0).mean() * 100
            st.caption(
                f"{color} {side}: {len(sub_closed)} trades · "
                f"Win Rate {wr:.1f}% · Avg {s.mean():.2f}% · "
                f"Best {s.max():.2f}% · Worst {s.min():.2f}%"
            )

    # ── Charts ────────────────────────────────────────────────────────────
    if PLOTLY_AVAILABLE and ret_col in closed.columns and len(closed) > 0:
        st.divider()
        tab_eq, tab_dist, tab_heatmap, tab_sym = st.tabs([
            "📈 Equity Curve", "📊 Distribution", "🗓 Monthly Heatmap", "🏆 By Symbol"
        ])

        with tab_eq:
            eq = closed[["signal_date", "signal", ret_col]].dropna().sort_values("signal_date")
            fig = go.Figure()
            for sig, color in [("bullish", "limegreen"), ("bearish", "tomato")]:
                sub = eq[eq["signal"] == sig].copy()
                if sub.empty: continue
                sub["cumret"] = sub[ret_col].cumsum()
                fig.add_trace(go.Scatter(
                    x=sub["signal_date"], y=sub["cumret"],
                    name=sig.capitalize(), mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=4)
                ))
            all_eq = eq.copy()
            all_eq["cumret"] = all_eq[ret_col].cumsum()
            fig.add_trace(go.Scatter(
                x=all_eq["signal_date"], y=all_eq["cumret"],
                name="Combined", mode="lines",
                line=dict(color="royalblue", width=2, dash="dot")
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Cumulative Returns (Exit = Hourly HA crosses EMA50)",
                xaxis_title="Signal Date", yaxis_title="Cumulative Return %",
                height=420, template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab_dist:
            vals   = closed[ret_col].dropna()
            colors = ["green" if v > 0 else "red" for v in vals]
            fig2   = go.Figure()
            fig2.add_trace(go.Histogram(
                x=vals, marker_color=colors, nbinsx=40, opacity=0.8,
                name="Return %"
            ))
            fig2.add_vline(x=0, line_dash="dash", line_color="white")
            fig2.add_vline(x=vals.mean(), line_dash="dot", line_color="yellow",
                           annotation_text=f"Avg {vals.mean():.2f}%")
            fig2.update_layout(
                title="Return % Distribution (Closed Trades)",
                xaxis_title="Return %", height=400, template="plotly_dark"
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab_heatmap:
            hm = closed[["signal_date", "signal", ret_col]].dropna().copy()
            hm["month"] = pd.to_datetime(hm["signal_date"]).dt.to_period("M").astype(str)
            hm["win"]   = (hm[ret_col] > 0).astype(int)
            pivot = hm.pivot_table(index="signal", columns="month",
                                    values="win", aggfunc="mean") * 100
            if not pivot.empty:
                fig3 = go.Figure(go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=[s.capitalize() for s in pivot.index.tolist()],
                    colorscale="RdYlGn", zmin=0, zmax=100,
                    text=pivot.values.round(0), texttemplate="%{text}%",
                    colorbar=dict(title="Win Rate %")
                ))
                fig3.update_layout(title="Monthly Win Rate Heatmap",
                                    height=300, template="plotly_dark")
                st.plotly_chart(fig3, use_container_width=True)

        with tab_sym:
            sym_stats = (
                closed.groupby("symbol")[ret_col]
                .agg(Trades="count",
                     WinRate=lambda x: round((x > 0).mean() * 100, 1),
                     AvgReturn=lambda x: round(x.mean(), 2),
                     Best=lambda x: round(x.max(), 2),
                     Worst=lambda x: round(x.min(), 2))
                .sort_values("AvgReturn", ascending=False)
                .reset_index()
            )
            sym_stats.columns = ["Symbol", "Trades", "Win Rate %", "Avg Return %", "Best %", "Worst %"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🏆 Top 20 Symbols")
                st.dataframe(sym_stats.head(20), use_container_width=True)
            with c2:
                st.markdown("#### 💀 Bottom 20 Symbols")
                st.dataframe(sym_stats.tail(20), use_container_width=True)

    # ── All trades table ──────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 All Trades")
    disp_cols = ["signal_date", "entry_datetime", "exit_datetime",
                  "symbol", "signal", "entry_price", "exit_price",
                  "return_%", "max_gain_%", "max_loss_%", "hold_hours", "trade_open"]
    disp_cols = [c for c in disp_cols if c in bt_df.columns]

    def color_cell(val):
        if isinstance(val, (int, float)):
            return f"color: {'green' if val > 0 else 'red'}"
        return ""

    color_subset = [c for c in ["return_%", "max_gain_%", "max_loss_%"] if c in bt_df.columns]
    styled = bt_df[disp_cols].style.applymap(color_cell, subset=color_subset)
    st.dataframe(styled, use_container_width=True)

    csv = bt_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, "backtest_results.csv", "text/csv")

    # ── Top / Bottom trades ───────────────────────────────────────────────
    if ret_col in closed.columns and len(closed) > 0:
        st.divider()
        sorted_bt = closed.sort_values(ret_col, ascending=False)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🏆 Top 10 Trades")
            st.dataframe(
                sorted_bt.head(10)[["signal_date","symbol","signal",
                                     "entry_price","exit_price","return_%","hold_hours"]],
                use_container_width=True
            )
        with c2:
            st.markdown("#### 💀 Bottom 10 Trades")
            st.dataframe(
                sorted_bt.tail(10)[["signal_date","symbol","signal",
                                     "entry_price","exit_price","return_%","hold_hours"]],
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
def fetch_with_retry(token, from_date, to_date, interval, retries=3):
    """Fetch historical data with automatic retry on rate limit errors."""
    for attempt in range(retries):
        try:
            data = kite.historical_data(token, from_date, to_date, interval)
            return pd.DataFrame(data)
        except Exception as e:
            err = str(e).lower()
            if "too many requests" in err or "rate" in err or "429" in err:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise e
    return pd.DataFrame()


def calculate_rsi(series, period=14):
    """Standard RSI using EWM (Wilder's method)."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def is_strong_bull_ha(ha_row, tol=0.001):
    """
    Strong Bullish HA candle:
      - HA Open == HA Low  (no lower wick at all)
      - HA Close > HA Open (green candle)
    tol: relative tolerance for float equality (0.1%)
    """
    open_eq_low = abs(ha_row["HA_Open"] - ha_row["HA_Low"]) <= abs(ha_row["HA_Open"]) * tol
    return open_eq_low and (ha_row["HA_Close"] > ha_row["HA_Open"])


def is_strong_bear_ha(ha_row, tol=0.001):
    """
    Strong Bearish HA candle:
      - HA Open == HA High (no upper wick at all)
      - HA Close < HA Open (red candle)
    """
    open_eq_high = abs(ha_row["HA_Open"] - ha_row["HA_High"]) <= abs(ha_row["HA_Open"]) * tol
    return open_eq_high and (ha_row["HA_Close"] < ha_row["HA_Open"])


def is_neutral_ha(ha_row):
    """
    Neutral HA candle:
      - Both upper wick AND lower wick exist
      - Body can be any size (no restriction)
    """
    upper_wick = ha_row["HA_High"] - max(ha_row["HA_Open"], ha_row["HA_Close"])
    lower_wick = min(ha_row["HA_Open"], ha_row["HA_Close"]) - ha_row["HA_Low"]
    return upper_wick > 0 and lower_wick > 0


def scan_symbol(row, debug_mode=False):
    symbol = row["tradingsymbol"]
    token  = row["instrument_token"]
    log    = []

    try:

        # ══════════════════════════════════════════════════════════════════
        # STEP 1 — WEEKLY HA
        #
        # BULLISH: HA Open == HA Low  AND  HA Close > HA Open
        # BEARISH: HA Open == HA High AND  HA Close < HA Open
        # ══════════════════════════════════════════════════════════════════
        weekly    = fetch_with_retry(token, START_WEEKLY, END_DATE, "week")
        if weekly.empty or len(weekly) < 10:
            return None, f"{symbol}: insufficient weekly data"

        ha_weekly = calculate_heikin_ashi(weekly)
        w         = ha_weekly.iloc[-1]

        weekly_bull = is_strong_bull_ha(w)
        weekly_bear = is_strong_bear_ha(w)

        if debug_mode:
            log.append(
                f"{symbol} | WEEKLY HA → "
                f"open={w['HA_Open']:.2f}  low={w['HA_Low']:.2f}  "
                f"high={w['HA_High']:.2f}  close={w['HA_Close']:.2f} | "
                f"strong_bull={weekly_bull}  strong_bear={weekly_bear}"
            )

        if not (weekly_bull or weekly_bear):
            return None, "\n".join(log) if log else f"{symbol}: WEEKLY failed — HA open≠low/high"

        # ══════════════════════════════════════════════════════════════════
        # STEP 2 — DAILY HA
        #
        # BULLISH condition (both must pass):
        #   Candle:  strong bull (open==low, close>open)
        #         OR neutral    (both wicks exist, any body size)
        #   EMA:     HA candle (close) is ABOVE EMA50
        #
        # BEARISH condition (both must pass):
        #   Candle:  strong bear (open==high, close<open)
        #         OR neutral    (both wicks exist, any body size)
        #   EMA:     HA candle (close) is BELOW EMA50
        # ══════════════════════════════════════════════════════════════════
        daily    = fetch_with_retry(token, START_DAILY, END_DATE, "day")
        if daily.empty or len(daily) < 60:
            return None, f"{symbol}: insufficient daily data"

        daily["EMA50"] = daily["close"].ewm(span=50).mean()
        ha_daily       = calculate_heikin_ashi(daily)

        # Merge EMA50 into HA frame (same index)
        ha_daily["EMA50"] = daily["EMA50"].values

        d          = ha_daily.iloc[-1]
        ema50_d    = d["EMA50"]

        # Candle type
        d_strong_bull = is_strong_bull_ha(d)
        d_strong_bear = is_strong_bear_ha(d)
        d_neutral     = is_neutral_ha(d)

        # EMA condition:
        # BULLISH → HA Close above EMA50 AND within EMA_NEAR_PCT of EMA50
        # BEARISH → HA Close below EMA50 AND within EMA_NEAR_PCT of EMA50
        d_above_ema = d["HA_Close"] > ema50_d
        d_below_ema = d["HA_Close"] < ema50_d
        d_near_ema  = abs(d["HA_Close"] - ema50_d) / ema50_d <= EMA_NEAR_PCT

        daily_bull_ok = (d_strong_bull or d_neutral) and d_above_ema and d_near_ema
        daily_bear_ok = (d_strong_bear or d_neutral) and d_below_ema and d_near_ema

        if debug_mode:
            pct_dist = abs(d["HA_Close"] - ema50_d) / ema50_d * 100
            log.append(
                f"{symbol} | DAILY HA → "
                f"open={d['HA_Open']:.2f}  low={d['HA_Low']:.2f}  "
                f"high={d['HA_High']:.2f}  close={d['HA_Close']:.2f}  ema50={ema50_d:.2f} | "
                f"dist_from_ema={pct_dist:.2f}% (limit={EMA_NEAR_PCT*100:.0f}%) | "
                f"strong_bull={d_strong_bull}  neutral={d_neutral}  strong_bear={d_strong_bear} | "
                f"above={d_above_ema}  below={d_below_ema}  near={d_near_ema} | "
                f"daily_bull_ok={daily_bull_ok}  daily_bear_ok={daily_bear_ok}"
            )

        if not (daily_bull_ok or daily_bear_ok):
            return None, "\n".join(log) if log else f"{symbol}: DAILY failed"

        # ══════════════════════════════════════════════════════════════════
        # STEP 3 — HOURLY HA
        #
        # THREE conditions — ALL must pass:
        #
        # 1. SIGNAL CANDLE
        #    BULLISH: HA Open == HA Low AND HA Close > HA Open  (strong bull)
        #    BEARISH: HA Open == HA High AND HA Close < HA Open (strong bear)
        #
        # 2. EMA50 INTERACTION  (price near EMA or fake break)
        #    BULLISH: price near EMA50 (within 0.5% above)
        #          OR fake breakdown (hourly low dipped below EMA but close > EMA)
        #    BEARISH: price near EMA50 (within 0.5% below)
        #          OR fake breakout  (hourly high spiked above EMA but close < EMA)
        #
        # 3. RSI(14) between 40 and 60  (not overbought, not oversold)
        # ══════════════════════════════════════════════════════════════════
        hourly    = fetch_with_retry(token, START_HOURLY, END_DATE, "60minute")
        if hourly.empty or len(hourly) < 20:
            return None, f"{symbol}: insufficient hourly data"

        hourly["EMA50"] = hourly["close"].ewm(span=50).mean()
        hourly["RSI"]   = calculate_rsi(hourly["close"], period=14)
        ha_hourly       = calculate_heikin_ashi(hourly)
        ha_hourly["EMA50"] = hourly["EMA50"].values
        ha_hourly["RSI"]   = hourly["RSI"].values

        # Use last COMPLETED hourly bar
        h      = ha_hourly.iloc[-1]
        last_h = hourly.iloc[-1]   # raw bar for low/high EMA interaction
        ema_h  = h["EMA50"]
        rsi_h  = h["RSI"]

        # ── Condition 1: Signal candle ────────────────────────────────
        h_strong_bull = is_strong_bull_ha(h)
        h_strong_bear = is_strong_bear_ha(h)

        # ── Condition 2: EMA interaction ──────────────────────────────
        # BULLISH
        near_ema_bull  = (0 < (h["HA_Close"] - ema_h) / ema_h <= 0.005)   # close within 0.5% above EMA
        fake_breakdown = (last_h["low"] < ema_h) and (last_h["close"] > ema_h)
        ema_bull_ok    = near_ema_bull or fake_breakdown

        # BEARISH
        near_ema_bear  = (0 < (ema_h - h["HA_Close"]) / ema_h <= 0.005)   # close within 0.5% below EMA
        fake_breakout  = (last_h["high"] > ema_h) and (last_h["close"] < ema_h)
        ema_bear_ok    = near_ema_bear or fake_breakout

        # ── Condition 3: RSI 40–60 ────────────────────────────────────
        rsi_ok = 40 <= rsi_h <= 60

        if debug_mode:
            log.append(
                f"{symbol} | HOURLY HA → "
                f"open={h['HA_Open']:.2f}  low={h['HA_Low']:.2f}  "
                f"high={h['HA_High']:.2f}  close={h['HA_Close']:.2f}  "
                f"ema50={ema_h:.2f}  rsi={rsi_h:.1f} | "
                f"strong_bull={h_strong_bull}  strong_bear={h_strong_bear} | "
                f"near_ema_bull={near_ema_bull}  fake_breakdown={fake_breakdown}  ema_bull_ok={ema_bull_ok} | "
                f"near_ema_bear={near_ema_bear}  fake_breakout={fake_breakout}  ema_bear_ok={ema_bear_ok} | "
                f"rsi_ok(40≤{rsi_h:.1f}≤60)={rsi_ok}"
            )

        # ── FINAL SIGNAL ──────────────────────────────────────────────
        #
        # BULLISH = weekly_bull AND daily_bull_ok AND h_strong_bull AND ema_bull_ok AND rsi_ok
        # BEARISH = weekly_bear AND daily_bear_ok AND h_strong_bear AND ema_bear_ok AND rsi_ok
        #
        if weekly_bull and daily_bull_ok and h_strong_bull and ema_bull_ok and rsi_ok:
            log.append(f"✅ {symbol} → BULLISH")
            return ("bullish", symbol), "\n".join(log)

        if weekly_bear and daily_bear_ok and h_strong_bear and ema_bear_ok and rsi_ok:
            log.append(f"✅ {symbol} → BEARISH")
            return ("bearish", symbol), "\n".join(log)

        log.append(f"{symbol}: passed weekly+daily but FAILED hourly check")
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
    MIN_VOLUME  = st.number_input("Min Volume Filter", value=MIN_VOLUME, step=50_000, min_value=0)
    MAX_WORKERS = st.number_input("Parallel Workers",  value=MAX_WORKERS, step=1,
                                   min_value=1, max_value=15,
                                   help="Keep ≤ 10 to avoid Zerodha rate limit errors")
    EMA_NEAR_PCT = st.slider(
        "Daily: max % distance from EMA50",
        min_value=1, max_value=10, value=int(EMA_NEAR_PCT * 100), step=1,
        help="Daily HA close must be within this % of EMA50 to qualify as 'near EMA'. Default = 3%"
    ) / 100.0

    st.divider()
    debug_mode  = st.checkbox("🐛 Debug Mode", value=False)
    auto_mode   = st.checkbox("🔄 Auto Scan (hourly)", value=False)
    st.divider()
    st.markdown("**📐 Active Conditions**")
    st.caption("🟢 **Bullish**")
    st.caption("Weekly HA: open = low, close > open")
    st.caption("Daily HA: (strong OR neutral) + close > EMA50 + within {:.0f}% of EMA50".format(EMA_NEAR_PCT * 100))
    st.caption("Hourly HA: open = low, close > open + near/fake EMA50 + RSI 40–60")
    st.caption("🔴 **Bearish** = mirror opposite")
    st.divider()
    st.caption(f"Workers: {MAX_WORKERS} · Min Vol: {MIN_VOLUME:,} · EMA±{EMA_NEAR_PCT*100:.0f}%")

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
        bc1, bc2 = st.columns(2)
        with bc1:
            bt_period = st.selectbox(
                "Backtest period",
                ["Last 1 month", "Last 2 months", "Last 3 months", "Custom range"],
                index=0
            )
        with bc2:
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

        st.caption(
            f"\U0001f4c5 Backtest window: **{bt_start}** \u2192 **{bt_end}** "
            f"({(bt_end - bt_start).days} calendar days) | "
            f"Entry = next hourly HA open | Exit = hourly HA close crosses EMA50"
        )

        with st.expander("\U0001f50e Filter to specific symbols (optional)"):
            all_syms     = sorted(instrument_df["tradingsymbol"].tolist())
            selected_sym = st.multiselect("Leave empty to scan all Nifty 500", all_syms)

        run_bt = st.button("\u25b6\ufe0f Run Historical Backtest", type="primary")

        if run_bt:
            bt_instr = instrument_df.copy()
            if selected_sym:
                bt_instr = bt_instr[bt_instr["tradingsymbol"].isin(selected_sym)]

            n_stocks = len(bt_instr)
            st.info(f"Running backtest on **{n_stocks} stocks** | window **{bt_start} \u2192 {bt_end}**")

            bt_progress = st.progress(0)
            bt_status   = st.empty()

            start_bt = time.time()
            bt_df = run_historical_backtest(bt_instr, bt_start, bt_end, bt_progress, bt_status)
            elapsed_bt = time.time() - start_bt

            bt_progress.progress(1.0)
            bt_status.success(f"\u2705 Done in {elapsed_bt:.1f}s \u2014 {len(bt_df)} signals found across {n_stocks} stocks")

            if filter_sig_bt == "Bullish only":
                bt_df = bt_df[bt_df["signal"] == "bullish"]
            elif filter_sig_bt == "Bearish only":
                bt_df = bt_df[bt_df["signal"] == "bearish"]

            st.divider()
            render_backtest_results(bt_df)


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
