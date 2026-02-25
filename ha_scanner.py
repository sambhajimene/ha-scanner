"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   HA TRIPLE SCREEN — FULL BACKTEST (Real Kite API Data)                    ║
║   Buy-the-Dip Strategy | 90 Days | All F&O + Cash Stocks                   ║
║   Exit Options: 1H / EOD / Next-Day / 2:1 RR SL-TP                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
  python ha_backtest_kite.py

Output:
  ha_backtest_results.csv   — All signals with PnL
  ha_backtest_report.txt    — Summary report
"""

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import json, os, time, pytz

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY      = "z9rful06a9890v8m"
ACCESS_TOKEN = "VMNx0TpO2y0SGvu6iz3L6jBhi0kziF9V"

BACKTEST_DAYS = 90          # how many days to backtest
WARMUP_DAYS   = 80          # extra days for indicator warmup
MAX_WORKERS   = 8           # parallel threads
OUTPUT_CSV    = "ha_backtest_results.csv"
OUTPUT_RPT    = "ha_backtest_report.txt"
IST           = pytz.timezone("Asia/Kolkata")

# ── INDICATOR PARAMS ──────────────────────────────────────────────────────────
EMA_PERIOD   = 50
RSI_PERIOD   = 14
ADX_PERIOD   = 14
STOCH_K      = 14
STOCH_D      = 3
ADX_MIN      = 15
TOLERANCE    = 0.0002
EMA_PROX     = 0.015   # 1.5% max from hourly EMA
RSI_BULL_MIN = 45;  RSI_BULL_MAX = 62
RSI_BEAR_MIN = 38;  RSI_BEAR_MAX = 55

# Exit strategy params
SL_PCT  = 0.01   # 1% stop loss
TP_PCT  = 0.02   # 2% target (2:1 RR)

# ── STOCK UNIVERSE ────────────────────────────────────────────────────────────
STOCKS = {
    # F&O Stocks
    "RELIANCE":  {"sector":"Energy",     "seg":"F&O"},
    "TCS":       {"sector":"IT",         "seg":"F&O"},
    "HDFCBANK":  {"sector":"Banks",      "seg":"F&O"},
    "INFY":      {"sector":"IT",         "seg":"F&O"},
    "ICICIBANK": {"sector":"Banks",      "seg":"F&O"},
    "SBIN":      {"sector":"Banks",      "seg":"F&O"},
    "SUNPHARMA": {"sector":"Pharma",     "seg":"F&O"},
    "TITAN":     {"sector":"Consumer",   "seg":"F&O"},
    "BAJFINANCE":{"sector":"Financials", "seg":"F&O"},
    "HCLTECH":   {"sector":"IT",         "seg":"F&O"},
    "WIPRO":     {"sector":"IT",         "seg":"F&O"},
    "MARUTI":    {"sector":"Auto",       "seg":"F&O"},
    "HAL":       {"sector":"Defence",    "seg":"F&O"},
    "EICHERMOT": {"sector":"Auto",       "seg":"F&O"},
    "NTPC":      {"sector":"Power",      "seg":"F&O"},
    "POWERGRID": {"sector":"Power",      "seg":"F&O"},
    "TATASTEEL": {"sector":"Metals",     "seg":"F&O"},
    "BAJAJ-AUTO":{"sector":"Auto",       "seg":"F&O"},
    "PFC":       {"sector":"Power",      "seg":"F&O"},
    "ADANIPORTS":{"sector":"Infra",      "seg":"F&O"},
    "ADANIENT":  {"sector":"Infra",      "seg":"F&O"},
    "COALINDIA": {"sector":"Energy",     "seg":"F&O"},
    "ONGC":      {"sector":"Energy",     "seg":"F&O"},
    "LT":        {"sector":"Infra",      "seg":"F&O"},
    "AXISBANK":  {"sector":"Banks",      "seg":"F&O"},
    "KOTAKBANK": {"sector":"Banks",      "seg":"F&O"},
    "ITC":       {"sector":"FMCG",       "seg":"F&O"},
    "HINDUNILVR":{"sector":"FMCG",       "seg":"F&O"},
    "NESTLEIND": {"sector":"FMCG",       "seg":"F&O"},
    "TATAMOTORS":{"sector":"Auto",       "seg":"F&O"},
    "M&M":       {"sector":"Auto",       "seg":"F&O"},
    "BPCL":      {"sector":"Energy",     "seg":"F&O"},
    "IOC":       {"sector":"Energy",     "seg":"F&O"},
    "GRASIM":    {"sector":"Cement",     "seg":"F&O"},
    "ULTRACEMCO":{"sector":"Cement",     "seg":"F&O"},
    "SHREECEM":  {"sector":"Cement",     "seg":"F&O"},
    "ASIANPAINT":{"sector":"Consumer",   "seg":"F&O"},
    "PIDILITIND": {"sector":"Consumer",  "seg":"F&O"},
    "VOLTAS":    {"sector":"Consumer",   "seg":"F&O"},
    "MPHASIS":   {"sector":"IT",         "seg":"F&O"},
    "TECHM":     {"sector":"IT",         "seg":"F&O"},
    "LTIM":      {"sector":"IT",         "seg":"F&O"},
    "PERSISTENT":{"sector":"IT",         "seg":"F&O"},
    "IRCTC":     {"sector":"Services",   "seg":"F&O"},
    "IRFC":      {"sector":"Financials", "seg":"F&O"},
    "RECLTD":    {"sector":"Power",      "seg":"F&O"},
    "PETRONET":  {"sector":"Energy",     "seg":"F&O"},
    "GAIL":      {"sector":"Energy",     "seg":"F&O"},
    "TATACOMM":  {"sector":"Telecom",    "seg":"F&O"},
    "BHARTIARTL":{"sector":"Telecom",    "seg":"F&O"},
    # Cash Stocks
    "DRREDDY":   {"sector":"Pharma",     "seg":"Cash"},
    "CIPLA":     {"sector":"Pharma",     "seg":"Cash"},
    "LUPIN":     {"sector":"Pharma",     "seg":"Cash"},
    "DIVI":      {"sector":"Pharma",     "seg":"Cash"},
    "AUROBINDO": {"sector":"Pharma",     "seg":"Cash"},
    "TORNTPHARM":{"sector":"Pharma",     "seg":"Cash"},
    "GLENMARK":  {"sector":"Pharma",     "seg":"Cash"},
    "KANSAINER": {"sector":"Consumer",   "seg":"Cash"},
    "REDINGTON": {"sector":"IT",         "seg":"Cash"},
    "SUPRAJIT":  {"sector":"Auto",       "seg":"Cash"},
    "ROHLTD":    {"sector":"Industrials","seg":"Cash"},
    "BLUEDART":  {"sector":"Logistics",  "seg":"Cash"},
    "JYOTHYLAB": {"sector":"FMCG",       "seg":"Cash"},
    "SUNTV":     {"sector":"Media",      "seg":"Cash"},
    "GICRE":     {"sector":"Financials", "seg":"Cash"},
}

# ── KITE SETUP ────────────────────────────────────────────────────────────────
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

def get_instrument_token(symbol):
    """Get NSE instrument token for symbol"""
    try:
        instruments = kite.instruments("NSE")
        for inst in instruments:
            if inst["tradingsymbol"] == symbol and inst["segment"] == "NSE":
                return inst["instrument_token"]
    except Exception as e:
        print(f"  Token fetch error {symbol}: {e}")
    return None

# Cache tokens
_token_cache = {}
def get_token(sym):
    if sym not in _token_cache:
        _token_cache[sym] = get_instrument_token(sym)
    return _token_cache[sym]

def fetch_ohlc(symbol, interval, from_dt, to_dt):
    """Fetch OHLC data from Kite"""
    token = get_token(symbol)
    if not token:
        return None
    try:
        data = kite.historical_data(token, from_dt, to_dt, interval, continuous=False)
        if not data:
            return None
        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.rename(columns={"open":"open","high":"high","low":"low",
                                 "close":"close","volume":"volume"})
        df = df[["open","high","low","close","volume"]].copy()
        df = df[~df.index.duplicated(keep="first")]
        df = df.sort_index()
        return df
    except Exception as e:
        print(f"  Fetch error {symbol} {interval}: {e}")
        return None

# ── INDICATORS ────────────────────────────────────────────────────────────────
def compute_ha(df):
    ha = pd.DataFrame(index=df.index)
    ha["ha_close"] = (df["open"]+df["high"]+df["low"]+df["close"])/4
    op = [(df["open"].iloc[0]+df["close"].iloc[0])/2]
    for i in range(1, len(df)):
        op.append((op[i-1]+ha["ha_close"].iloc[i-1])/2)
    ha["ha_open"] = op
    ha["ha_high"] = df[["high"]].join(ha[["ha_open","ha_close"]]).max(axis=1)
    ha["ha_low"]  = df[["low"]].join(ha[["ha_open","ha_close"]]).min(axis=1)
    return ha

def compute_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def compute_rsi(close, p=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100/(1 + g/l.replace(0, float("inf")))

def compute_adx(df, p=14):
    h,l,c = df["high"],df["low"],df["close"]
    pdm   = h.diff().clip(lower=0);  ndm = (-l.diff()).clip(lower=0)
    pdm   = pdm.where(pdm>ndm, 0);   ndm = ndm.where(ndm>pdm, 0)
    tr    = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr   = tr.ewm(com=p-1,adjust=False).mean()
    pdi   = 100*pdm.ewm(com=p-1,adjust=False).mean()/atr
    ndi   = 100*ndm.ewm(com=p-1,adjust=False).mean()/atr
    dx    = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0,float("inf"))
    return dx.ewm(com=p-1,adjust=False).mean()

def compute_stoch(df, k=14, d=3):
    lo = df["low"].rolling(k).min()
    hi = df["high"].rolling(k).max()
    fk = 100*(df["close"]-lo)/(hi-lo).replace(0,float("inf"))
    sk = fk.rolling(d).mean()
    sd = sk.rolling(d).mean()
    return sk, sd

# ── HA CHECKS ─────────────────────────────────────────────────────────────────
def check_weekly_ha(ha):
    def _ok(r, bull):
        o,h,l,c = r["ha_open"],r["ha_high"],r["ha_low"],r["ha_close"]
        tol = max(o,1)*TOLERANCE
        return (abs(o-l)<=tol and c>o) if bull else (abs(o-h)<=tol and c<o)
    curr = ha.iloc[-1]
    prev = ha.iloc[-2] if len(ha)>=2 else None
    for bull in (True, False):
        if _ok(curr, bull) or (prev is not None and _ok(prev, bull)):
            return "BULLISH" if bull else "BEARISH"
    return None

def check_daily_ha(ha):
    r = ha.iloc[-1]
    o,h,l,c = r["ha_open"],r["ha_high"],r["ha_low"],r["ha_close"]
    tol = max(o,1)*TOLERANCE
    if abs(o-l)<=tol and c>o:  return "BULLISH"
    if abs(o-h)<=tol and c<o:  return "BEARISH"
    return "NEUTRAL_BEAR" if c<o else "NEUTRAL_BULL"

# ── RIPPLE CHECK ──────────────────────────────────────────────────────────────
def check_ripple(idx, raw_h, ha_h, ema_h, rsi_h, adx_h, sk_h, sd_h, direction):
    ci = idx; pi = idx-1
    try:
        ha_c = ha_h.iloc[ci];  ha_p = ha_h.iloc[pi]
        rc   = raw_h.iloc[ci]
        ec   = float(ema_h.iloc[ci]); ep = float(ema_h.iloc[pi])
        rsi  = float(rsi_h.iloc[ci]); adx = float(adx_h.iloc[ci])
        sk   = float(sk_h.iloc[ci]);  sd  = float(sd_h.iloc[ci])
        cl   = float(rc["close"])
        tol  = max(float(ha_c["ha_open"]),1)*TOLERANCE
        ptol = max(float(ha_p["ha_open"]),1)*TOLERANCE
    except: return None

    # ① Strong HA candle
    if direction=="BULL":
        if not (ha_c["ha_close"]>ha_c["ha_open"] and abs(ha_c["ha_open"]-ha_c["ha_low"])<=tol):
            return None
    else:
        if not (ha_c["ha_close"]<ha_c["ha_open"] and abs(ha_c["ha_open"]-ha_c["ha_high"])<=tol):
            return None

    # Previous candle ANY 1 of 3
    if direction=="BULL":
        pa = ha_p["ha_open"]>=ha_p["ha_close"]
        pb = ha_p["ha_open"]>ha_p["ha_low"]+ptol
        pc = ha_p["ha_close"]<ep
    else:
        pa = ha_p["ha_open"]<=ha_p["ha_close"]
        pb = ha_p["ha_open"]<ha_p["ha_high"]-ptol
        pc = ha_p["ha_close"]>ep
    if not (pa or pb or pc): return None

    # ② EMA zone
    if direction=="BULL" and cl<ec: return None
    if direction=="BEAR" and cl>ec: return None

    # ③ RSI dip zone
    if direction=="BULL" and not (RSI_BULL_MIN<=rsi<=RSI_BULL_MAX): return None
    if direction=="BEAR" and not (RSI_BEAR_MIN<=rsi<=RSI_BEAR_MAX): return None

    # ④ ADX
    if adx<=ADX_MIN: return None

    # ⑤ Stochastic
    if direction=="BULL" and sk<=sd: return None
    if direction=="BEAR" and sk>=sd: return None

    # ⑥ EMA proximity 1.5%
    if direction=="BULL":
        if cl>ec*(1+EMA_PROX): return None
        prox = (cl-ec)/ec*100
    else:
        if cl<ec*(1-EMA_PROX): return None
        prox = (ec-cl)/ec*100

    return {
        "rsi":round(rsi,1), "adx":round(adx,1),
        "stoch_k":round(sk,1), "stoch_d":round(sd,1),
        "close":round(cl,2), "ema50":round(ec,2),
        "prox_pct":round(prox,2),
        "candle": "STRONG_BUY" if direction=="BULL" else "STRONG_SELL"
    }

# ── SIMULATE EXIT ─────────────────────────────────────────────────────────────
def simulate_exit(h_df, signal_idx, direction, entry_price):
    """
    Simulate 4 exit strategies:
    1. Next 1H candle close
    2. EOD (end of same trading day)
    3. Next day close (~3:20pm)
    4. SL/TP: 1% SL, 2% TP — whichever hits first in next candles
    """
    results = {}
    n = len(h_df)

    # Exit 1: Next 1H close
    if signal_idx+1 < n:
        results["exit_1h"] = round(float(h_df["close"].iloc[signal_idx+1]), 2)
    else:
        results["exit_1h"] = entry_price

    # Exit 2: EOD
    sig_date = h_df.index[signal_idx].date()
    day_bars = h_df[h_df.index.date == sig_date]
    results["exit_eod"] = round(float(day_bars["close"].iloc[-1]), 2) if len(day_bars)>0 else entry_price

    # Exit 3: Next day (last bar of next trading day)
    future_bars = h_df[h_df.index > h_df.index[signal_idx]]
    next_dates  = sorted(set(future_bars.index.date))
    if len(next_dates)>=1:
        nd_bars = h_df[h_df.index.date == next_dates[0]]
        results["exit_next"] = round(float(nd_bars["close"].iloc[-1]), 2)
    else:
        results["exit_next"] = results["exit_eod"]

    # Exit 4: SL/TP (scan bar by bar)
    if direction=="BULL":
        sl_price = entry_price * (1 - SL_PCT)
        tp_price = entry_price * (1 + TP_PCT)
    else:
        sl_price = entry_price * (1 + SL_PCT)
        tp_price = entry_price * (1 - TP_PCT)

    sl_hit = False; tp_hit = False; sltp_exit = entry_price
    for j in range(signal_idx+1, min(signal_idx+13, n)):  # up to ~2 days
        bar = h_df.iloc[j]
        if direction=="BULL":
            if bar["low"]  <= sl_price: sl_hit=True;  sltp_exit=sl_price; break
            if bar["high"] >= tp_price: tp_hit=True;  sltp_exit=tp_price; break
        else:
            if bar["high"] >= sl_price: sl_hit=True;  sltp_exit=sl_price; break
            if bar["low"]  <= tp_price: tp_hit=True;  sltp_exit=tp_price; break
    if not sl_hit and not tp_hit:
        sltp_exit = results["exit_next"]  # timeout = next day exit
    results["exit_sltp"] = round(sltp_exit, 2)
    results["sl_hit"]    = sl_hit
    results["tp_hit"]    = tp_hit

    # Calculate PnL for each exit
    def pnl(ep):
        p = (ep - entry_price)/entry_price*100
        return round(p if direction=="BULL" else -p, 3)

    results["pnl_1h"]   = pnl(results["exit_1h"])
    results["pnl_eod"]  = pnl(results["exit_eod"])
    results["pnl_next"] = pnl(results["exit_next"])
    results["pnl_sltp"] = pnl(results["exit_sltp"])
    return results

# ── PROCESS ONE STOCK ─────────────────────────────────────────────────────────
def process_stock(symbol):
    signals = []
    try:
        now_ist = datetime.now(IST)
        to_dt   = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        if now_ist.hour < 9:
            to_dt = to_dt - timedelta(days=1)

        total_days = BACKTEST_DAYS + WARMUP_DAYS
        from_daily  = to_dt - timedelta(days=total_days + 10)
        from_hourly = to_dt - timedelta(days=BACKTEST_DAYS + 10)

        # Fetch data
        raw_d = fetch_ohlc(symbol, "day",   from_daily.date(),  to_dt.date())
        raw_h = fetch_ohlc(symbol, "60minute", from_hourly.date(), to_dt.date())

        if raw_d is None or raw_h is None: return signals
        if len(raw_d) < EMA_PERIOD+5 or len(raw_h) < EMA_PERIOD+25: return signals

        # Compute hourly indicators
        ha_h  = compute_ha(raw_h)
        ema_h = compute_ema(raw_h["close"], EMA_PERIOD)
        rsi_h = compute_rsi(raw_h["close"], RSI_PERIOD)
        adx_h = compute_adx(raw_h, ADX_PERIOD)
        sk_h, sd_h = compute_stoch(raw_h, STOCH_K, STOCH_D)

        bt_start = now_ist - timedelta(days=BACKTEST_DAYS)
        bt_start_ts = pd.Timestamp(bt_start).tz_localize(IST) if bt_start.tzinfo is None else pd.Timestamp(bt_start)

        min_i    = max(EMA_PERIOD+20, 30)
        date_cache = {}
        seen_keys  = set()

        for i in range(min_i, len(raw_h)-2):
            bar_time = raw_h.index[i]
            # Convert to comparable
            try:
                if hasattr(bar_time, 'tzinfo') and bar_time.tzinfo:
                    bt_cmp = bt_start_ts
                else:
                    bt_cmp = bt_start_ts.replace(tzinfo=None)
                if bar_time < bt_cmp: continue
            except: pass

            bar_date = bar_time.date()

            if bar_date not in date_cache:
                d_sl = raw_d[raw_d.index.date <= bar_date]
                if len(d_sl) < EMA_PERIOD+3:
                    date_cache[bar_date] = None; continue
                w_sl = d_sl.resample("W-FRI").agg(
                    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
                ).dropna()
                if len(w_sl) < 3:
                    date_cache[bar_date] = None; continue
                wsig  = check_weekly_ha(compute_ha(w_sl))
                dsig  = check_daily_ha(compute_ha(d_sl))
                ema_d = compute_ema(d_sl["close"], EMA_PERIOD)
                en    = float(ema_d.iloc[-1]); ep_d = float(ema_d.iloc[-2])
                cd    = float(d_sl["close"].iloc[-1])
                wema  = compute_ema(w_sl["close"], EMA_PERIOD)
                w_en  = float(wema.iloc[-1])
                w_cd  = float(w_sl["close"].iloc[-1])
                date_cache[bar_date] = (wsig, dsig, en, ep_d, cd, w_en, w_cd)

            cached = date_cache.get(bar_date)
            if not cached: continue
            wsig, dsig, en, ep_d, cd, w_en, w_cd = cached
            if not wsig: continue

            bull_ok = (wsig=="BULLISH" and dsig in ("BULLISH","NEUTRAL_BULL")
                       and cd>en and en>ep_d and (cd-en)/en<=0.05)
            bear_ok = (wsig=="BEARISH" and dsig in ("BEARISH","NEUTRAL_BEAR")
                       and cd<en and en<ep_d)
            if not (bull_ok or bear_ok): continue

            direction = "BULL" if bull_ok else "BEAR"
            r = check_ripple(i, raw_h, ha_h, ema_h, rsi_h, adx_h, sk_h, sd_h, direction)
            if not r: continue

            dir_str = "BULLISH" if direction=="BULL" else "BEARISH"
            sig_key = f"{symbol}_{bar_date}_{dir_str}"
            if sig_key in seen_keys: continue
            seen_keys.add(sig_key)

            # Simulate exits
            exits = simulate_exit(raw_h, i, direction, r["close"])
            meta  = STOCKS.get(symbol, {})

            sig = {
                "symbol":    symbol,
                "sector":    meta.get("sector","Unknown"),
                "seg":       meta.get("seg","Cash"),
                "date":      str(bar_date),
                "time":      bar_time.strftime("%Y-%m-%d %H:%M"),
                "direction": dir_str,
                "entry":     r["close"],
                "ema50":     r["ema50"],
                "rsi":       r["rsi"],
                "adx":       r["adx"],
                "stoch_k":   r["stoch_k"],
                "stoch_d":   r["stoch_d"],
                "prox_pct":  r["prox_pct"],
                "weekly":    wsig,
                "daily":     dsig,
                "d_ema50":   round(en,2),
                "d_close":   round(cd,2),
                "d_ema_dist":round((cd-en)/en*100,2),
                # Exits
                "exit_1h":   exits["exit_1h"],
                "exit_eod":  exits["exit_eod"],
                "exit_next": exits["exit_next"],
                "exit_sltp": exits["exit_sltp"],
                "sl_hit":    exits["sl_hit"],
                "tp_hit":    exits["tp_hit"],
                # PnL
                "pnl_1h":    exits["pnl_1h"],
                "pnl_eod":   exits["pnl_eod"],
                "pnl_next":  exits["pnl_next"],
                "pnl_sltp":  exits["pnl_sltp"],
            }
            signals.append(sig)

    except Exception as e:
        print(f"  Error {symbol}: {e}")
    return signals

# ── MAIN BACKTEST RUNNER ──────────────────────────────────────────────────────
def run_backtest():
    print("="*70)
    print("  HA TRIPLE SCREEN — BACKTEST STARTING")
    print(f"  Period: Last {BACKTEST_DAYS} trading days")
    print(f"  Universe: {len(STOCKS)} stocks")
    print(f"  Strategy: Buy-the-Dip | RSI 45-62 | EMA±1.5%")
    print("="*70)

    # Pre-load instrument tokens
    print("\nLoading instrument tokens...")
    try:
        instruments = kite.instruments("NSE")
        for inst in instruments:
            sym = inst["tradingsymbol"]
            if sym in STOCKS and inst["segment"]=="NSE":
                _token_cache[sym] = inst["instrument_token"]
        print(f"  Tokens loaded: {len(_token_cache)}/{len(STOCKS)}")
    except Exception as e:
        print(f"  Token load error: {e}")

    all_signals = []
    symbols = list(STOCKS.keys())

    print(f"\nFetching data + scanning {len(symbols)} stocks...\n")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_stock, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym  = futures[fut]
            sigs = fut.result()
            if sigs:
                print(f"  ✅ {sym:<14} → {len(sigs)} signals")
                all_signals.extend(sigs)
            else:
                print(f"  ○  {sym:<14} → 0 signals")
            time.sleep(0.05)

    if not all_signals:
        print("\n⚠ No signals found. Check API token or date range.")
        return

    df = pd.DataFrame(all_signals)
    df = df.sort_values("time")

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Results saved: {OUTPUT_CSV} ({len(df)} signals)")

    # ── GENERATE REPORT ───────────────────────────────────────────────────────
    report = generate_report(df)
    with open(OUTPUT_RPT, "w") as f:
        f.write(report)
    print(f"✅ Report saved: {OUTPUT_RPT}")
    print("\n" + report)

def generate_report(df):
    bull = df[df["direction"]=="BULLISH"]
    bear = df[df["direction"]=="BEARISH"]

    lines = []
    lines.append("="*70)
    lines.append("  HA TRIPLE SCREEN — BACKTEST REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Period: Last {BACKTEST_DAYS} trading days")
    lines.append("="*70)

    def stats(sub, label):
        if len(sub)==0: return
        lines.append(f"\n{'─'*50}")
        lines.append(f"  {label} ({len(sub)} signals)")
        lines.append(f"{'─'*50}")
        for col, name in [("pnl_1h","1H Exit"), ("pnl_eod","EOD Exit"),
                          ("pnl_next","Next Day"), ("pnl_sltp","SL/TP 1%/2%")]:
            s     = sub[col]
            wins  = (s>0).sum(); loss = (s<=0).sum()
            wr    = wins/len(s)*100 if len(s)>0 else 0
            avg   = s.mean(); tot = s.sum()
            lines.append(f"  {name:<12}: WR={wr:5.1f}%  Avg={avg:+.3f}%  "
                         f"Win={wins:3d}  Loss={loss:3d}  Total={tot:+.2f}%")
        if "sl_hit" in sub.columns:
            sl_c = sub["sl_hit"].sum(); tp_c = sub["tp_hit"].sum()
            to_c = len(sub)-sl_c-tp_c
            lines.append(f"  SL/TP breakdown: SL={sl_c}  TP={tp_c}  Timeout={to_c}")

    stats(bull, "BULLISH (Long)")
    stats(bear, "BEARISH (Short)")
    stats(df,   "ALL SIGNALS COMBINED")

    # Sector breakdown
    lines.append(f"\n{'─'*50}")
    lines.append("  SECTOR BREAKDOWN (pnl_sltp)")
    lines.append(f"{'─'*50}")
    for sec, grp in df.groupby("sector"):
        wr = (grp["pnl_sltp"]>0).mean()*100
        avg= grp["pnl_sltp"].mean()
        lines.append(f"  {sec:<15} n={len(grp):3d}  WR={wr:5.1f}%  Avg={avg:+.3f}%")

    # Best signals
    lines.append(f"\n{'─'*50}")
    lines.append("  TOP 10 SIGNALS (by pnl_sltp)")
    lines.append(f"{'─'*50}")
    top = df.nlargest(10,"pnl_sltp")[["date","symbol","direction","entry","rsi","prox_pct","pnl_sltp"]]
    for _, r in top.iterrows():
        lines.append(f"  {r['date']} {r['symbol']:<12} {r['direction']:<8} "
                     f"₹{r['entry']:>8.1f}  RSI:{r['rsi']:5.1f}  "
                     f"EMA:{r['prox_pct']:+5.2f}%  PnL:{r['pnl_sltp']:+.2f}%")

    # Worst signals
    lines.append(f"\n{'─'*50}")
    lines.append("  BOTTOM 10 SIGNALS (by pnl_sltp)")
    lines.append(f"{'─'*50}")
    bot = df.nsmallest(10,"pnl_sltp")[["date","symbol","direction","entry","rsi","prox_pct","pnl_sltp"]]
    for _, r in bot.iterrows():
        lines.append(f"  {r['date']} {r['symbol']:<12} {r['direction']:<8} "
                     f"₹{r['entry']:>8.1f}  RSI:{r['rsi']:5.1f}  "
                     f"EMA:{r['prox_pct']:+5.2f}%  PnL:{r['pnl_sltp']:+.2f}%")

    # Monthly breakdown
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    lines.append(f"\n{'─'*50}")
    lines.append("  MONTHLY BREAKDOWN (pnl_sltp)")
    lines.append(f"{'─'*50}")
    for mo, grp in df.groupby("month"):
        wr  = (grp["pnl_sltp"]>0).mean()*100
        avg = grp["pnl_sltp"].mean()
        b   = (grp["direction"]=="BULLISH").sum()
        s   = (grp["direction"]=="BEARISH").sum()
        lines.append(f"  {mo}  n={len(grp):3d}  Bull={b:2d}  Bear={s:2d}  "
                     f"WR={wr:5.1f}%  Avg={avg:+.3f}%")

    lines.append("\n" + "="*70)
    return "\n".join(lines)

if __name__ == "__main__":
    run_backtest()
