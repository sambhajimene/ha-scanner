"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   HA TRIPLE SCREEN SCANNER v2.0                                             ║
║   Cash + F&O + NSE500 | Parallel Fetch | Signal History | Email Alert       ║
║   Tide(W) + Wave(D) + Ripple(H): HA Candle + EMA50 + RSI + ADX             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from kiteconnect import KiteConnect
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import smtplib, ssl, json, os, time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pytz
import numpy as np

# ── Custom JSON encoder — handles numpy int64/float64 ─────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)

def sanitize(sig):
    """Convert all numpy scalar types to native Python types in a signal dict."""
    clean = {}
    for k, v in sig.items():
        if isinstance(v, (np.integer,)):  clean[k] = int(v)
        elif isinstance(v, (np.floating,)): clean[k] = float(v)
        else: clean[k] = v
    return clean

# ════════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════════
API_KEY       = "z9rful06a9890v8m"
ACCESS_TOKEN  = "ef8t3HtYR225C1vbkQk4k2H0PyowmvsA"
SMTP_SERVER   = "smtp.gmail.com"
SMTP_PORT     = 465
EMAIL_FROM    = "sambhajimene@gmail.com"
EMAIL_PASSWORD= "jgebigpsoeqqwrfa"
EMAIL_TO      = ["sambhajimene@gmail.com"]

HISTORY_FILE  = "signal_history.json"
DASHBOARD_OUT = "ha_dashboard.html"
MAX_WORKERS   = 12          # parallel threads for data fetch
HISTORY_DAYS  = 30          # signal history lookback
WARMUP_DAYS   = 80          # extra days for indicator warmup
IST           = pytz.timezone("Asia/Kolkata")

# ── INDICATOR PARAMS ──────────────────────────────────────────────────────────
EMA_PERIOD  = 50
RSI_PERIOD  = 14
ADX_PERIOD  = 14
ADX_MIN     = 15
TOLERANCE   = 0.0001  # 0.01% — HA wick equality
BODY_RATIO  = 0.60    # BUY/SELL: body must be ≥60% of range
WICK_RATIO  = 0.20    # BUY/SELL: each wick must be ≤20% of range

# ════════════════════════════════════════════════════════════════════════════════
#  STOCK UNIVERSE — Cash + F&O + NSE500 with Sector Tags
# ════════════════════════════════════════════════════════════════════════════════
STOCKS = {
    # ── NIFTY 50 ──────────────────────────────────────────────────────────────
    "RELIANCE"   :{"sector":"Energy",         "segment":"F&O"},
    "TCS"        :{"sector":"IT",             "segment":"F&O"},
    "HDFCBANK"   :{"sector":"Banks",          "segment":"F&O"},
    "INFY"       :{"sector":"IT",             "segment":"F&O"},
    "ICICIBANK"  :{"sector":"Banks",          "segment":"F&O"},
    "HINDUNILVR" :{"sector":"FMCG",           "segment":"F&O"},
    "SBIN"       :{"sector":"Banks",          "segment":"F&O"},
    "BHARTIARTL" :{"sector":"Telecom",        "segment":"F&O"},
    "ITC"        :{"sector":"FMCG",           "segment":"F&O"},
    "KOTAKBANK"  :{"sector":"Banks",          "segment":"F&O"},
    "LT"         :{"sector":"Industrials",    "segment":"F&O"},
    "AXISBANK"   :{"sector":"Banks",          "segment":"F&O"},
    "MARUTI"     :{"sector":"Auto",           "segment":"F&O"},
    "TITAN"      :{"sector":"Consumer",       "segment":"F&O"},
    "SUNPHARMA"  :{"sector":"Pharma",         "segment":"F&O"},
    "BAJFINANCE" :{"sector":"Financials",     "segment":"F&O"},
    "WIPRO"      :{"sector":"IT",             "segment":"F&O"},
    "ULTRACEMCO" :{"sector":"Cement",         "segment":"F&O"},
    "NTPC"       :{"sector":"Power",          "segment":"F&O"},
    "POWERGRID"  :{"sector":"Power",          "segment":"F&O"},
    "ONGC"       :{"sector":"Energy",         "segment":"F&O"},
    "TATASTEEL"  :{"sector":"Metals",         "segment":"F&O"},
    "JSWSTEEL"   :{"sector":"Metals",         "segment":"F&O"},
    "HINDALCO"   :{"sector":"Metals",         "segment":"F&O"},
    "ADANIENT"   :{"sector":"Industrials",    "segment":"F&O"},
    "ADANIPORTS" :{"sector":"Industrials",    "segment":"F&O"},
    "TMPV"       :{"sector":"Auto",           "segment":"F&O"},
    "M&M"        :{"sector":"Auto",           "segment":"F&O"},
    "BAJAJFINSV" :{"sector":"Financials",     "segment":"F&O"},
    "HCLTECH"    :{"sector":"IT",             "segment":"F&O"},
    "TECHM"      :{"sector":"IT",             "segment":"F&O"},
    "NESTLEIND"  :{"sector":"FMCG",           "segment":"F&O"},
    "CIPLA"      :{"sector":"Pharma",         "segment":"F&O"},
    "EICHERMOT"  :{"sector":"Auto",           "segment":"F&O"},
    "HEROMOTOCO" :{"sector":"Auto",           "segment":"F&O"},
    "BAJAJ-AUTO" :{"sector":"Auto",           "segment":"F&O"},
    "BRITANNIA"  :{"sector":"FMCG",           "segment":"F&O"},
    "COALINDIA"  :{"sector":"Mining",         "segment":"F&O"},
    "BPCL"       :{"sector":"Energy",         "segment":"F&O"},
    "GAIL"       :{"sector":"Energy",         "segment":"F&O"},
    "INDUSINDBK" :{"sector":"Banks",          "segment":"F&O"},
    "SHRIRAMFIN" :{"sector":"Financials",     "segment":"F&O"},
    "TRENT"      :{"sector":"Consumer",       "segment":"F&O"},
    "BEL"        :{"sector":"Defence",        "segment":"F&O"},
    "TATACONSUM" :{"sector":"FMCG",           "segment":"F&O"},
    "APOLLOHOSP" :{"sector":"Healthcare",     "segment":"F&O"},
    # ── F&O MIDCAP ────────────────────────────────────────────────────────────
    "AMBUJACEM"  :{"sector":"Cement",         "segment":"F&O"},
    "ACC"        :{"sector":"Cement",         "segment":"F&O"},
    "AUROPHARMA" :{"sector":"Pharma",         "segment":"F&O"},
    "BANKBARODA" :{"sector":"Banks",          "segment":"F&O"},
    "BERGEPAINT" :{"sector":"Consumer",       "segment":"F&O"},
    "BIOCON"     :{"sector":"Pharma",         "segment":"F&O"},
    "CANBK"      :{"sector":"Banks",          "segment":"F&O"},
    "CHOLAFIN"   :{"sector":"Financials",     "segment":"F&O"},
    "COLPAL"     :{"sector":"FMCG",           "segment":"F&O"},
    "CONCOR"     :{"sector":"Logistics",      "segment":"F&O"},
    "CUMMINSIND" :{"sector":"Industrials",    "segment":"F&O"},
    "DABUR"      :{"sector":"FMCG",           "segment":"F&O"},
    "DEEPAKNTR"  :{"sector":"Chemicals",      "segment":"F&O"},
    "DLF"        :{"sector":"Realty",         "segment":"F&O"},
    "FEDERALBNK" :{"sector":"Banks",          "segment":"F&O"},
    "GLENMARK"   :{"sector":"Pharma",         "segment":"F&O"},
    "GODREJCP"   :{"sector":"FMCG",           "segment":"F&O"},
    "GODREJPROP" :{"sector":"Realty",         "segment":"F&O"},
    "GUJGASLTD"  :{"sector":"Energy",         "segment":"F&O"},
    "HAL"        :{"sector":"Defence",        "segment":"F&O"},
    "HAVELLS"    :{"sector":"Consumer",       "segment":"F&O"},
    "HINDCOPPER" :{"sector":"Metals",         "segment":"F&O"},
    "HINDPETRO"  :{"sector":"Energy",         "segment":"F&O"},
    "IDFCFIRSTB" :{"sector":"Banks",          "segment":"F&O"},
    "IGL"        :{"sector":"Energy",         "segment":"F&O"},
    "INDHOTEL"   :{"sector":"Consumer",       "segment":"F&O"},
    "INDIGO"     :{"sector":"Aviation",       "segment":"F&O"},
    "INDUSTOWER" :{"sector":"Telecom",        "segment":"F&O"},
    "IRCTC"      :{"sector":"Services",       "segment":"F&O"},
    "JUBLFOOD"   :{"sector":"FMCG",           "segment":"F&O"},
    "LICHSGFIN"  :{"sector":"Financials",     "segment":"F&O"},
    "LUPIN"      :{"sector":"Pharma",         "segment":"F&O"},
    "MARICO"     :{"sector":"FMCG",           "segment":"F&O"},
    "MCX"        :{"sector":"Financials",     "segment":"F&O"},
    "MOTHERSON"  :{"sector":"Auto",           "segment":"F&O"},
    "MPHASIS"    :{"sector":"IT",             "segment":"F&O"},
    "MRF"        :{"sector":"Auto",           "segment":"F&O"},
    "MUTHOOTFIN" :{"sector":"Financials",     "segment":"F&O"},
    "NAUKRI"     :{"sector":"IT",             "segment":"F&O"},
    "NMDC"       :{"sector":"Mining",         "segment":"F&O"},
    "OIL"        :{"sector":"Energy",         "segment":"F&O"},
    "PAGEIND"    :{"sector":"Consumer",       "segment":"F&O"},
    "PERSISTENT" :{"sector":"IT",             "segment":"F&O"},
    "PETRONET"   :{"sector":"Energy",         "segment":"F&O"},
    "PFC"        :{"sector":"Power",          "segment":"F&O"},
    "PIDILITIND" :{"sector":"Chemicals",      "segment":"F&O"},
    "PNB"        :{"sector":"Banks",          "segment":"F&O"},
    "POLYCAB"    :{"sector":"Industrials",    "segment":"F&O"},
    "PVRINOX"    :{"sector":"Consumer",       "segment":"F&O"},
    "RECLTD"     :{"sector":"Power",          "segment":"F&O"},
    "SAIL"       :{"sector":"Metals",         "segment":"F&O"},
    "SIEMENS"    :{"sector":"Industrials",    "segment":"F&O"},
    "SRF"        :{"sector":"Chemicals",      "segment":"F&O"},
    "TATACHEM"   :{"sector":"Chemicals",      "segment":"F&O"},
    "TATACOMM"   :{"sector":"Telecom",        "segment":"F&O"},
    "TATAELXSI"  :{"sector":"IT",             "segment":"F&O"},
    "TATAPOWER"  :{"sector":"Power",          "segment":"F&O"},
    "TORNTPHARM" :{"sector":"Pharma",         "segment":"F&O"},
    "TORNTPOWER" :{"sector":"Power",          "segment":"F&O"},
    "TVSMOTOR"   :{"sector":"Auto",           "segment":"F&O"},
    "UPL"        :{"sector":"Chemicals",      "segment":"F&O"},
    "VEDL"       :{"sector":"Metals",         "segment":"F&O"},
    "VOLTAS"     :{"sector":"Consumer",       "segment":"F&O"},
    "ZOMATO"     :{"sector":"Consumer",       "segment":"F&O"},
    "ZYDUSLIFE"  :{"sector":"Pharma",         "segment":"F&O"},
    "LICI"       :{"sector":"Financials",     "segment":"F&O"},
    "IRFC"       :{"sector":"Financials",     "segment":"F&O"},
    # ── NSE500 CASH STOCKS ────────────────────────────────────────────────────
    "ASIANPAINT" :{"sector":"Consumer",       "segment":"Cash"},
    "BOSCHLTD"   :{"sector":"Auto",           "segment":"Cash"},
    "CROMPTON"   :{"sector":"Consumer",       "segment":"Cash"},
    "DIVISLAB"   :{"sector":"Pharma",         "segment":"Cash"},
    "DRREDDY"    :{"sector":"Pharma",         "segment":"Cash"},
    "GRANULES"   :{"sector":"Pharma",         "segment":"Cash"},
    "JINDALSTEL" :{"sector":"Metals",         "segment":"Cash"},
    "OBEROIRLTY" :{"sector":"Realty",         "segment":"Cash"},
    "OFSS"       :{"sector":"IT",             "segment":"Cash"},
    "IOC"        :{"sector":"Energy",         "segment":"Cash"},
    "LALPATHLAB" :{"sector":"Healthcare",     "segment":"Cash"},
    "METROPOLIS" :{"sector":"Healthcare",     "segment":"Cash"},
    "LAURUSLABS" :{"sector":"Pharma",         "segment":"Cash"},
    "JKCEMENT"   :{"sector":"Cement",         "segment":"Cash"},
    "IPCALAB"    :{"sector":"Pharma",         "segment":"Cash"},
    "RELAXO"     :{"sector":"Consumer",       "segment":"Cash"},
    "SUNDRMFAST" :{"sector":"Auto",           "segment":"Cash"},
    "AAVAS"      :{"sector":"Financials",     "segment":"Cash"},
    "ABCAPITAL"  :{"sector":"Financials",     "segment":"Cash"},
    "ALKEM"      :{"sector":"Pharma",         "segment":"Cash"},
    "APOLLOTYRE" :{"sector":"Auto",           "segment":"Cash"},
    "ASTRAL"     :{"sector":"Industrials",    "segment":"Cash"},
    "ATUL"       :{"sector":"Chemicals",      "segment":"Cash"},
    "BALRAMCHIN" :{"sector":"FMCG",           "segment":"Cash"},
    "BANDHANBNK" :{"sector":"Banks",          "segment":"Cash"},
    "BATAINDIA"  :{"sector":"Consumer",       "segment":"Cash"},
    "BHARATFORG" :{"sector":"Auto",           "segment":"Cash"},
    "BLUEDART"   :{"sector":"Logistics",      "segment":"Cash"},
    "BSOFT"      :{"sector":"IT",             "segment":"Cash"},
    "CANFINHOME" :{"sector":"Financials",     "segment":"Cash"},
    "CASTROLIND" :{"sector":"Energy",         "segment":"Cash"},
    "CDSL"       :{"sector":"Financials",     "segment":"Cash"},
    "CESC"       :{"sector":"Power",          "segment":"Cash"},
    "CHAMBLFERT" :{"sector":"Chemicals",      "segment":"Cash"},
    "COFORGE"    :{"sector":"IT",             "segment":"Cash"},
    "CRAFTSMAN"  :{"sector":"Industrials",    "segment":"Cash"},
    "DALBHARAT"  :{"sector":"Cement",         "segment":"Cash"},
    "DATAPATTNS" :{"sector":"Defence",        "segment":"Cash"},
    "EDELWEISS"  :{"sector":"Financials",     "segment":"Cash"},
    "EMAMILTD"   :{"sector":"FMCG",           "segment":"Cash"},
    "ENGINERSIN" :{"sector":"Industrials",    "segment":"Cash"},
    "EQUITASBNK" :{"sector":"Banks",          "segment":"Cash"},
    "FINEORG"    :{"sector":"Chemicals",      "segment":"Cash"},
    "FLUOROCHEM" :{"sector":"Chemicals",      "segment":"Cash"},
    "GAEL"       :{"sector":"FMCG",           "segment":"Cash"},
    "GHCL"       :{"sector":"Chemicals",      "segment":"Cash"},
    "GICRE"      :{"sector":"Financials",     "segment":"Cash"},
    "GLAXO"      :{"sector":"Pharma",         "segment":"Cash"},
    "GPIL"       :{"sector":"Metals",         "segment":"Cash"},
    "GRINDWELL"  :{"sector":"Industrials",    "segment":"Cash"},
    "GSPL"       :{"sector":"Energy",         "segment":"Cash"},
    "HONASA"     :{"sector":"Consumer",       "segment":"Cash"},
    "JYOTHYLAB"  :{"sector":"FMCG",           "segment":"Cash"},
    "KAJARIACER" :{"sector":"Consumer",       "segment":"Cash"},
    "KANSAINER"  :{"sector":"Consumer",       "segment":"Cash"},
    "KAYNES"     :{"sector":"Industrials",    "segment":"Cash"},
    "KPITTECH"   :{"sector":"IT",             "segment":"Cash"},
    "KRISHNANEK" :{"sector":"Consumer",       "segment":"Cash"},
    "LTFOODS"    :{"sector":"FMCG",           "segment":"Cash"},
    "MEDANTA"    :{"sector":"Healthcare",     "segment":"Cash"},
    "MFSL"       :{"sector":"Financials",     "segment":"Cash"},
    "NAVINFLUOR" :{"sector":"Chemicals",      "segment":"Cash"},
    "NCC"        :{"sector":"Industrials",    "segment":"Cash"},
    "NIACL"      :{"sector":"Financials",     "segment":"Cash"},
    "NIITLTD"    :{"sector":"IT",             "segment":"Cash"},
    "NLCINDIA"   :{"sector":"Power",          "segment":"Cash"},
    "PGHH"       :{"sector":"FMCG",           "segment":"Cash"},
    "PRAJIND"    :{"sector":"Industrials",    "segment":"Cash"},
    "RADICO"     :{"sector":"FMCG",           "segment":"Cash"},
    "RAJRATAN"   :{"sector":"Metals",         "segment":"Cash"},
    "RAYMOND"    :{"sector":"Consumer",       "segment":"Cash"},
    "REDINGTON"  :{"sector":"IT",             "segment":"Cash"},
    "RITES"      :{"sector":"Industrials",    "segment":"Cash"},
    "ROHLTD"     :{"sector":"Industrials",    "segment":"Cash"},
    "SAPPHIRE"   :{"sector":"Consumer",       "segment":"Cash"},
    "SCHNEIDER"  :{"sector":"Industrials",    "segment":"Cash"},
    "SONACOMS"   :{"sector":"Auto",           "segment":"Cash"},
    "STARHEALTH" :{"sector":"Financials",     "segment":"Cash"},
    "SUNTV"      :{"sector":"Media",          "segment":"Cash"},
    "SUPRAJIT"   :{"sector":"Auto",           "segment":"Cash"},
    "SURYAROSNI" :{"sector":"Industrials",    "segment":"Cash"},
    "SUZLON"     :{"sector":"Power",          "segment":"Cash"},
    "TANLA"      :{"sector":"IT",             "segment":"Cash"},
    "TATAINVEST" :{"sector":"Financials",     "segment":"Cash"},
    "THYROCARE"  :{"sector":"Healthcare",     "segment":"Cash"},
    "TIMETECHNO" :{"sector":"IT",             "segment":"Cash"},
    "TTKPRESTIG" :{"sector":"Consumer",       "segment":"Cash"},
    "UJJIVANSFB" :{"sector":"Banks",          "segment":"Cash"},
    "UTIAMC"     :{"sector":"Financials",     "segment":"Cash"},
    "VGUARD"     :{"sector":"Consumer",       "segment":"Cash"},
    "VINATIORGA" :{"sector":"Chemicals",      "segment":"Cash"},
    "WELCORP"    :{"sector":"Industrials",    "segment":"Cash"},
    "WELSPUNIND" :{"sector":"Consumer",       "segment":"Cash"},
    "WESTLIFE"   :{"sector":"Consumer",       "segment":"Cash"},
    "WONDERLA"   :{"sector":"Consumer",       "segment":"Cash"},
    "WOCKPHARMA" :{"sector":"Pharma",         "segment":"Cash"},
}

SECTOR_COLORS = {
    "Auto":"#38bdf8","Banks":"#f87171","IT":"#a78bfa","Pharma":"#34d399",
    "Energy":"#fb923c","FMCG":"#fbbf24","Metals":"#94a3b8","Power":"#22d3ee",
    "Financials":"#f472b6","Industrials":"#4ade80","Chemicals":"#facc15",
    "Defence":"#6ee7b7","Telecom":"#c084fc","Consumer":"#ff6b6b",
    "Cement":"#a8edea","Realty":"#fed9b7","Healthcare":"#b7e4c7",
    "Mining":"#d4a373","Logistics":"#90e0ef","Aviation":"#caf0f8",
    "Media":"#e9c46a","Services":"#264653","Default":"#64748b",
}

# ════════════════════════════════════════════════════════════════════════════════
#  KITE HELPERS
# ════════════════════════════════════════════════════════════════════════════════
def get_kite():
    k = KiteConnect(api_key=API_KEY)
    k.set_access_token(ACCESS_TOKEN)
    return k

def build_token_map(instruments):
    tm = {}
    for i in instruments:
        if i["segment"] == "NSE":
            tm[i["tradingsymbol"]] = i["instrument_token"]
    return tm

def fetch_ohlc(kite, token, from_date, to_date, interval):
    all_data, chunk = [], 60
    cur = from_date
    while cur < to_date:
        end = min(cur + timedelta(days=chunk), to_date)
        try:
            d = kite.historical_data(token, cur.strftime("%Y-%m-%d"),
                                     end.strftime("%Y-%m-%d"), interval)
            if d: all_data.extend(d)
        except Exception:
            pass
        cur = end + timedelta(days=1)
    if not all_data:
        return None
    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df[~df.index.duplicated(keep="first")]

def fetch_stock(kite, token, symbol, data_from, now):
    """Fetch daily + hourly for one stock. Returns (symbol, daily, hourly) or None."""
    try:
        daily  = fetch_ohlc(kite, token, data_from, now, "day")
        hourly = fetch_ohlc(kite, token, data_from, now, "60minute")
        if daily is not None and hourly is not None:
            if len(daily) >= EMA_PERIOD+5 and len(hourly) >= EMA_PERIOD+20:
                return symbol, daily, hourly
    except Exception:
        pass
    return None

# ════════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ════════════════════════════════════════════════════════════════════════════════
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

def compute_ema(s, p): return s.ewm(span=p, adjust=False).mean()

def compute_rsi(close, p=14):
    d  = close.diff()
    g  = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l  = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - 100/(1 + g/l.replace(0, float("inf")))

def compute_adx(df, p=14):
    h,l,c  = df["high"],df["low"],df["close"]
    pdm    = h.diff().clip(lower=0)
    ndm    = (-l.diff()).clip(lower=0)
    pdm    = pdm.where(pdm>ndm, 0)
    ndm    = ndm.where(ndm>pdm, 0)
    tr     = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr    = tr.ewm(com=p-1,adjust=False).mean()
    pdi    = 100*pdm.ewm(com=p-1,adjust=False).mean()/atr
    ndi    = 100*ndm.ewm(com=p-1,adjust=False).mean()/atr
    dx     = 100*(pdi-ndi).abs()/(pdi+ndi).replace(0,float("inf"))
    return dx.ewm(com=p-1,adjust=False).mean()

# ════════════════════════════════════════════════════════════════════════════════
#  HA CANDLE CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════════
def ha_bull_type(ha_row):
    o,h,l,c = ha_row["ha_open"],ha_row["ha_high"],ha_row["ha_low"],ha_row["ha_close"]
    rng = h - l
    if rng == 0: return None
    tol = o * TOLERANCE
    # STRONG BUY — no lower wick
    if abs(o-l) <= tol and c > o:
        return "STRONG_BUY"
    # BUY — big body, small wicks
    if c > o:
        body = c - o
        if (body/rng >= BODY_RATIO and
            (o-l)/rng <= WICK_RATIO and
            (h-c)/rng <= WICK_RATIO):
            return "BUY"
    return None

def ha_bear_type(ha_row):
    o,h,l,c = ha_row["ha_open"],ha_row["ha_high"],ha_row["ha_low"],ha_row["ha_close"]
    rng = h - l
    if rng == 0: return None
    tol = o * TOLERANCE
    # STRONG SELL — no upper wick
    if abs(o-h) <= tol and c < o:
        return "STRONG_SELL"
    # SELL — big body, small wicks
    if c < o:
        body = o - c
        if (body/rng >= BODY_RATIO and
            (h-o)/rng <= WICK_RATIO and
            (c-l)/rng <= WICK_RATIO):
            return "SELL"
    return None

# ════════════════════════════════════════════════════════════════════════════════
#  WEEKLY / DAILY HA
# ════════════════════════════════════════════════════════════════════════════════
def check_weekly_ha(ha):
    r = ha.iloc[-1]
    o,h,l,c = r["ha_open"],r["ha_high"],r["ha_low"],r["ha_close"]
    tol = o*TOLERANCE
    if abs(o-l)<=tol and c>o: return "BULLISH"
    if abs(o-h)<=tol and c<o: return "BEARISH"
    return None

def check_daily_ha(ha):
    r = ha.iloc[-1]
    o,h,l,c = r["ha_open"],r["ha_high"],r["ha_low"],r["ha_close"]
    tol = o*TOLERANCE
    if abs(o-l)<=tol and c>o: return "BULLISH"
    if abs(o-h)<=tol and c<o: return "BEARISH"
    has_lower = l < min(o,c)-tol
    has_upper = h > max(o,c)+tol
    if has_lower and has_upper: return "NEUTRAL"
    return None

# ════════════════════════════════════════════════════════════════════════════════
#  RIPPLE CHECK
# ════════════════════════════════════════════════════════════════════════════════
def check_ripple(idx, raw_h, ha_h, ema_h, rsi_h, adx_h, direction):
    ha_r  = ha_h.iloc[idx]
    raw_r = raw_h.iloc[idx]
    ema_v = ema_h.iloc[idx]
    rsi_n = rsi_h.iloc[idx]
    rsi_p = rsi_h.iloc[idx-1]
    adx_n = adx_h.iloc[idx]
    adx_p = adx_h.iloc[idx-1]

    # ① HA Candle
    if direction == "BULL":
        ctype = ha_bull_type(ha_r)
    else:
        ctype = ha_bear_type(ha_r)
    if not ctype: return None

    # ② EMA Zone
    close = raw_r["close"]
    high  = raw_r["high"]
    low   = raw_r["low"]
    if direction == "BULL":
        if low < ema_v and close > ema_v:
            ema_zone = "FAKE_BREAKDOWN"
        elif close > ema_v:
            ema_zone = "PRICE>EMA50"
        else:
            return None
    else:
        if high > ema_v and close < ema_v:
            ema_zone = "FAKE_BREAKOUT"
        elif close < ema_v:
            ema_zone = "PRICE<EMA50"
        else:
            return None

    # ③ RSI
    if direction == "BULL":
        if 40 <= rsi_n <= 60:
            rsi_cond = f"RANGE({rsi_n:.1f})"
        elif rsi_p < 60 and rsi_n >= 60:
            rsi_cond = f"CROSS↑60({rsi_n:.1f})"
        else:
            return None
    else:
        if 40 <= rsi_n <= 60:
            rsi_cond = f"RANGE({rsi_n:.1f})"
        elif rsi_p > 40 and rsi_n <= 40:
            rsi_cond = f"CROSS↓40({rsi_n:.1f})"
        else:
            return None

    # ④ ADX mandatory
    if not (adx_n >= ADX_MIN and adx_n > adx_p):
        return None

    return {
        "candle_type": ctype,
        "ema_zone"   : ema_zone,
        "rsi_cond"   : rsi_cond,
        "adx"        : round(adx_n, 1),
        "rsi"        : round(rsi_n, 1),
        "close"      : round(close, 2),
        "ema50"      : round(ema_v, 2),
    }

# ════════════════════════════════════════════════════════════════════════════════
#  PROCESS ONE STOCK — History Mode
# ════════════════════════════════════════════════════════════════════════════════
def process_stock_history(symbol, raw_daily, raw_hourly, bt_start_ts):
    signals = []
    try:
        ha_h  = compute_ha(raw_hourly)
        ema_h = compute_ema(raw_hourly["close"], EMA_PERIOD)
        rsi_h = compute_rsi(raw_hourly["close"], RSI_PERIOD)
        adx_h = compute_adx(raw_hourly, ADX_PERIOD)
        min_i = EMA_PERIOD + 15

        # ── SPEED OPTIMIZATION: cache daily results per date ──────────────────
        date_cache = {}   # date → (wsig, dsig, en, ep, cd) or None

        prev_date = None
        for i in range(min_i, len(raw_hourly)-1):
            bar_time = raw_hourly.index[i]
            if bar_time < bt_start_ts: continue

            bar_date = bar_time.date()

            # Only recompute daily/weekly when date changes
            if bar_date not in date_cache:
                d_slice = raw_daily[raw_daily.index.date <= bar_date]
                if len(d_slice) < EMA_PERIOD+2:
                    date_cache[bar_date] = None
                else:
                    w_from = bar_time - timedelta(weeks=12)
                    wkly   = d_slice[d_slice.index >= str(w_from.date())].resample("W-FRI").agg(
                        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
                    ).dropna()
                    if len(wkly) < 2:
                        date_cache[bar_date] = None
                    else:
                        wsig  = check_weekly_ha(compute_ha(wkly))
                        dsig  = check_daily_ha(compute_ha(d_slice))
                        ema_d = compute_ema(d_slice["close"], EMA_PERIOD)
                        en    = float(ema_d.iloc[-1])
                        ep    = float(ema_d.iloc[-2])
                        cd    = float(d_slice["close"].iloc[-1])
                        date_cache[bar_date] = (wsig, dsig, en, ep, cd)

            cached = date_cache.get(bar_date)
            if not cached: continue
            wsig, dsig, en, ep, cd = cached

            if not wsig: continue
            bull_ok = wsig=="BULLISH" and dsig in ("BULLISH","NEUTRAL") and cd>en and en>ep
            bear_ok = wsig=="BEARISH" and dsig in ("BEARISH","NEUTRAL") and cd<en and en<ep
            if not bull_ok and not bear_ok: continue

            direction = "BULL" if bull_ok else "BEAR"
            r = check_ripple(i, raw_hourly, ha_h, ema_h, rsi_h, adx_h, direction)
            if r:
                meta = STOCKS.get(symbol, {})
                r.update({
                    "symbol"      : symbol,
                    "direction"   : "BULLISH" if direction=="BULL" else "BEARISH",
                    "signal_time" : bar_time.strftime("%Y-%m-%d %H:%M"),
                    "weekly_sig"  : wsig,
                    "daily_sig"   : dsig or "NEUTRAL",
                    "daily_close" : round(cd, 2),
                    "daily_ema50" : round(en, 2),
                    "ema_slope"   : "RISING" if en>ep else "FALLING",
                    "sector"      : meta.get("sector","Unknown"),
                    "segment"     : meta.get("segment","Cash"),
                })
                signals.append(sanitize(r))
    except Exception:
        pass
    return signals

# ════════════════════════════════════════════════════════════════════════════════
#  PROCESS ONE STOCK — Live Mode
# ════════════════════════════════════════════════════════════════════════════════
def process_stock_live(symbol, raw_daily, raw_hourly):
    try:
        ha_h  = compute_ha(raw_hourly)
        ema_h = compute_ema(raw_hourly["close"], EMA_PERIOD)
        rsi_h = compute_rsi(raw_hourly["close"], RSI_PERIOD)
        adx_h = compute_adx(raw_hourly, ADX_PERIOD)

        wsig_now = check_weekly_ha
        dsig_now = check_daily_ha(compute_ha(raw_daily))
        ema_d    = compute_ema(raw_daily["close"], EMA_PERIOD)
        en, ep   = ema_d.iloc[-1], ema_d.iloc[-2]
        cd       = raw_daily["close"].iloc[-1]

        # Weekly from last 12 weeks
        wkly = raw_daily.resample("W-FRI").agg(
            {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        ).dropna()
        wsig = check_weekly_ha(compute_ha(wkly))
        if not wsig: return None

        bull_ok = wsig=="BULLISH" and dsig_now in ("BULLISH","NEUTRAL") and cd>en and en>ep
        bear_ok = wsig=="BEARISH" and dsig_now in ("BEARISH","NEUTRAL") and cd<en and en<ep
        if not bull_ok and not bear_ok: return None

        direction = "BULL" if bull_ok else "BEAR"
        r = check_ripple(-2, raw_hourly, ha_h, ema_h, rsi_h, adx_h, direction)
        if r:
            meta = STOCKS.get(symbol, {})
            r.update({
                "symbol"      : symbol,
                "direction"   : "BULLISH" if direction=="BULL" else "BEARISH",
                "signal_time" : raw_hourly.index[-2].strftime("%Y-%m-%d %H:%M"),
                "weekly_sig"  : wsig,
                "daily_sig"   : dsig_now or "NEUTRAL",
                "daily_close" : round(float(cd), 2),
                "daily_ema50" : round(float(en), 2),
                "ema_slope"   : "RISING" if en>ep else "FALLING",
                "sector"      : meta.get("sector","Unknown"),
                "segment"     : meta.get("segment","Cash"),
            })
            return sanitize(r)
    except Exception:
        pass
    return None

# ════════════════════════════════════════════════════════════════════════════════
#  EMAIL
# ════════════════════════════════════════════════════════════════════════════════
def send_email(live_signals):
    if not live_signals: return
    bull = [s for s in live_signals if s["direction"]=="BULLISH"]
    bear = [s for s in live_signals if s["direction"]=="BEARISH"]

    def rows(sigs, arrow, col):
        out = ""
        for s in sigs:
            seg_badge = f'<span style="background:#1e293b;color:#64748b;padding:1px 5px;border-radius:3px;font-size:10px">{s["segment"]}</span>'
            out += f"""<tr>
              <td style="padding:9px 12px;border-bottom:1px solid #1a2234;color:#94a3b8;font-family:monospace;font-size:11px;white-space:nowrap">{s['signal_time']}</td>
              <td style="padding:9px 12px;border-bottom:1px solid #1a2234;font-weight:800;color:{col};font-size:15px">{arrow} {s['symbol']} {seg_badge}</td>
              <td style="padding:9px 12px;border-bottom:1px solid #1a2234"><span style="background:{'rgba(74,222,128,0.2)' if 'STRONG' in s['candle_type'] else 'rgba(56,189,248,0.15)'};color:{'#4ade80' if 'STRONG' in s['candle_type'] else '#38bdf8'};padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700">{'💪 ' if 'STRONG' in s['candle_type'] else ''}{s['candle_type']}</span></td>
              <td style="padding:9px 12px;border-bottom:1px solid #1a2234;color:#94a3b8;font-size:12px">{s['ema_zone']}</td>
              <td style="padding:9px 12px;border-bottom:1px solid #1a2234;color:#e2e8f0;font-family:monospace;font-weight:600">₹{s['close']}</td>
              <td style="padding:9px 12px;border-bottom:1px solid #1a2234;color:#64748b;font-size:11px">RSI:{s['rsi']} ADX:{s['adx']}</td>
              <td style="padding:9px 12px;border-bottom:1px solid #1a2234;color:#475569;font-size:11px">{s.get('sector','')}</td>
            </tr>"""
        return out

    thead = """<tr style="background:#070d1a">
      <th style="padding:8px 12px;color:#334155;font-size:10px;text-align:left;font-family:monospace">TIME</th>
      <th style="padding:8px 12px;color:#334155;font-size:10px;text-align:left">SYMBOL</th>
      <th style="padding:8px 12px;color:#334155;font-size:10px;text-align:left">CANDLE</th>
      <th style="padding:8px 12px;color:#334155;font-size:10px;text-align:left">EMA ZONE</th>
      <th style="padding:8px 12px;color:#334155;font-size:10px;text-align:left">PRICE</th>
      <th style="padding:8px 12px;color:#334155;font-size:10px;text-align:left">INDICATORS</th>
      <th style="padding:8px 12px;color:#334155;font-size:10px;text-align:left">SECTOR</th>
    </tr>"""

    html = f"""<html><body style="background:#070c14;font-family:'Segoe UI',sans-serif;padding:24px">
    <div style="max-width:800px;margin:0 auto">
      <div style="background:linear-gradient(135deg,#0d1929,#111827);border:1px solid #1e293b;border-radius:14px;padding:22px;margin-bottom:20px">
        <h1 style="margin:0;font-size:24px;background:linear-gradient(90deg,#4ade80,#22d3ee,#f87171);-webkit-background-clip:text;-webkit-text-fill-color:transparent">⚡ HA Triple Screen Alert</h1>
        <p style="color:#475569;margin:6px 0 0;font-size:13px">
          {datetime.now(IST).strftime('%d %b %Y, %H:%M IST')} &nbsp;|&nbsp;
          <span style="color:#4ade80">{len(bull)} Bullish</span> &nbsp;|&nbsp;
          <span style="color:#f87171">{len(bear)} Bearish</span>
        </p>
      </div>
      {'<div style="background:#0d1929;border:1px solid rgba(74,222,128,0.2);border-radius:12px;overflow:hidden;margin-bottom:16px"><div style="padding:12px 16px;border-bottom:1px solid #1e293b;color:#4ade80;font-weight:700;font-size:14px">🟢 BULLISH SIGNALS</div><table width="100%" cellspacing="0">'+thead+rows(bull,"▲","#4ade80")+'</table></div>' if bull else ''}
      {'<div style="background:#0d1929;border:1px solid rgba(248,113,113,0.2);border-radius:12px;overflow:hidden"><div style="padding:12px 16px;border-bottom:1px solid #1e293b;color:#f87171;font-weight:700;font-size:14px">🔴 BEARISH SIGNALS</div><table width="100%" cellspacing="0">'+thead+rows(bear,"▼","#f87171")+'</table></div>' if bear else ''}
      <p style="color:#1e293b;font-size:10px;text-align:center;margin-top:16px">HA Triple Screen Scanner • Not financial advice</p>
    </div></body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"⚡ HA Scanner: {len(bull)}🟢 {len(bear)}🔴 — {datetime.now(IST).strftime('%d %b %H:%M')}"
    msg["From"]    = EMAIL_FROM
    msg["To"]      = ", ".join(EMAIL_TO)
    msg.attach(MIMEText(html,"html"))
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=ctx) as s:
            s.login(EMAIL_FROM, EMAIL_PASSWORD)
            s.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        print(f"  ✅ Email sent → {EMAIL_TO}")
    except Exception as e:
        print(f"  ❌ Email error: {e}")

# ════════════════════════════════════════════════════════════════════════════════
#  PERSISTENT HISTORY
# ════════════════════════════════════════════════════════════════════════════════
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            try: return json.load(f)
            except: return []
    return []

def save_history(history):
    # Keep only last 60 days
    cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    history = [s for s in history if s.get("signal_time","") >= cutoff]
    with open(HISTORY_FILE,"w") as f:
        json.dump(history, f, indent=2, default=str)

def merge_history(existing, new_signals):
    keys = {(s["symbol"],s["signal_time"],s["direction"]) for s in existing}
    added = []
    for s in new_signals:
        k = (s["symbol"],s["signal_time"],s["direction"])
        if k not in keys:
            existing.append(s)
            added.append(s)
            keys.add(k)
    return existing, added

# ════════════════════════════════════════════════════════════════════════════════
#  HTML DASHBOARD GENERATOR  (fully self-contained, all JS inline)
# ════════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════════
#  DASHBOARD BUILDER  — All data embedded as JS, zero server-side rendering bugs
# ════════════════════════════════════════════════════════════════════════════════
def build_dashboard(live_sigs, hist_sigs, run_time, scan_dur):
    from collections import Counter

    bull_live = [s for s in live_sigs if s["direction"] == "BULLISH"]
    bear_live = [s for s in live_sigs if s["direction"] == "BEARISH"]
    bull_hist = [s for s in hist_sigs if s["direction"] == "BULLISH"]
    bear_hist = [s for s in hist_sigs if s["direction"] == "BEARISH"]

    total   = len(STOCKS)
    strong  = sum(1 for s in hist_sigs if "STRONG" in s.get("candle_type",""))

    # ── Pre-compute chart data in Python → embed as JS literals ───────────────
    all_dates   = sorted({s["signal_time"][:10] for s in hist_sigs})
    bday = Counter(s["signal_time"][:10] for s in bull_hist)
    rday = Counter(s["signal_time"][:10] for s in bear_hist)

    candle_cnt = Counter(s["candle_type"] for s in hist_sigs)
    ema_cnt    = Counter(s["ema_zone"]    for s in hist_sigs)
    sec_cnt    = Counter(s.get("sector","?") for s in bull_hist)

    hbull = {str(h): sum(1 for s in bull_hist if s["signal_time"][11:13] == f"{h:02d}") for h in range(9,16)}
    hbear = {str(h): sum(1 for s in bear_hist if s["signal_time"][11:13] == f"{h:02d}") for h in range(9,16)}

    # Serialize everything ONCE here — no json.dumps inside f-string
    J = lambda x: json.dumps(x, cls=NpEncoder)

    j_live  = J(live_sigs)
    j_hist  = J(hist_sigs)
    j_dates = J(all_dates)
    j_bull  = J([int(bday.get(d, 0)) for d in all_dates])
    j_bear  = J([int(rday.get(d, 0)) for d in all_dates])
    j_candle= J({k: int(v) for k, v in candle_cnt.items()})
    j_ema   = J({k: int(v) for k, v in ema_cnt.items()})
    j_secl  = J(list(sec_cnt.keys()))
    j_secv  = J([int(v) for v in sec_cnt.values()])
    j_secc  = J([SECTOR_COLORS.get(k, "#64748b") for k in sec_cnt.keys()])
    j_hbull = J(hbull)
    j_hbear = J(hbear)
    j_scmap = J(SECTOR_COLORS)

    fo_count   = len([s for s in STOCKS if STOCKS[s].get("segment") == "F&O"])
    cash_count = len([s for s in STOCKS if STOCKS[s].get("segment") == "Cash"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HA Scanner Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;-webkit-font-smoothing:antialiased}}
:root{{
  --bg0:#03070f;--bg1:#070e1c;--bg2:#0b1628;--bg3:#0f1e36;
  --bdr:#16253a;--bdr2:#1e334e;
  --g:#4ade80;--r:#f87171;--b:#38bdf8;--y:#fbbf24;
  --pu:#a78bfa;--or:#fb923c;--tl:#2dd4bf;
  --t1:#ecf2f8;--t2:#7a9ab8;--t3:#3d5870;--t4:#1e3048;
}}
html,body{{height:100%;overflow-x:hidden}}
body{{
  background:var(--bg0);color:var(--t1);
  font-family:'Outfit',sans-serif;
  background-image:
    radial-gradient(ellipse at 8% 8%,rgba(74,222,128,.05) 0%,transparent 50%),
    radial-gradient(ellipse at 92% 92%,rgba(248,113,113,.05) 0%,transparent 50%);
}}

/* ── LAYOUT ── */
.w{{max-width:1600px;margin:0 auto;padding:16px 20px}}

/* ── HEADER ── */
.hdr{{
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;
  background:var(--bg1);border:1px solid var(--bdr);border-radius:14px;
  padding:14px 24px;margin-bottom:14px;
  box-shadow:0 4px 24px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.03)
}}
.hdr h1{{
  font-family:'Bebas Neue',sans-serif;font-size:30px;letter-spacing:2px;
  background:linear-gradient(90deg,var(--g),var(--b),var(--r));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent
}}
.hdr p{{font-size:10px;color:var(--t3);font-family:'DM Mono',monospace;margin-top:2px}}
.htags{{display:flex;gap:6px;flex-wrap:wrap}}
.ht{{
  padding:3px 11px;border-radius:20px;font-size:11px;font-weight:600;
  font-family:'DM Mono',monospace;border:1px solid
}}
.ht-g{{background:rgba(74,222,128,.08);border-color:rgba(74,222,128,.2);color:var(--g)}}
.ht-b{{background:rgba(56,189,248,.08);border-color:rgba(56,189,248,.2);color:var(--b)}}
.ht-p{{background:rgba(167,139,250,.08);border-color:rgba(167,139,250,.2);color:var(--pu)}}
.ht-y{{background:rgba(251,191,36,.08);border-color:rgba(251,191,36,.2);color:var(--y)}}
.pill{{
  display:inline-flex;align-items:center;gap:6px;
  background:rgba(74,222,128,.08);border:1px solid rgba(74,222,128,.2);
  color:var(--g);padding:5px 13px;border-radius:20px;font-size:11px;font-weight:700
}}
.dot{{width:6px;height:6px;background:var(--g);border-radius:50%;animation:blink 1.4s infinite}}
@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:.2}}}}
.hmeta{{font-size:10px;color:var(--t4);font-family:'DM Mono',monospace;margin-top:4px}}

/* ── STATS ── */
.stats{{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;margin-bottom:14px}}
.sc{{
  background:var(--bg1);border:1px solid var(--bdr);border-radius:12px;
  padding:13px 10px;text-align:center;position:relative;overflow:hidden;
  transition:border-color .2s,transform .15s;cursor:default
}}
.sc:hover{{border-color:var(--bdr2);transform:translateY(-1px)}}
.sc-top{{position:absolute;top:0;left:0;right:0;height:2px;border-radius:2px}}
.sc-n{{font-family:'Bebas Neue',sans-serif;font-size:36px;line-height:1;letter-spacing:1px}}
.sc-l{{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.8px;margin-top:3px}}
.sc-s{{font-size:9px;color:var(--t4);margin-top:2px;font-family:'DM Mono',monospace}}

/* ── TABS ── */
.tabs{{
  display:flex;gap:3px;background:var(--bg1);border:1px solid var(--bdr);
  border-radius:10px;padding:4px;width:fit-content;margin-bottom:14px
}}
.tab{{
  padding:7px 20px;border-radius:7px;font-size:13px;font-weight:600;
  cursor:pointer;color:var(--t3);border:none;background:transparent;
  font-family:'Outfit',sans-serif;transition:all .18s;user-select:none
}}
.tab:hover{{color:var(--t2)}}
.tab.on{{background:var(--bg3);color:var(--t1);box-shadow:0 2px 8px rgba(0,0,0,.3)}}

/* ── SECTIONS ── */
.sec{{display:none}}.sec.on{{display:block}}

/* ── FILTER BAR ── */
.fbar{{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap;align-items:center}}
.fb{{
  padding:5px 13px;border-radius:7px;border:1px solid var(--bdr);
  background:transparent;color:var(--t3);font-size:12px;cursor:pointer;
  font-family:'Outfit',sans-serif;transition:all .18s
}}
.fb:hover{{border-color:var(--bdr2);color:var(--t2)}}
.fb.on{{background:var(--bg3);color:var(--t1);border-color:var(--bdr2)}}
.srch{{
  padding:6px 13px;border-radius:7px;border:1px solid var(--bdr);
  background:var(--bg1);color:var(--t1);font-size:12px;
  font-family:'DM Mono',monospace;width:200px;outline:none;transition:border-color .18s
}}
.srch:focus{{border-color:var(--bdr2)}}
.ml{{margin-left:auto}}
.dlbtn{{
  padding:6px 14px;border-radius:7px;background:var(--bg3);
  border:1px solid var(--bdr2);color:var(--t2);font-size:12px;
  cursor:pointer;font-family:'Outfit',sans-serif;transition:all .18s
}}
.dlbtn:hover{{color:var(--t1)}}

/* ── LIVE CARDS ── */
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:13px}}
.card{{
  border-radius:13px;padding:16px 16px 13px 20px;
  position:relative;transition:transform .18s,box-shadow .18s
}}
.card:hover{{transform:translateY(-2px);box-shadow:0 12px 36px rgba(0,0,0,.5)}}
.card-bar{{position:absolute;left:0;top:0;bottom:0;width:4px;border-radius:4px 0 0 4px}}
.card-top{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:11px}}
.csym{{font-family:'Bebas Neue',sans-serif;font-size:24px;letter-spacing:1px}}
.cprice{{font-family:'Bebas Neue',sans-serif;font-size:24px;letter-spacing:1px}}
.cbadge{{
  padding:2px 7px;border-radius:5px;border:1px solid;
  font-size:10px;font-family:'DM Mono',monospace
}}
.cstrong{{
  background:rgba(251,191,36,.12);color:var(--y);
  border:1px solid rgba(251,191,36,.25);
  padding:2px 7px;border-radius:5px;font-size:10px;font-weight:700
}}
/* Condition panel */
.cpanel{{
  background:rgba(0,0,0,.3);border:1px solid var(--bdr);
  border-radius:10px;padding:11px;margin-bottom:9px
}}
.cpanel-hd{{
  font-size:9px;text-transform:uppercase;letter-spacing:1px;
  color:var(--t4);font-family:'DM Mono',monospace;margin-bottom:7px
}}
.crow{{
  display:flex;align-items:center;gap:7px;padding:5px 8px;
  background:rgba(255,255,255,.02);border-radius:6px;margin-bottom:3px;flex-wrap:wrap
}}
.ckey{{font-size:10px;color:var(--t3);min-width:86px;font-family:'DM Mono',monospace}}
.cval{{font-size:12px;font-weight:700}}
.rip{{
  padding:8px 10px;background:rgba(56,189,248,.04);
  border:1px solid rgba(56,189,248,.1);border-radius:8px;
  display:grid;grid-template-columns:1fr 1fr;gap:5px
}}
.ri{{display:flex;align-items:center;gap:5px;font-size:11px}}
.rk{{color:var(--t4);font-size:10px}}
.cfoot{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px}}
.ftag{{
  padding:2px 7px;border:1px solid var(--bdr);border-radius:5px;
  font-size:10px;color:var(--t3);font-family:'DM Mono',monospace
}}
.ftime{{font-size:10px;color:var(--t4);font-family:'DM Mono',monospace}}
.empty{{text-align:center;padding:60px;color:var(--t4)}}
.empty-i{{font-size:48px;margin-bottom:14px}}
.empty-t{{font-size:14px;font-weight:600;color:var(--t3)}}
.empty-s{{font-size:11px;margin-top:6px;font-family:'DM Mono',monospace}}

/* ── HISTORY TABLE ── */
.twrap{{background:var(--bg1);border:1px solid var(--bdr);border-radius:14px;overflow:hidden}}
.tbar{{
  display:flex;align-items:center;gap:6px;padding:11px 15px;
  border-bottom:1px solid var(--bdr);flex-wrap:wrap
}}
.tscroll{{overflow-x:auto;max-height:580px;overflow-y:auto}}
table{{width:100%;border-collapse:collapse}}
thead{{position:sticky;top:0;z-index:10}}
th{{
  padding:8px 11px;text-align:left;font-size:10px;text-transform:uppercase;
  letter-spacing:.6px;color:var(--t4);background:var(--bg0);
  border-bottom:1px solid var(--bdr);font-family:'DM Mono',monospace;
  white-space:nowrap
}}
td{{padding:8px 11px;border-bottom:1px solid rgba(22,37,58,.6);font-size:12px}}
tbody tr:hover td{{background:rgba(11,22,40,.8)}}
.hbadge{{
  padding:2px 7px;border-radius:5px;font-size:10px;
  font-weight:700;font-family:'DM Mono',monospace
}}

/* ── CHARTS ── */
.cgrid2{{display:grid;grid-template-columns:3fr 2fr;gap:13px;margin-bottom:13px}}
.cgrid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:13px}}
.cbox{{background:var(--bg1);border:1px solid var(--bdr);border-radius:13px;padding:18px}}
.ctitle{{font-size:13px;font-weight:600;color:var(--t2);margin-bottom:14px}}

/* ── CONDITIONS ── */
.cond-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:13px}}
.cc{{background:var(--bg1);border-radius:13px;padding:20px}}
.cc-bull{{border:1px solid rgba(74,222,128,.15)}}
.cc-bear{{border:1px solid rgba(248,113,113,.15)}}
.cc-head{{display:flex;align-items:center;gap:12px;margin-bottom:16px}}
.cc-icon{{width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px}}
.cc-title{{font-family:'Bebas Neue',sans-serif;font-size:20px;letter-spacing:1px}}
.tier{{margin-bottom:13px}}
.tier-n{{
  font-size:9px;text-transform:uppercase;letter-spacing:1px;
  color:var(--t4);font-family:'DM Mono',monospace;
  margin-bottom:5px;padding-bottom:4px;border-bottom:1px solid var(--bdr)
}}
.tr{{
  display:flex;gap:9px;padding:7px 10px;
  background:rgba(0,0,0,.2);border-radius:7px;margin-bottom:3px
}}
.tn{{color:var(--b);font-weight:700;font-size:12px;font-family:'DM Mono',monospace;min-width:17px}}
.tc{{font-size:12px;color:var(--t2);line-height:1.55}}
.tc b{{color:var(--t1)}}
.tor{{font-size:10px;color:var(--t4);text-align:center;padding:1px 0;font-family:'DM Mono',monospace}}
.mand{{
  display:inline-block;background:rgba(251,191,36,.1);color:var(--y);
  border:1px solid rgba(251,191,36,.25);padding:1px 5px;border-radius:3px;
  font-size:9px;font-weight:700;margin-left:5px
}}
.pbox{{background:var(--bg1);border:1px solid var(--bdr);border-radius:13px;padding:18px}}
.pgrid{{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:11px}}
.pi{{background:rgba(0,0,0,.25);border:1px solid var(--bdr);border-radius:8px;padding:10px}}
.pi-k{{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:.5px;font-family:'DM Mono',monospace}}
.pi-v{{font-size:17px;font-weight:700;color:var(--b);font-family:'Bebas Neue',sans-serif;letter-spacing:1px;margin-top:3px}}

@media(max-width:1100px){{.stats{{grid-template-columns:repeat(4,1fr)}}}}
@media(max-width:800px){{
  .cgrid2,.cgrid3,.cond-grid{{grid-template-columns:1fr}}
  .cards{{grid-template-columns:1fr}}
}}
</style>
</head>
<body>
<div class="w">

<!-- HEADER -->
<div class="hdr">
  <div>
    <h1>⚡ HA SCANNER</h1>
    <p>TRIPLE SCREEN · TIDE(W) + WAVE(D) + RIPPLE(H) · EMA50 · RSI14 · ADX14</p>
  </div>
  <div class="htags">
    <span class="ht ht-g">F&amp;O: {fo_count}</span>
    <span class="ht ht-b">Cash: {cash_count}</span>
    <span class="ht ht-p">NSE500+</span>
    <span class="ht ht-y">Total: {total}</span>
  </div>
  <div style="text-align:right">
    <div class="pill"><span class="dot"></span>LIVE</div>
    <div class="hmeta">{run_time} · ⏱ {scan_dur}s</div>
  </div>
</div>

<!-- STATS -->
<div class="stats">
  <div class="sc"><div class="sc-top" style="background:linear-gradient(90deg,#4ade80,#2dd4bf)"></div>
    <div class="sc-n" style="color:var(--g)">{len(bull_live)}</div><div class="sc-l">Live Bullish</div><div class="sc-s">Active now</div></div>
  <div class="sc"><div class="sc-top" style="background:linear-gradient(90deg,#f87171,#fb923c)"></div>
    <div class="sc-n" style="color:var(--r)">{len(bear_live)}</div><div class="sc-l">Live Bearish</div><div class="sc-s">Active now</div></div>
  <div class="sc"><div class="sc-top" style="background:linear-gradient(90deg,#38bdf8,#818cf8)"></div>
    <div class="sc-n" style="color:var(--b)">{len(bull_hist)}</div><div class="sc-l">Bull History</div><div class="sc-s">30 days</div></div>
  <div class="sc"><div class="sc-top" style="background:linear-gradient(90deg,#fbbf24,#fb923c)"></div>
    <div class="sc-n" style="color:var(--y)">{len(bear_hist)}</div><div class="sc-l">Bear History</div><div class="sc-s">30 days</div></div>
  <div class="sc"><div class="sc-top" style="background:linear-gradient(90deg,#a78bfa,#38bdf8)"></div>
    <div class="sc-n" style="color:var(--pu)">{len(hist_sigs)}</div><div class="sc-l">Total Signals</div><div class="sc-s">This month</div></div>
  <div class="sc"><div class="sc-top" style="background:linear-gradient(90deg,#2dd4bf,#4ade80)"></div>
    <div class="sc-n" style="color:var(--tl)">{strong}</div><div class="sc-l">Strong</div><div class="sc-s">High conviction</div></div>
  <div class="sc"><div class="sc-top" style="background:linear-gradient(90deg,#3d5870,#1e3048)"></div>
    <div class="sc-n" style="color:var(--t2)">{total}</div><div class="sc-l">Scanned</div><div class="sc-s">F&amp;O + Cash</div></div>
</div>

<!-- TABS -->
<div class="tabs">
  <button class="tab on"  onclick="showTab('live',this)">⚡ Live Signals</button>
  <button class="tab"     onclick="showTab('hist',this)">📅 Signal History</button>
  <button class="tab"     onclick="showTab('chart',this)">📊 Charts</button>
  <button class="tab"     onclick="showTab('cond',this)">📋 Conditions</button>
</div>

<!-- LIVE -->
<div id="s-live" class="sec on">
  <div class="fbar">
    <button class="fb on" onclick="flLive('all',this)">All ({len(live_sigs)})</button>
    <button class="fb"    onclick="flLive('bull',this)">🟢 Bull ({len(bull_live)})</button>
    <button class="fb"    onclick="flLive('bear',this)">🔴 Bear ({len(bear_live)})</button>
    <button class="fb"    onclick="flLive('strong',this)">💪 Strong</button>
    <button class="fb"    onclick="flLive('fo',this)">F&amp;O</button>
    <button class="fb"    onclick="flLive('cash',this)">Cash</button>
  </div>
  <div id="live-wrap" class="cards"></div>
</div>

<!-- HISTORY -->
<div id="s-hist" class="sec">
  <div class="twrap">
    <div class="tbar">
      <button class="fb on" onclick="flHist('all',this)">All ({len(hist_sigs)})</button>
      <button class="fb"    onclick="flHist('bull',this)">🟢 Bull ({len(bull_hist)})</button>
      <button class="fb"    onclick="flHist('bear',this)">🔴 Bear ({len(bear_hist)})</button>
      <button class="fb"    onclick="flHist('strong',this)">💪 Strong</button>
      <button class="fb"    onclick="flHist('fo',this)">F&amp;O</button>
      <button class="fb"    onclick="flHist('cash',this)">Cash</button>
      <input  class="srch"  id="srch" placeholder="🔍 Symbol / Sector..." oninput="srchHist(this.value)">
      <button class="dlbtn ml" onclick="doCSV()">⬇ CSV</button>
    </div>
    <div class="tscroll">
      <table>
        <thead><tr>
          <th>DATE &amp; TIME</th><th>SYMBOL</th><th>DIR</th><th>CANDLE</th>
          <th>EMA ZONE</th><th>RSI COND</th><th>PRICE</th><th>RSI</th><th>ADX</th>
          <th>WEEKLY</th><th>DAILY</th><th>D-CLOSE</th><th>D-EMA50</th>
          <th>EMA SLOPE</th><th>SECTOR</th><th>SEG</th>
        </tr></thead>
        <tbody id="htbody"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- CHARTS -->
<div id="s-chart" class="sec">
  <div class="cgrid2">
    <div class="cbox"><div class="ctitle">📊 Signal Frequency — Last 30 Days</div><canvas id="cBar" height="110"></canvas></div>
    <div class="cbox"><div class="ctitle">🕐 Signals by Hour</div><canvas id="cHour" height="200"></canvas></div>
  </div>
  <div class="cgrid3">
    <div class="cbox"><div class="ctitle">🕯 Candle Type</div><canvas id="cCandle" height="240"></canvas></div>
    <div class="cbox"><div class="ctitle">📍 EMA Zone</div><canvas id="cEma" height="240"></canvas></div>
    <div class="cbox"><div class="ctitle">🏭 Top Sectors (Bull)</div><canvas id="cSec" height="240"></canvas></div>
  </div>
</div>

<!-- CONDITIONS -->
<div id="s-cond" class="sec">
  <div class="cond-grid">
    <div class="cc cc-bull">
      <div class="cc-head">
        <div class="cc-icon" style="background:rgba(74,222,128,.1)">🟢</div>
        <div><div class="cc-title" style="color:var(--g)">BULLISH SIGNAL</div>
          <div style="font-size:11px;color:var(--t3);margin-top:2px">All 4 Ripple conditions must pass</div></div>
      </div>
      <div class="tier"><div class="tier-n">⬛ TIDE — Weekly</div>
        <div class="tr"><span class="tc"><b>HA Green</b> — ha_open == ha_low (no lower wick) AND ha_close &gt; ha_open</span></div></div>
      <div class="tier"><div class="tier-n">⬜ WAVE — Daily</div>
        <div class="tr"><span class="tc"><b>HA Bullish/Neutral</b> AND <b>Close &gt; EMA50</b> AND <b>EMA50 Rising</b></span></div></div>
      <div class="tier"><div class="tier-n">🔵 RIPPLE — Hourly (all 4 required)</div>
        <div class="tr"><span class="tn">①</span><span class="tc"><b>Candle</b>: 💪 Strong Buy (no lower wick) OR 📊 Buy (body≥60%, wicks≤20%)</span></div>
        <div class="tor">── OR ──</div>
        <div class="tr"><span class="tn">②</span><span class="tc"><b>EMA Zone</b>: Close &gt; EMA50 OR Fake Breakdown (low&lt;EMA, close&gt;EMA)</span></div>
        <div class="tor">── OR ──</div>
        <div class="tr"><span class="tn">③</span><span class="tc"><b>RSI</b>: 40–60 range OR Crossover above 60</span></div>
        <div class="tr" style="margin-top:5px"><span class="tn">④</span><span class="tc"><b>ADX ≥ 15 AND Rising</b><span class="mand">MANDATORY</span></span></div>
      </div>
    </div>
    <div class="cc cc-bear">
      <div class="cc-head">
        <div class="cc-icon" style="background:rgba(248,113,113,.1)">🔴</div>
        <div><div class="cc-title" style="color:var(--r)">BEARISH SIGNAL</div>
          <div style="font-size:11px;color:var(--t3);margin-top:2px">All 4 Ripple conditions must pass</div></div>
      </div>
      <div class="tier"><div class="tier-n">⬛ TIDE — Weekly</div>
        <div class="tr"><span class="tc"><b>HA Red</b> — ha_open == ha_high (no upper wick) AND ha_close &lt; ha_open</span></div></div>
      <div class="tier"><div class="tier-n">⬜ WAVE — Daily</div>
        <div class="tr"><span class="tc"><b>HA Bearish/Neutral</b> AND <b>Close &lt; EMA50</b> AND <b>EMA50 Falling</b></span></div></div>
      <div class="tier"><div class="tier-n">🔵 RIPPLE — Hourly (all 4 required)</div>
        <div class="tr"><span class="tn">①</span><span class="tc"><b>Candle</b>: 💪 Strong Sell (no upper wick) OR 📊 Sell (body≥60%, wicks≤20%)</span></div>
        <div class="tor">── OR ──</div>
        <div class="tr"><span class="tn">②</span><span class="tc"><b>EMA Zone</b>: Close &lt; EMA50 OR Fake Breakout (high&gt;EMA, close&lt;EMA)</span></div>
        <div class="tor">── OR ──</div>
        <div class="tr"><span class="tn">③</span><span class="tc"><b>RSI</b>: 40–60 range OR Crossover below 40</span></div>
        <div class="tr" style="margin-top:5px"><span class="tn">④</span><span class="tc"><b>ADX ≥ 15 AND Rising</b><span class="mand">MANDATORY</span></span></div>
      </div>
    </div>
  </div>
  <div class="pbox">
    <div style="font-size:13px;font-weight:600;color:var(--t2)">⚙️ Scanner Parameters</div>
    <div class="pgrid">
      <div class="pi"><div class="pi-k">EMA Period</div><div class="pi-v">50</div></div>
      <div class="pi"><div class="pi-k">RSI Period</div><div class="pi-v">14</div></div>
      <div class="pi"><div class="pi-k">ADX Period</div><div class="pi-v">14</div></div>
      <div class="pi"><div class="pi-k">ADX Min</div><div class="pi-v">15</div></div>
      <div class="pi"><div class="pi-k">Body Ratio</div><div class="pi-v">≥60%</div></div>
      <div class="pi"><div class="pi-k">Wick Ratio</div><div class="pi-v">≤20%</div></div>
      <div class="pi"><div class="pi-k">Total Stocks</div><div class="pi-v">{total}</div></div>
      <div class="pi"><div class="pi-k">Threads</div><div class="pi-v">{MAX_WORKERS}</div></div>
    </div>
  </div>
</div>

</div><!-- /w -->

<!-- ═══ ALL DATA + LOGIC ═══ -->
<script>
// ── Embedded data (Python → JS) ───────────────────────────────────────────────
const LIVE  = {j_live};
const HIST  = {j_hist};
const SCMAP = {j_scmap};

// ── Tab switching ─────────────────────────────────────────────────────────────
function showTab(id, btn) {{
  document.querySelectorAll('.sec').forEach(s => s.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('on'));
  document.getElementById('s-' + id).classList.add('on');
  btn.classList.add('on');
  if (id === 'chart') setTimeout(drawCharts, 60);
}}

// ── Live card builder ─────────────────────────────────────────────────────────
function buildCard(s) {{
  const bull   = s.direction === 'BULLISH';
  const col    = bull ? '#4ade80' : '#f87171';
  const bg     = bull ? 'rgba(74,222,128,.07)' : 'rgba(248,113,113,.07)';
  const bd     = bull ? 'rgba(74,222,128,.25)' : 'rgba(248,113,113,.25)';
  const arrow  = bull ? '▲' : '▼';
  const strong = (s.candle_type || '').includes('STRONG');
  const segcol = s.segment === 'F&O' ? '#38bdf8' : '#94a3b8';
  const seccol = SCMAP[s.sector] || '#64748b';
  const slope  = s.ema_slope || (bull ? 'RISING' : 'FALLING');
  const slcol  = slope === 'RISING' ? '#4ade80' : '#f87171';
  const dcl    = s.daily_close  || '—';
  const dema   = s.daily_ema50  || '—';
  return `
  <div class="card" style="background:${{bg}};border:1px solid ${{bd}}"
       data-dir="${{bull?'bull':'bear'}}" data-type="${{s.candle_type}}" data-seg="${{s.segment}}">
    <div class="card-bar" style="background:linear-gradient(180deg,${{col}},${{col}}70)"></div>
    <div class="card-top">
      <div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap">
        <span class="csym" style="color:${{col}}">${{arrow}} ${{s.symbol}}</span>
        <span class="cbadge" style="color:${{segcol}};border-color:${{segcol}}40">${{s.segment}}</span>
        ${{strong
          ? '<span class="cstrong">💪 STRONG</span>'
          : '<span style="background:rgba(56,189,248,.08);color:#38bdf8;border:1px solid rgba(56,189,248,.2);padding:2px 7px;border-radius:5px;font-size:10px">NORMAL</span>'
        }}
        <span style="color:${{seccol}};font-size:10px;background:rgba(0,0,0,.3);padding:2px 8px;border-radius:5px;border:1px solid #16253a">● ${{s.sector||''}}</span>
      </div>
      <div style="text-align:right">
        <div class="cprice" style="color:${{col}}">₹${{s.close}}</div>
        <div style="font-size:10px;color:#3d5870;font-family:'DM Mono',monospace;margin-top:2px">EMA50: ₹${{s.ema50}}</div>
      </div>
    </div>

    <div class="cpanel">
      <div class="cpanel-hd">CONDITION VERIFICATION</div>
      <div class="crow">
        <span>✅</span><span class="ckey">TIDE (Weekly)</span>
        <span class="cval" style="color:${{col}}">${{s.weekly_sig}}</span>
        <span style="font-size:10px;color:#1e3048">— HA ${{bull?'Green · No Lower Wick':'Red · No Upper Wick'}}</span>
      </div>
      <div class="crow" style="flex-direction:column;align-items:flex-start;gap:4px">
        <div style="display:flex;align-items:center;gap:7px">
          <span>✅</span><span class="ckey">WAVE (Daily)</span>
          <span class="cval" style="color:#e2e8f0">${{s.daily_sig}}</span>
        </div>
        <div style="display:flex;gap:12px;padding-left:26px;flex-wrap:wrap">
          <span style="font-size:10px;color:#2dd4bf">✓ ${{bull?'Close > EMA50':'Close < EMA50'}} (₹${{dcl}} / ₹${{dema}})</span>
          <span style="font-size:10px;color:${{slcol}}">✓ EMA ${{slope}}</span>
          <span style="font-size:10px;color:#2dd4bf">✓ Daily HA: ${{s.daily_sig}}</span>
        </div>
      </div>
      <div class="rip">
        <div style="grid-column:1/-1;font-size:10px;color:#38bdf8;font-weight:600;margin-bottom:4px;font-family:'DM Mono',monospace">🔵 RIPPLE (Hourly) — All 4 ✅</div>
        <div class="ri"><span>✅</span><span class="rk">① Candle:</span><b style="color:${{strong?'#4ade80':'#38bdf8'}}">${{s.candle_type}}</b></div>
        <div class="ri"><span>✅</span><span class="rk">② EMA:</span><b style="color:#a78bfa">${{s.ema_zone}}</b></div>
        <div class="ri"><span>✅</span><span class="rk">③ RSI:</span><b style="color:#fbbf24">${{s.rsi_cond}}</b></div>
        <div class="ri"><span>✅</span><span class="rk">④ ADX:</span><b style="color:#fb923c">${{s.adx}} ↑</b></div>
      </div>
    </div>
    <div class="cfoot">
      <div style="display:flex;gap:5px;flex-wrap:wrap">
        <span class="ftag">RSI: ${{s.rsi}}</span>
        <span class="ftag">ADX: ${{s.adx}}</span>
        <span class="ftag">H-EMA50: ₹${{s.ema50}}</span>
      </div>
      <span class="ftime">⏰ ${{s.signal_time}}</span>
    </div>
  </div>`;
}}

function renderLive(data) {{
  const el = document.getElementById('live-wrap');
  if (!data.length) {{
    el.innerHTML = `<div class="empty">
      <div class="empty-i">🔍</div>
      <div class="empty-t">No live signals right now</div>
      <div class="empty-s">All conditions must align · Runs hourly during market hours</div>
    </div>`;
    return;
  }}
  el.innerHTML = data.map(buildCard).join('');
}}

function flLive(t, btn) {{
  document.querySelectorAll('#s-live .fb').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  let d = LIVE;
  if (t === 'bull')   d = LIVE.filter(s => s.direction === 'BULLISH');
  if (t === 'bear')   d = LIVE.filter(s => s.direction === 'BEARISH');
  if (t === 'strong') d = LIVE.filter(s => (s.candle_type||'').includes('STRONG'));
  if (t === 'fo')     d = LIVE.filter(s => s.segment === 'F&O');
  if (t === 'cash')   d = LIVE.filter(s => s.segment === 'Cash');
  renderLive(d);
}}

// ── History table ─────────────────────────────────────────────────────────────
function buildRow(s) {{
  const bull   = s.direction === 'BULLISH';
  const col    = bull ? '#4ade80' : '#f87171';
  const arrow  = bull ? '▲' : '▼';
  const strong = (s.candle_type||'').includes('STRONG');
  const seccol = SCMAP[s.sector] || '#64748b';
  const slcol  = s.ema_slope === 'RISING' ? '#4ade80' : '#f87171';
  return `<tr data-dir="${{bull?'bull':'bear'}}" data-type="${{s.candle_type||''}}" data-seg="${{s.segment||''}}">
    <td style="color:#3d5870;font-family:'DM Mono',monospace;white-space:nowrap;font-size:11px">${{s.signal_time}}</td>
    <td><b style="color:${{col}};font-family:'DM Mono',monospace">${{arrow}} ${{s.symbol}}</b></td>
    <td><span class="hbadge" style="background:${{bull?'rgba(74,222,128,.15)':'rgba(248,113,113,.15)'}};color:${{col}}">${{s.direction}}</span></td>
    <td><span class="hbadge" style="background:${{strong?'rgba(74,222,128,.15)':'rgba(56,189,248,.12)'}};color:${{strong?'#4ade80':'#38bdf8'}}">${{strong?'💪 ':''}}${{s.candle_type}}</span></td>
    <td style="color:#a78bfa;font-size:11px">${{s.ema_zone}}</td>
    <td style="color:#fbbf24;font-size:11px">${{s.rsi_cond}}</td>
    <td style="font-family:'DM Mono',monospace;color:#ecf2f8;font-weight:600">₹${{s.close}}</td>
    <td style="color:#64748b;font-size:11px">${{s.rsi}}</td>
    <td style="color:#fb923c;font-size:11px">${{s.adx}}</td>
    <td style="font-weight:700;color:${{col}};font-size:11px">${{s.weekly_sig||''}}</td>
    <td style="color:#e2e8f0;font-size:11px">${{s.daily_sig||''}}</td>
    <td style="font-family:'DM Mono',monospace;color:#7a9ab8;font-size:11px">₹${{s.daily_close||'—'}}</td>
    <td style="font-family:'DM Mono',monospace;color:#3d5870;font-size:11px">₹${{s.daily_ema50||'—'}}</td>
    <td style="font-weight:600;font-size:11px;color:${{slcol}}">${{s.ema_slope||''}}</td>
    <td><span style="color:${{seccol}}">●</span> <span style="color:#3d5870;font-size:11px">${{s.sector||''}}</span></td>
    <td style="color:#1e3048;font-size:11px">${{s.segment||''}}</td>
  </tr>`;
}}

let hFilter = 'all', hSearch = '';

function renderHist() {{
  let d = [...HIST].sort((a,b) => b.signal_time.localeCompare(a.signal_time));
  if (hFilter === 'bull')   d = d.filter(s => s.direction === 'BULLISH');
  if (hFilter === 'bear')   d = d.filter(s => s.direction === 'BEARISH');
  if (hFilter === 'strong') d = d.filter(s => (s.candle_type||'').includes('STRONG'));
  if (hFilter === 'fo')     d = d.filter(s => s.segment === 'F&O');
  if (hFilter === 'cash')   d = d.filter(s => s.segment === 'Cash');
  if (hSearch) {{
    const q = hSearch.toLowerCase();
    d = d.filter(s => ((s.symbol||'') + ' ' + (s.sector||'')).toLowerCase().includes(q));
  }}
  const tb = document.getElementById('htbody');
  tb.innerHTML = d.length
    ? d.map(buildRow).join('')
    : '<tr><td colspan="16" style="text-align:center;padding:40px;color:#1e3048">No signals match filter</td></tr>';
}}

function flHist(t, btn) {{
  document.querySelectorAll('#s-hist .fb').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  hFilter = t;
  renderHist();
}}

function srchHist(v) {{ hSearch = v; renderHist(); }}

// ── CSV ───────────────────────────────────────────────────────────────────────
function doCSV() {{
  let d = [...HIST].sort((a,b) => b.signal_time.localeCompare(a.signal_time));
  if (hFilter === 'bull')   d = d.filter(s => s.direction === 'BULLISH');
  if (hFilter === 'bear')   d = d.filter(s => s.direction === 'BEARISH');
  if (hFilter === 'strong') d = d.filter(s => (s.candle_type||'').includes('STRONG'));
  if (hFilter === 'fo')     d = d.filter(s => s.segment === 'F&O');
  if (hFilter === 'cash')   d = d.filter(s => s.segment === 'Cash');
  if (hSearch) {{ const q=hSearch.toLowerCase(); d=d.filter(s=>((s.symbol||'')+' '+(s.sector||'')).toLowerCase().includes(q)); }}
  const hdr = ['Time','Symbol','Dir','Candle','EMA Zone','RSI Cond','Price','RSI','ADX','Weekly','Daily','D-Close','D-EMA50','Slope','Sector','Seg'];
  const rows = [hdr, ...d.map(s=>[s.signal_time,s.symbol,s.direction,s.candle_type,s.ema_zone,s.rsi_cond,s.close,s.rsi,s.adx,s.weekly_sig,s.daily_sig,s.daily_close||'',s.daily_ema50||'',s.ema_slope||'',s.sector||'',s.segment||''])];
  const csv = rows.map(r=>r.map(c=>'"'+String(c||'').replace(/"/g,'""')+'"').join(',')).join('\\n');
  const a = document.createElement('a');
  a.href = 'data:text/csv;charset=utf-8,\uFEFF' + encodeURIComponent(csv);
  a.download = 'ha_signals_' + new Date().toISOString().slice(0,10) + '.csv';
  a.click();
}}

// ── Charts ────────────────────────────────────────────────────────────────────
let chartsOk = false;
const cg = {{
  responsive: true, maintainAspectRatio: true,
  plugins: {{
    legend: {{ labels: {{ color:'#3d5870', font:{{size:11}}, padding:14 }} }},
    tooltip: {{ backgroundColor:'#0b1628', borderColor:'#16253a', borderWidth:1 }}
  }},
  scales: {{
    x: {{ ticks:{{color:'#3d5870'}}, grid:{{color:'rgba(22,37,58,.8)'}} }},
    y: {{ ticks:{{color:'#3d5870'}}, grid:{{color:'rgba(22,37,58,.8)'}} }}
  }}
}};

function drawCharts() {{
  if (chartsOk) return;
  chartsOk = true;

  // Daily bar
  new Chart(document.getElementById('cBar'), {{
    type:'bar',
    data:{{
      labels: {j_dates},
      datasets:[
        {{label:'Bullish', data:{j_bull}, backgroundColor:'rgba(74,222,128,.75)', borderRadius:4, borderSkipped:false}},
        {{label:'Bearish', data:{j_bear}, backgroundColor:'rgba(248,113,113,.75)', borderRadius:4, borderSkipped:false}}
      ]
    }},
    options: cg
  }});

  // Hourly bar
  const HB = {j_hbull};
  const HR = {j_hbear};
  new Chart(document.getElementById('cHour'), {{
    type:'bar',
    data:{{
      labels:['9:00','10:00','11:00','12:00','13:00','14:00','15:00'],
      datasets:[
        {{label:'Bull', data:[9,10,11,12,13,14,15].map(h=>HB[String(h)]||0), backgroundColor:'rgba(74,222,128,.7)', borderRadius:3}},
        {{label:'Bear', data:[9,10,11,12,13,14,15].map(h=>HR[String(h)]||0), backgroundColor:'rgba(248,113,113,.7)', borderRadius:3}}
      ]
    }},
    options: cg
  }});

  // Candle doughnut
  const CD = {j_candle};
  new Chart(document.getElementById('cCandle'), {{
    type:'doughnut',
    data:{{
      labels: Object.keys(CD),
      datasets:[{{ data:Object.values(CD), backgroundColor:['#4ade80','#22d3ee','#f87171','#fb923c'], borderWidth:0, hoverOffset:8 }}]
    }},
    options:{{ plugins:{{ legend:{{ labels:{{ color:'#7a9ab8', font:{{size:11}} }} }} }} }}
  }});

  // EMA doughnut
  const ED = {j_ema};
  new Chart(document.getElementById('cEma'), {{
    type:'doughnut',
    data:{{
      labels: Object.keys(ED),
      datasets:[{{ data:Object.values(ED), backgroundColor:['#38bdf8','#818cf8','#f87171','#fb923c'], borderWidth:0, hoverOffset:8 }}]
    }},
    options:{{ plugins:{{ legend:{{ labels:{{ color:'#7a9ab8', font:{{size:11}} }} }} }} }}
  }});

  // Sector horizontal
  new Chart(document.getElementById('cSec'), {{
    type:'bar', indexAxis:'y',
    data:{{
      labels: {j_secl},
      datasets:[{{ label:'Signals', data:{j_secv}, backgroundColor:{j_secc}, borderRadius:4 }}]
    }},
    options:{{
      ...cg,
      plugins:{{ ...cg.plugins, legend:{{display:false}} }},
      scales:{{
        x:{{ ticks:{{color:'#3d5870'}}, grid:{{color:'rgba(22,37,58,.8)'}} }},
        y:{{ ticks:{{color:'#7a9ab8'}}, grid:{{color:'rgba(22,37,58,.8)'}} }}
      }}
    }}
  }});
}}

// ── INIT ──────────────────────────────────────────────────────────────────────
renderLive(LIVE);
renderHist();
</script>
</body>
</html>"""


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN — with auto HTTP server + auto browser open
# ════════════════════════════════════════════════════════════════════════════════
def main():
    import threading, webbrowser, http.server, socketserver
    t0 = time.time()

    print("╔" + "═"*68 + "╗")
    print("║  HA TRIPLE SCREEN SCANNER v2.0 — Fast Parallel Engine            ║")
    print(f"║  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                                               ║")
    print("╚" + "═"*68 + "╝")

    kite = get_kite()
    now  = datetime.now()
    data_from   = now - timedelta(days=WARMUP_DAYS)
    hist_from   = now - timedelta(days=HISTORY_DAYS)
    bt_start_ts = IST.localize(hist_from)

    print("\n📡 Fetching instrument list...", flush=True)
    instruments = kite.instruments("NSE")
    token_map   = build_token_map(instruments)
    print(f"   Loaded {len(instruments)} instruments | {len(token_map)} tokens")

    symbols_to_fetch = [s for s in STOCKS if s in token_map]
    print(f"\n⚡ Fetching {len(symbols_to_fetch)} stocks [{MAX_WORKERS} threads]...", flush=True)

    stock_data, futures = {}, {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for sym in symbols_to_fetch:
            futures[ex.submit(fetch_stock, kite, token_map[sym], sym, data_from, now)] = sym
        done = 0
        for f in as_completed(futures):
            done += 1
            r = f.result()
            if r:
                sym, daily, hourly = r
                stock_data[sym] = (daily, hourly)
            if done % 20 == 0 or done == len(futures):
                print(f"   [{done}/{len(futures)}] {len(stock_data)} loaded ✓", flush=True)

    print(f"\n✅ Data: {len(stock_data)} stocks ready")

    # History scan
    print(f"\n📅 History scan ({HISTORY_DAYS}d)...", flush=True)
    history_signals = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_stock_history, sym, d, h, bt_start_ts): sym
                for sym, (d, h) in stock_data.items()}
        for f in as_completed(futs):
            history_signals.extend(f.result())
    history_signals.sort(key=lambda x: x["signal_time"])
    bh = sum(1 for s in history_signals if s["direction"] == "BULLISH")
    rh = len(history_signals) - bh
    print(f"   {len(history_signals)} signals | {bh}🟢 {rh}🔴")

    saved = load_history()
    saved, added = merge_history(saved, history_signals)
    save_history(saved)
    print(f"   +{len(added)} new saved to {HISTORY_FILE}")

    # Live scan
    print(f"\n⚡ Live scan...", flush=True)
    live_signals = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_stock_live, sym, d, h): sym
                for sym, (d, h) in stock_data.items()}
        for f in as_completed(futs):
            sig = f.result()
            if sig: live_signals.append(sig)
    live_signals.sort(key=lambda x: (x["direction"], x["symbol"]))
    bl = sum(1 for s in live_signals if s["direction"] == "BULLISH")
    rl = len(live_signals) - bl
    print(f"   {len(live_signals)} live | {bl}🟢 {rl}🔴")
    for s in live_signals:
        print(f"   {'▲' if s['direction']=='BULLISH' else '▼'} {s['symbol']} | {s['candle_type']} | {s['ema_zone']} | RSI:{s['rsi']} ADX:{s['adx']}")

    if live_signals:
        send_email(live_signals)

    # Build dashboard
    scan_dur = round(time.time() - t0, 1)
    run_time = datetime.now(IST).strftime("%d %b %Y, %H:%M IST")
    html = build_dashboard(live_signals, history_signals, run_time, scan_dur)
    out_path = os.path.abspath(DASHBOARD_OUT)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'='*70}")
    print(f"  ✅ Dashboard : {out_path}")
    print(f"  ✅ History   : {HISTORY_FILE}")
    print(f"  ⏱  Scan time: {scan_dur}s")
    print(f"  📊 Signals  : {len(live_signals)} live | {len(history_signals)} history")
    print(f"{'='*70}")

    # ── Auto HTTP server ───────────────────────────────────────────────────────
    PORT = 8877
    dash_dir  = os.path.dirname(out_path)
    dash_file = os.path.basename(out_path)
    url = f"http://localhost:{PORT}/{dash_file}"

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=dash_dir, **kw)
        def log_message(self, *a): pass

    def serve():
        with socketserver.TCPServer(("", PORT), QuietHandler) as srv:
            srv.serve_forever()

    threading.Thread(target=serve, daemon=True).start()
    time.sleep(0.4)

    print(f"\n  🌐 URL     : {url}")
    webbrowser.open(url)
    print(f"  ✅ Browser opened automatically!")
    print(f"\n  Ctrl+C to stop. Dashboard auto-refreshes on each scan run.\n")

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\n  ⏹ Server stopped.")

if __name__ == "__main__":
    main()
