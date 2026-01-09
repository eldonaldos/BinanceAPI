import json
import time
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import requests

BASE_UM = "https://fapi.binance.com"


# =============================
# CONFIG
# =============================
@dataclass
class Config:
    # Default Watchlist (easy edit)
    symbols: List[str] = field(default_factory=lambda: ["CLOUSDT", "BLESSUSDT", "RIVERUSDT"])

    # Liquidity structure timeframes
    intervals: List[str] = field(default_factory=lambda: ["15m", "4h", "1d"])

    # Candle history to fetch per timeframe (pagination)
    history_limits: Dict[str, int] = field(default_factory=lambda: {
        "15m": 3000,
        "4h": 2000,
        "1d": 1500
    })

    page_limit: int = 1500
    timeout_s: int = 20

    # Pivot clustering
    pivot_left: int = 3
    pivot_right: int = 3
    eq_cluster_threshold: float = 0.002  # 0.2%

    # Timeframe weights
    tf_weight: Dict[str, int] = field(default_factory=lambda: {"15m": 1, "4h": 3, "1d": 6})

    # HARD FILTER: zones beyond +/- X% from current price are NOT used for triggers/state/primary zones
    max_zone_distance_pct: float = 0.30  # 30%

    # Soft relevance label (near/far) for display
    relevant_distance_pct: float = 0.04  # 4%

    # Funding extremes (per funding interval)
    funding_extreme_pos: float = 0.0015
    funding_extreme_neg: float = -0.0015

    # ATR zone width
    zone_atr_mult: float = 0.35

    # Chop detection using 15m rolling 50 range relative to ATR
    chop_range_atr_mult: float = 2.2

    # Enforce active box from rolling highs/lows (15m)
    active_box_window: int = 50

    # No-trade middle band inside active box (percent of box height)
    no_trade_mid_band_pct: float = 0.35  # middle 35% of the box is "avoid" if chop-like / funding extreme


# =============================
# HTTP
# =============================
class BinanceHTTPError(RuntimeError):
    pass


def http_get(path: str, params=None, timeout=20):
    url = BASE_UM + path
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code == 451:
        raise BinanceHTTPError("HTTP 451 – Binance Futures API blockiert dieses Netzwerk (Eligibility/Region).")
    r.raise_for_status()
    return r.json()


def now_utc_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def iso_utc(ts_ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts_ms / 1000))


# =============================
# KLINES + PAGINATION
# =============================
def parse_klines(raw):
    # [ openTime, open, high, low, close, volume, closeTime, quoteVol, trades, takerBuyBase, takerBuyQuote, ignore ]
    bars = []
    for k in raw:
        bars.append({
            "openTime": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "closeTime": int(k[6]),
            "quoteVol": float(k[7]),
            "trades": int(k[8]),
            "takerBuyQuote": float(k[10]),
        })
    return bars


def fetch_klines_full(symbol: str, interval: str, target: int, cfg: Config) -> List[Dict[str, Any]]:
    """
    Fetch klines backwards using endTime paging. Returns ascending order.
    """
    candles: List[Dict[str, Any]] = []
    end_time: Optional[int] = None

    while len(candles) < target:
        params = {"symbol": symbol, "interval": interval, "limit": cfg.page_limit}
        if end_time is not None:
            params["endTime"] = end_time

        raw = http_get("/fapi/v1/klines", params, cfg.timeout_s)
        if not raw:
            break

        batch = parse_klines(raw)
        end_time = batch[0]["openTime"] - 1  # go further back next loop
        candles = batch + candles

        if len(batch) < cfg.page_limit:
            break

    if len(candles) > target:
        candles = candles[-target:]

    return candles


# =============================
# INDICATORS
# =============================
def atr14(bars: List[Dict[str, Any]]) -> Optional[float]:
    if len(bars) < 15:
        return None
    trs = []
    for i in range(1, len(bars)):
        hi = bars[i]["high"]
        lo = bars[i]["low"]
        prev = bars[i - 1]["close"]
        tr = max(hi - lo, abs(hi - prev), abs(lo - prev))
        trs.append(tr)
    return sum(trs[-14:]) / 14.0


def rolling_high_low(bars: List[Dict[str, Any]], n: int) -> Tuple[Optional[float], Optional[float]]:
    if not bars:
        return None, None
    w = bars[-n:] if len(bars) >= n else bars
    return max(x["high"] for x in w), min(x["low"] for x in w)


def pivot_swings_levels(bars: List[Dict[str, Any]], left: int, right: int) -> Tuple[List[float], List[float]]:
    highs: List[float] = []
    lows: List[float] = []
    for i in range(left, len(bars) - right):
        win = bars[i - left:i + right + 1]
        if bars[i]["high"] == max(x["high"] for x in win):
            highs.append(bars[i]["high"])
        if bars[i]["low"] == min(x["low"] for x in win):
            lows.append(bars[i]["low"])
    return highs, lows


def cluster_equal_levels(levels: List[float], threshold_rel: float) -> List[Dict[str, Any]]:
    if not levels:
        return []
    levels = sorted(levels)
    clusters: List[List[float]] = [[levels[0]]]

    for x in levels[1:]:
        last = clusters[-1][-1]
        rel = abs(x - last) / abs(last) if last != 0 else abs(x - last)
        if rel <= threshold_rel:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    out = []
    for c in clusters:
        out.append({
            "level": sum(c) / len(c),
            "count": len(c),
            "min": min(c),
            "max": max(c),
        })
    out.sort(key=lambda d: d["count"], reverse=True)
    return out


def distance_pct(price: float, level: float) -> Optional[float]:
    if price == 0:
        return None
    return (level - price) / price


# =============================
# LIQUIDITY STRUCTURE PER TF
# =============================
def build_liquidity_tf(tf: str, bars: List[Dict[str, Any]], cfg: Config) -> Dict[str, Any]:
    if len(bars) < 80:
        return {"error": "not enough candles"}

    closed = bars[:-1] if len(bars) > 1 else bars
    last = closed[-1]

    a = atr14(closed)
    r50_hi, r50_lo = rolling_high_low(closed, 50)
    r200_hi, r200_lo = rolling_high_low(closed, 200)

    piv_hi, piv_lo = pivot_swings_levels(closed, cfg.pivot_left, cfg.pivot_right)
    eq_high = cluster_equal_levels(piv_hi, cfg.eq_cluster_threshold)[:12]
    eq_low = cluster_equal_levels(piv_lo, cfg.eq_cluster_threshold)[:12]

    lookback_high = max(x["high"] for x in closed)
    lookback_low = min(x["low"] for x in closed)

    return {
        "timeframe": tf,
        "last_closed": last,
        "atr14": a,
        "atr14_pct": (a / last["close"]) if a and last["close"] else None,
        "lookback_high": lookback_high,
        "lookback_low": lookback_low,
        "rolling_50": {"high": r50_hi, "low": r50_lo},
        "rolling_200": {"high": r200_hi, "low": r200_lo},
        "equal_high_zones": eq_high,
        "equal_low_zones": eq_low,
    }


# =============================
# ACTIVE BOX + ZONE PRIORITY (FIX)
# =============================
def build_active_box(liq_15m: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """
    Defines the 'active trading box' from rolling window high/low (15m).
    This box is ALWAYS used for triggers/state/primary zones.
    """
    r = liq_15m.get("rolling_50") or {}
    hi = r.get("high")
    lo = r.get("low")
    if hi is None or lo is None:
        return {"status": "insufficient"}

    height = hi - lo
    if height <= 0:
        return {"status": "insufficient"}

    mid = (hi + lo) / 2.0

    # middle "no trade band" inside the box
    band = cfg.no_trade_mid_band_pct
    band = min(max(band, 0.05), 0.80)
    band_half = (height * band) / 2.0
    mid_from = mid - band_half
    mid_to = mid + band_half

    return {
        "status": "ok",
        "window": cfg.active_box_window,
        "high": hi,
        "low": lo,
        "mid": mid,
        "height": height,
        "no_trade_mid_band": {"from": mid_from, "to": mid_to},
    }


def zone_is_usable(price: float, level: float, cfg: Config) -> bool:
    d = distance_pct(price, level)
    if d is None:
        return False
    return abs(d) <= cfg.max_zone_distance_pct


def prioritize_zones(price: float, all_tf_liq: Dict[str, Any], active_box: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """
    FIXED:
    1) Hard-filter zones too far from current price.
    2) Always inject active_box edges as the PRIMARY zones with dominant score.
    3) Keep other zones only for context if they are within distance filter.
    """
    zones: List[Dict[str, Any]] = []

    # Inject active box edges as "forced" primary liquidity
    if active_box.get("status") == "ok":
        ah = float(active_box["high"])
        al = float(active_box["low"])

        zones.append({
            "type": "ACTIVE_TOP",
            "timeframe": "15m",
            "weight": 999,
            "count": 999,
            "score": 999 * 999,
            "level": ah,
            "min": ah,
            "max": ah,
            "distance_pct": distance_pct(price, ah),
            "relevance": "near" if abs(distance_pct(price, ah) or 999) <= cfg.relevant_distance_pct else "far",
            "forced": True
        })

        zones.append({
            "type": "ACTIVE_BOTTOM",
            "timeframe": "15m",
            "weight": 999,
            "count": 999,
            "score": 999 * 999,
            "level": al,
            "min": al,
            "max": al,
            "distance_pct": distance_pct(price, al),
            "relevance": "near" if abs(distance_pct(price, al) or 999) <= cfg.relevant_distance_pct else "far",
            "forced": True
        })

    # Add usable equal-high/low zones (filtered)
    for tf, liq in all_tf_liq.items():
        if not isinstance(liq, dict) or liq.get("error"):
            continue
        w = cfg.tf_weight.get(tf, 1)

        for z in liq.get("equal_high_zones", []):
            lvl = float(z["level"])
            if not zone_is_usable(price, lvl, cfg):
                continue
            d = distance_pct(price, lvl)
            zones.append({
                "type": "EQH",
                "timeframe": tf,
                "weight": w,
                "count": int(z["count"]),
                "score": w * int(z["count"]),
                "level": lvl,
                "min": float(z.get("min", lvl)),
                "max": float(z.get("max", lvl)),
                "distance_pct": d,
                "relevance": "near" if (d is not None and abs(d) <= cfg.relevant_distance_pct) else "far",
                "forced": False
            })

        for z in liq.get("equal_low_zones", []):
            lvl = float(z["level"])
            if not zone_is_usable(price, lvl, cfg):
                continue
            d = distance_pct(price, lvl)
            zones.append({
                "type": "EQL",
                "timeframe": tf,
                "weight": w,
                "count": int(z["count"]),
                "score": w * int(z["count"]),
                "level": lvl,
                "min": float(z.get("min", lvl)),
                "max": float(z.get("max", lvl)),
                "distance_pct": d,
                "relevance": "near" if (d is not None and abs(d) <= cfg.relevant_distance_pct) else "far",
                "forced": False
            })

    # Sort forced first, then score, then closeness
    def sort_key(x):
        forced_rank = 0 if x.get("forced") else 1
        closeness = abs(x["distance_pct"]) if x.get("distance_pct") is not None else 999
        return (forced_rank, -x["score"], closeness)

    zones.sort(key=sort_key)

    # Primary: forced box edges
    primary_top = next((z for z in zones if z["type"] == "ACTIVE_TOP"), None)
    primary_bottom = next((z for z in zones if z["type"] == "ACTIVE_BOTTOM"), None)

    # Provide additional nearby zones for context
    top_eqh = [z for z in zones if z["type"] in ("EQH", "ACTIVE_TOP")][:8]
    bot_eql = [z for z in zones if z["type"] in ("EQL", "ACTIVE_BOTTOM")][:8]
    top_all = zones[:16]

    return {
        "primary_top": primary_top,
        "primary_bottom": primary_bottom,
        "top_zones": top_eqh,
        "bottom_zones": bot_eql,
        "top_all": top_all,
        "filter": {
            "max_zone_distance_pct": cfg.max_zone_distance_pct,
            "relevant_distance_pct": cfg.relevant_distance_pct
        }
    }


# =============================
# STATE FLAGS + TRADE PERMISSION (FIX)
# =============================
def compute_state(price: float, funding: float, liq_15m: Dict[str, Any], active_box: Dict[str, Any],
                  priority: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    atr = liq_15m.get("atr14")
    atr = atr if isinstance(atr, (int, float)) else None
    zone_width = (cfg.zone_atr_mult * atr) if atr else None

    # Funding state
    if funding <= cfg.funding_extreme_neg:
        funding_state = "extreme_negative"
    elif funding >= cfg.funding_extreme_pos:
        funding_state = "extreme_positive"
    else:
        funding_state = "normal"

    # Chop detection using active box range relative to ATR
    chop_likely = None
    if atr and active_box.get("status") == "ok":
        height = float(active_box["height"])
        chop_likely = height <= (cfg.chop_range_atr_mult * atr)

    # Near edges (active)
    near_top = None
    near_bottom = None
    if zone_width and active_box.get("status") == "ok":
        top = float(active_box["high"])
        bot = float(active_box["low"])
        near_top = (top - zone_width) <= price <= (top + zone_width)
        near_bottom = (bot - zone_width) <= price <= (bot + zone_width)

    # Middle band
    in_mid_band = None
    if active_box.get("status") == "ok":
        mb = active_box["no_trade_mid_band"]
        in_mid_band = float(mb["from"]) <= price <= float(mb["to"])

    # Distribution/Accumulation heuristics
    in_distribution = bool(near_top and funding_state == "extreme_negative")
    in_accumulation = bool(near_bottom and funding_state == "extreme_positive")

    # Trade permission logic (more practical):
    # - NO_TRADE if (chop_likely OR funding extreme) AND in mid band
    # - WAIT_TRIGGER if funding extreme AND not near edges
    # - OK if near edges OR normal funding and not chop
    trade_permission = "OK"
    if (chop_likely is True or funding_state != "normal") and in_mid_band is True:
        trade_permission = "NO_TRADE"
    elif funding_state != "normal" and (near_top is False and near_bottom is False):
        trade_permission = "WAIT_TRIGGER"

    return {
        "funding_state": funding_state,
        "atr14_15m": atr,
        "zone_width": zone_width,
        "chop_likely": chop_likely,
        "near_top_liquidity": near_top,
        "near_bottom_liquidity": near_bottom,
        "in_mid_band": in_mid_band,
        "in_distribution": in_distribution,
        "in_accumulation": in_accumulation,
        "trade_permission": trade_permission
    }


# =============================
# TRIGGERS (FIX)
# =============================
def compute_triggers(price: float, liq_15m: Dict[str, Any], active_box: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """
    All triggers are derived from the ACTIVE BOX, not from historical far zones.
    """
    if active_box.get("status") != "ok":
        return {"status": "insufficient"}

    top = float(active_box["high"])
    bot = float(active_box["low"])
    zone_width = state.get("zone_width")

    out: Dict[str, Any] = {"status": "ok"}

    out["active_box"] = {
        "high": top,
        "low": bot,
        "mid": float(active_box["mid"]),
        "no_trade_mid_band": active_box["no_trade_mid_band"]
    }

    if zone_width:
        out["top_sweep_zone"] = {"from": top - zone_width, "to": top + zone_width}
        out["bottom_sweep_zone"] = {"from": bot - zone_width, "to": bot + zone_width}
        # breakout/breakdown: edge + zone_width (sweep-confirm)
        out["breakout_trigger"] = top + zone_width
        out["breakdown_trigger"] = bot - zone_width
    else:
        out["top_sweep_zone"] = None
        out["bottom_sweep_zone"] = None
        out["breakout_trigger"] = top
        out["breakdown_trigger"] = bot

    return out


# =============================
# TOKEN PACK
# =============================
def build_token_pack(symbol: str, cfg: Config) -> Dict[str, Any]:
    premium = http_get("/fapi/v1/premiumIndex", {"symbol": symbol}, cfg.timeout_s)
    oi = http_get("/fapi/v1/openInterest", {"symbol": symbol}, cfg.timeout_s)
    ticker = http_get("/fapi/v1/ticker/24hr", {"symbol": symbol}, cfg.timeout_s)

    price = float(ticker["lastPrice"])
    funding = float(premium["lastFundingRate"])
    oi_now = float(oi["openInterest"])

    # Fetch liquidity timeframes
    kline_sets: Dict[str, List[Dict[str, Any]]] = {}
    liq: Dict[str, Any] = {}

    for tf in cfg.intervals:
        target = cfg.history_limits.get(tf, 1000)
        bars = fetch_klines_full(symbol, tf, target, cfg)
        kline_sets[tf] = bars
        liq[tf] = build_liquidity_tf(tf, bars, cfg)

    liq_15m = liq.get("15m", {})

    # ACTIVE BOX from 15m rolling levels
    active_box = build_active_box(liq_15m, cfg)

    # Priority zones (now forced to active box edges + filtered zones near price)
    priority = prioritize_zones(price, liq, active_box, cfg)

    # State flags using active box
    state = compute_state(price, funding, liq_15m, active_box, priority, cfg)

    # Triggers from active box edges
    triggers = compute_triggers(price, liq_15m, active_box, state)

    # Bias (mechanical)
    if funding <= cfg.funding_extreme_neg:
        bias = "LONG_BIASED"
    elif funding >= cfg.funding_extreme_pos:
        bias = "SHORT_BIASED"
    else:
        bias = "NEUTRAL"

    snapshot = {
        "timestamp_utc": now_utc_str(),
        "lastPrice": price,
        "markPrice": float(premium["markPrice"]),
        "indexPrice": float(premium["indexPrice"]),
        "basis_mark_minus_index": float(premium["markPrice"]) - float(premium["indexPrice"]),
        "funding_last": funding,
        "nextFundingTime_utc": iso_utc(int(premium["nextFundingTime"])),
        "openInterest_now": oi_now,
        "quoteVolume_24h": float(ticker["quoteVolume"]),
        "high_24h": float(ticker["highPrice"]),
        "low_24h": float(ticker["lowPrice"]),
    }

    klines_tail = {tf: bars[-300:] for tf, bars in kline_sets.items()}

    return {
        "meta": {
            "symbol": symbol,
            "generated_at_utc": now_utc_str(),
            "intervals": cfg.intervals,
            "config": {
                "max_zone_distance_pct": cfg.max_zone_distance_pct,
                "active_box_window": cfg.active_box_window,
                "no_trade_mid_band_pct": cfg.no_trade_mid_band_pct
            }
        },
        "snapshot": snapshot,
        "bias": bias,
        "active_box": active_box,
        "priority_zones": priority,
        "state": state,
        "triggers": triggers,
        "liquidity": liq,
        "kline_counts": {tf: len(bars) for tf, bars in kline_sets.items()},
        "klines_tail": klines_tail
    }


# =============================
# CLI / MAIN
# =============================
def parse_symbols_from_args(args: List[str], defaults: List[str]) -> List[str]:
    """
    Usage:
      python binance_pack_multi_signals.py
      python binance_pack_multi_signals.py CLOUSDT
      python binance_pack_multi_signals.py CLOUSDT,BLESSUSDT,RIVERUSDT
    """
    if len(args) < 2:
        return defaults
    raw = args[1].strip().upper()
    if "," in raw:
        syms = [s.strip() for s in raw.split(",") if s.strip()]
        return syms if syms else defaults
    return [raw]


def main():
    cfg = Config()
    cfg.symbols = parse_symbols_from_args(sys.argv, cfg.symbols)

    all_out: Dict[str, Any] = {
        "meta": {
            "generated_at_utc": now_utc_str(),
            "symbols": cfg.symbols,
            "intervals": cfg.intervals
        },
        "tokens": {}
    }

    for sym in cfg.symbols:
        print(f"Fetching {sym} ...")
        pack = build_token_pack(sym, cfg)
        all_out["tokens"][sym] = pack

        fn = f"{sym}_pack.json"
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(pack, f, indent=2)
        print(f"  saved: {fn}")

    fn_all = "binance_pack_all.json"
    with open(fn_all, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2)
    print(f"\n✅ saved combined: {fn_all}")


if __name__ == "__main__":
    main()

