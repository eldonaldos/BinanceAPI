import os
import time
import json
import requests
from datetime import datetime, timezone

FAPI = "https://fapi.binance.com"

TOKENS = ["CLOUSDT", "RIVERUSDT", "BLESSUSDT"]
TIMEFRAMES = ["15m", "1h", "4h", "1d"]

TIMEFRAME_TO_INTERVAL = {
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

def http_get(path: str, params: dict, timeout_s: int = 10):
    url = f"{FAPI}{path}"
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    return r.json()

def get_mark_price(symbol: str):
    data = http_get("/fapi/v1/premiumIndex", {"symbol": symbol})
    return {
        "markPrice": float(data["markPrice"]),
        "lastFundingRate": float(data["lastFundingRate"]),
        "time": int(data["time"]),
    }

def get_open_interest(symbol: str):
    data = http_get("/fapi/v1/openInterest", {"symbol": symbol})
    return float(data["openInterest"])

def get_last_price(symbol: str):
    data = http_get("/fapi/v1/ticker/price", {"symbol": symbol})
    return float(data["price"])

def get_klines(symbol: str, interval: str, limit: int = 2):
    data = http_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    return data

def get_taker_buy_sell(symbol: str, period: str, limit: int = 1):
    """
    Binance Futures Data endpoint expects period like: 5m, 15m, 1h, 4h, 1d
    Endpoint:
    /futures/data/takerlongshortRatio OR /futures/data/takerBuySellVol

    In deinem früheren Error war der Pfad falsch.
    Der korrekte Pfad ist:
    /futures/data/takerBuySellVol
    """
    data = http_get("/futures/data/takerBuySellVol", {"symbol": symbol, "period": period, "limit": limit})
    # returns list of dicts
    return data

def utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def fmt_pct(x: float) -> str:
    return f"{x*100:.3f}%"

def fmt_num(x: float) -> str:
    if abs(x) >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.2f}K"
    return f"{x:.4f}"

def build_snapshot(symbol: str, tf: str):
    interval = TIMEFRAME_TO_INTERVAL[tf]

    last_price = get_last_price(symbol)
    mp = get_mark_price(symbol)
    oi = get_open_interest(symbol)

    kl = get_klines(symbol, interval=interval, limit=2)
    # kline fields: openTime, open, high, low, close, volume, closeTime, quoteAssetVolume, trades, takerBuyBase, takerBuyQuote, ignore
    last_closed = kl[-2]
    open_ = float(last_closed[1])
    high_ = float(last_closed[2])
    low_ = float(last_closed[3])
    close_ = float(last_closed[4])
    vol_ = float(last_closed[5])
    quote_vol_ = float(last_closed[7])
    taker_buy_quote_ = float(last_closed[10])

    taker = get_taker_buy_sell(symbol, period=tf, limit=1)
    # fields include: buyVol, sellVol, buyVolValue, sellVolValue, timestamp
    taker0 = taker[0]
    buy_q = float(taker0.get("buyVolValue", 0.0))
    sell_q = float(taker0.get("sellVolValue", 0.0))

    return {
        "symbol": symbol,
        "tf": tf,
        "timestamp_utc": utc_now_str(),
        "price": last_price,
        "markPrice": mp["markPrice"],
        "fundingRate": mp["lastFundingRate"],
        "openInterest": oi,
        "candle": {
            "open": open_,
            "high": high_,
            "low": low_,
            "close": close_,
            "volume": vol_,
            "quoteVol": quote_vol_,
            "takerBuyQuote": taker_buy_quote_,
        },
        "takerBuySell": {
            "buyQuote": buy_q,
            "sellQuote": sell_q,
        },
    }

def summarize_symbol(symbol: str, snapshots: list[dict]) -> str:
    lines = []
    lines.append(f"{symbol}  UTC {utc_now_str()}")

    for s in snapshots:
        tf = s["tf"]
        price = s["price"]
        mp = s["markPrice"]
        fr = s["fundingRate"]
        oi = s["openInterest"]
        c = s["candle"]
        buyq = s["takerBuySell"]["buyQuote"]
        sellq = s["takerBuySell"]["sellQuote"]

        candle_dir = "G" if c["close"] >= c["open"] else "R"
        range_ = c["high"] - c["low"]
        cvd = buyq - sellq

        lines.append(
            f"{tf}  P {price:.4f}  MP {mp:.4f}  FR {fmt_pct(fr)}  OI {fmt_num(oi)}  "
            f"C {candle_dir}  H {c['high']:.4f} L {c['low']:.4f} Rng {range_:.4f}  "
            f"CVDq {fmt_num(cvd)}"
        )

    return "\n".join(lines)

def send_telegram(text: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in env")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()

def main():
    all_msgs = []
    for symbol in TOKENS:
        snaps = []
        for tf in TIMEFRAMES:
            try:
                snaps.append(build_snapshot(symbol, tf))
                time.sleep(0.15)
            except Exception as e:
                snaps.append({
                    "symbol": symbol,
                    "tf": tf,
                    "timestamp_utc": utc_now_str(),
                    "error": str(e),
                })

        msg = summarize_symbol(symbol, [s for s in snaps if "error" not in s])
        errs = [s for s in snaps if "error" in s]
        if errs:
            msg += "\nErrors:"
            for er in errs:
                msg += f"\n{er['tf']}  {er['error']}"

        all_msgs.append(msg)

    final_text = "\n\n".join(all_msgs)
    send_telegram(final_text)

if __name__ == "__main__":
    main()
