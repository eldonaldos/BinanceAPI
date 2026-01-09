import os
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TOKENS = ["CLOUSDT", "BLESSUSDT", "RIVERUSDT"]  # hier easy erweitern

TIMEFRAMES = ["15m", "1h", "4h", "1d"]          # wie du willst: 15m, 1h, 4h, daily

OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def now_utc_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def run_pack(pack_script: str, symbol: str, tf: str) -> dict:
    """
    Erwartet, dass binance_pack.py eine JSON-Zeile auf stdout ausgibt.
    Falls dein pack-script anders arbeitet, sag kurz wie (Argumente/Output),
    dann passe ich runner+workflow an.
    """
    cmd = ["python", pack_script, "--symbol", symbol, "--timeframe", tf]
    p = subprocess.run(cmd, capture_output=True, text=True)

    if p.returncode != 0:
        return {
            "meta": {"symbol": symbol, "timeframe": tf, "timestamp_utc": now_utc_str()},
            "error": "pack_script_failed",
            "stderr": p.stderr[-4000:],
            "stdout": p.stdout[-4000:],
            "returncode": p.returncode,
        }

    # nimmt die letzte nicht-leere Zeile als JSON
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    if not lines:
        return {
            "meta": {"symbol": symbol, "timeframe": tf, "timestamp_utc": now_utc_str()},
            "error": "no_output_from_pack_script",
        }

    try:
        return json.loads(lines[-1])
    except Exception as e:
        return {
            "meta": {"symbol": symbol, "timeframe": tf, "timestamp_utc": now_utc_str()},
            "error": "invalid_json_from_pack_script",
            "exception": str(e),
            "raw_last_line": lines[-1][:2000],
        }

def main():
    pack_script = os.getenv("PACK_SCRIPT", "binance_pack.py")

    summary = {
        "run_timestamp_utc": now_utc_str(),
        "pack_script": pack_script,
        "tokens": TOKENS,
        "timeframes": TIMEFRAMES,
        "files_written": []
    }

    for sym in TOKENS:
        for tf in TIMEFRAMES:
            data = run_pack(pack_script, sym, tf)
            fname = OUT_DIR / f"{sym}_{tf}.json"
            fname.write_text(json.dumps(data, indent=2), encoding="utf-8")
            summary["files_written"].append(str(fname))

    # immer ein Summary schreiben
    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
