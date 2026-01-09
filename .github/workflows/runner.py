import os
import json
import subprocess
import requests
from datetime import datetime, timezone

SYMBOLS = ["CLOUSDT", "BLESSUSDT", "RIVERUSDT"]

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
GIST_ID = os.environ["GIST_ID"]
TOKEN_GIST = os.environ["GITHUB_TOKEN_GIST"]

# ---- SET THIS to your pack script filename ----
PACK_SCRIPT = "binance_pack_multi_signals.py"  # <-- change if needed

def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def tg_send(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True
    }, timeout=20)
    r.raise_for_status()

def gist_update(filename: str, content: str):
    url = f"https://api.github.com/gists/{GIST_ID}"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN_GIST}",
        "Accept": "application/vnd.github+json"
    }
    payload = {"files": {filename: {"content": content}}}
    r = requests.patch(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()

def load_pack(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compact_line(symbol: str, pack: dict) -> str:
    snap = pack.get("snapshot", {})
    px = snap.get("lastPrice")
    fr = snap.get("funding_last")
    oi = snap.get("openInterest_now")
    return f"{symbol}: px={px} funding={fr} OI={oi}"

def main():
    # 1) run your pack script
    # Assumes it creates files like CLOUSDT_pack.json etc.
    cmd = ["python", PACK_SCRIPT, ",".join(SYMBOLS)]
    subprocess.run(cmd, check=True)

    # 2) load packs, update gist, build telegram message
    lines = [f"Run {utc_now()} UTC"]
    for sym in SYMBOLS:
        fn = f"{sym}_pack.json"
        pack = load_pack(fn)

        # update gist
        gist_update(fn, json.dumps(pack, indent=2))

        lines.append(compact_line(sym, pack))

    # also update a combined file if present
    if os.path.exists("binance_pack_all.json"):
        gist_update("binance_pack_all.json", open("binance_pack_all.json", "r", encoding="utf-8").read())

    tg_send("\n".join(lines))

if __name__ == "__main__":
    main()
