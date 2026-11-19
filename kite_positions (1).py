"""
Kite (Zerodha) position tracker.

Setup:
  pip install kiteconnect

s1.ini (same folder):
  [kite]
  api_key = xxxx
  api_secret = yyyy          # only needed for --login

  [telegram]                 # optional
  bot_token = 123:ABC
  chat_id = 123456

Daily (token expires every morning):
  python kite_positions.py --login     # opens login URL, paste request_token, saves Kite_token.txt
Then:
  python kite_positions.py             # starts tracking
"""
import configparser
import csv
import os
import sys
import threading
import queue
import time
import urllib.parse
import urllib.request
from datetime import datetime

from kiteconnect import KiteConnect, exceptions

# ---------------- Settings ----------------
POLL_SEC = 5              # how often positions are fetched
PRINT_SEC = 15            # how often the full table is printed
MAX_LOSS = -10000.0        # alert when total P&L falls to this (set None to disable)
TARGET = 11000.0          # alert when total P&L reaches this (set None to disable)
POS_MAX_LOSS = -3000.0    # per-position loss alert (set None to disable)
ALERT_COOLDOWN = 300      # seconds before the same alert repeats
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(SCRIPT_DIR, "Kite_token.txt")
CONFIG_FILE = os.path.join(SCRIPT_DIR, "s1.ini")
LOG_FILE = os.path.join(SCRIPT_DIR, "position_log.csv")
RUN_LOG = os.path.join(SCRIPT_DIR, "position_monitor_log.txt")

TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

_tg_queue = queue.Queue()
_log_lock = threading.Lock()


def log(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    try:
        with _log_lock, open(RUN_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}\n")
    except Exception as e:
        print("Log write failed:", e)


def _tg_worker():
    while True:
        text = _tg_queue.get()
        try:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": TG_CHAT, "text": text}).encode()
            urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
        except Exception as e:
            log("Telegram send failed:", e)


def send_telegram(text):
    if TG_TOKEN and TG_CHAT:
        _tg_queue.put(text)


# ---------------- Login helper ----------------
def do_login(api_key, api_secret):
    kite = KiteConnect(api_key=api_key)
    print("1) Open this URL, login, and copy 'request_token' from the redirected URL:")
    print(kite.login_url())
    request_token = input("2) Paste request_token here: ").strip()
    data = kite.generate_session(request_token, api_secret=api_secret)
    with open(TOKEN_FILE, "w") as f:
        f.write(data["access_token"])
    print("Access token saved to", TOKEN_FILE)


# ---------------- Tracker ----------------
class PositionTracker:
    def __init__(self, kite):
        self.kite = kite
        self.prev_qty = None          # (exchange, symbol, product) -> qty
        self.last_alert = {}
        self.last_print = 0

    def fetch(self):
        net = self.kite.positions()["net"]
        return {(p["exchange"], p["tradingsymbol"], p["product"]): p for p in net}

    def alert(self, key, title, detail):
        now = time.time()
        if now - self.last_alert.get(key, 0) < ALERT_COOLDOWN:
            return
        self.last_alert[key] = now
        stamp = datetime.now().strftime("%H:%M:%S")
        log(f"[{stamp}] !! {title:<14} {detail}")
        new = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "alert", "detail"])
            w.writerow([stamp, title, detail])
        send_telegram(f"{title}\n{detail}\n{stamp}")

    def check_changes(self, cur):
        cur_qty = {k: p["quantity"] for k, p in cur.items()}
        if self.prev_qty is None:
            self.prev_qty = cur_qty
            open_n = sum(1 for q in cur_qty.values() if q != 0)
            log(f"Tracking started. Open positions: {open_n}")
            return
        for k, q in cur_qty.items():
            old = self.prev_qty.get(k, 0)
            sym = k[1]
            if old == 0 and q != 0:
                side = "BUY" if q > 0 else "SELL"
                self.alert((k, "NEW", q), "NEW_POSITION", f"{sym} {side} qty {q} @ avg {cur[k]['average_price']:.2f}")
            elif old != 0 and q == 0:
                self.alert((k, "CLOSED", old), "CLOSED", f"{sym} closed | P&L {cur[k]['pnl']:+.2f}")
            elif old != q:
                self.alert((k, "QTY", q), "QTY_CHANGE", f"{sym} qty {old} -> {q}")
        self.prev_qty = cur_qty

    def check_limits(self, cur):
        total = sum(p["pnl"] for p in cur.values())
        if MAX_LOSS is not None and total <= MAX_LOSS:
            self.alert(("TOTAL", "LOSS"), "MAX_LOSS_HIT", f"Total P&L {total:+.2f} (limit {MAX_LOSS:,.0f})")
        if TARGET is not None and total >= TARGET:
            self.alert(("TOTAL", "TARGET"), "TARGET_HIT", f"Total P&L {total:+.2f} (target {TARGET:,.0f})")
        if POS_MAX_LOSS is not None:
            for k, p in cur.items():
                if p["quantity"] != 0 and p["pnl"] <= POS_MAX_LOSS:
                    self.alert((k, "POSLOSS"), "POSITION_LOSS", f"{k[1]} P&L {p['pnl']:+.2f} (limit {POS_MAX_LOSS:,.0f})")
        return total

    def print_table(self, cur, total):
        if time.time() - self.last_print < PRINT_SEC:
            return
        self.last_print = time.time()
        log(f"--- {datetime.now():%H:%M:%S} ---")
        log(f"{'Symbol':<26}{'Prod':<6}{'Qty':>7}{'Avg':>10}{'LTP':>10}{'P&L':>12}")
        for k, p in sorted(cur.items(), key=lambda kv: kv[1]["quantity"] == 0):
            log(f"{k[1]:<26}{k[2]:<6}{p['quantity']:>7}{p['average_price']:>10.2f}"
                f"{p['last_price']:>10.2f}{p['pnl']:>12.2f}")
        log(f"{'TOTAL P&L':<59}{total:>12.2f}")

    def run(self):
        while True:
            now_hm = datetime.now().strftime("%H:%M")
            if not (MARKET_OPEN <= now_hm < MARKET_CLOSE):
                time.sleep(30)
                continue
            try:
                cur = self.fetch()
                self.check_changes(cur)
                total = self.check_limits(cur)
                self.print_table(cur, total)
            except exceptions.TokenException:
                log("Access token expired/invalid. Run: python kite_positions.py --login")
                send_telegram("Kite token expired. Login again.")
                sys.exit(1)
            except Exception as e:
                log("Fetch failed:", e)
            time.sleep(POLL_SEC)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    if not config.read(CONFIG_FILE):
        sys.exit(f"s1.ini not found at: {CONFIG_FILE}\nPut s1.ini in the same folder as this script.")
    if not config.has_section("kite"):
        sys.exit(f"No [kite] section in {CONFIG_FILE}\n"
                 f"Sections found: {config.sections()}\n"
                 "Add:\n[kite]\napi_key = YOUR_KEY\napi_secret = YOUR_SECRET")
    api_key = config["kite"]["api_key"]

    if config.has_section("telegram"):
        TG_TOKEN = config["telegram"].get("bot_token", TG_TOKEN).strip()
        TG_CHAT = config["telegram"].get("chat_id", TG_CHAT).strip()

    if "--login" in sys.argv:
        do_login(api_key, config["kite"]["api_secret"])
        sys.exit(0)

    if not os.path.exists(TOKEN_FILE):
        print("Kite_token.txt not found, starting login...")
        do_login(api_key, config["kite"]["api_secret"])

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(open(TOKEN_FILE).read().strip())

    threading.Thread(target=_tg_worker, daemon=True).start()
    log("Starting Kite position tracker...")
    send_telegram("Kite position tracker started")
    try:
        PositionTracker(kite).run()
    except KeyboardInterrupt:
        log("Stopped.")
