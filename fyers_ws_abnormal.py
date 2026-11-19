import configparser
import csv
import os
import signal
import statistics
import sys
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from datetime import datetime

import matplotlib.pyplot as plt
from fyers_apiv3.FyersWebsocket import data_ws

# ---------------- Set your expiry here ----------------
# Weekly: YY + month code + DD  (e.g. 10 Nov 2026 -> "26N10", 5 Dec 2026 -> "26D05")
# Monthly: YY + MON             (e.g. "26NOV")
EXPIRY = "26NOV"

# ---------------- Telegram (phone alerts) ----------------
# Do NOT hardcode the token here. Put it in s1.ini:
#
#   [telegram]
#   bot_token = 123456789:ABC...
#   chat_id = 123456789
#
# or set environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()


# ---------------- Abnormal activity settings ----------------
BUCKET_SEC = 10          # ticks are grouped into buckets of this many seconds
HISTORY = 60             # buckets kept per symbol as its own baseline (~10 min)
MIN_HISTORY = 12         # buckets needed before z-score alerts start (~2 min)
VOL_Z = 8             # volume-in-bucket z-score threshold
MIN_BUCKET_VOL = 10000   # ignore small volume bursts (quantity, not lots)
PREM_JUMP_PCT = 15.0     # % premium move inside one bucket
MIN_PREMIUM = 5.0        # ignore moves in options priced below this
SIDE_RATIO = 3.0         # CE vs PE volume imbalance within a bucket
SIDE_MIN_VOL = 10000     # minimum total volume in bucket for imbalance alert
CHAIN_SURGE = 2.0        # whole chain trading this many x its normal = market-wide move
RELATIVE_FACTOR = 2.0    # a strike must beat the market-wide surge by this factor
MARKET_MOVE_COUNT = 6    # this many strikes jumping together = one MARKET_MOVE alert
ALERT_COOLDOWN = 60      # seconds before the same symbol+alert type repeats
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(SCRIPT_DIR, "alerts_log.csv")       # alerts only (CSV)
RUN_LOG = os.path.join(SCRIPT_DIR, "monitor_log.txt")       # everything printed to console
MARKET_OPEN = "09:15"      # watchdog only runs from this time
MARKET_CLOSE = "15:30"     # no alerts after this time (HH:MM)
MIN_BURST_PREMIUM = 2.0    # ignore volume bursts on options priced below this
NO_TICK_TIMEOUT = 30       # seconds without any tick before the watchdog warns
STATUS_COOLDOWN = 120      # min seconds between repeated connection-status Telegram messages


_tg_queue = queue.Queue()
_log_lock = threading.Lock()


def log(*args):
    """Print to console and append to monitor_log.txt in the script's folder."""
    text = " ".join(str(a) for a in args)
    print(text)
    try:
        with _log_lock, open(RUN_LOG, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}\n")
    except Exception as e:
        print("Log write failed:", e)


def _tg_worker():
    """One long-running thread that sends queued messages (started once at startup)."""
    while True:
        text = _tg_queue.get()
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": text}).encode()
            urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            log(f"Telegram error {e.code}: {body}")
        except Exception as e:
            log("Telegram send failed:", e)


def send_telegram(text):
    """Queue a message for your phone. Never blocks the data feed."""
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        _tg_queue.put(text)


class FyersOptionChainWebsocket:
    def __init__(self, client_id, token, expiry):
        self.client_id = client_id
        self.token = token
        self.stage = 0  # 0 = waiting for spot price, 1 = subscribed to options

        self.expiry = expiry  # e.g. weekly "26N10" or monthly "26NOV"
        self.spot = None
        self.spot_bucket_open = None   # Nifty spot at start of the bucket
        self.atm_strike_value = None

        self.strikes = []
        self.symbol_info = {}   # symbol -> (strike, type)
        self.ltp_data = {}      # symbol -> last traded price
        self.last_vol = {}      # symbol -> last cumulative day volume seen

        # Bucket state
        self.bucket_start = time.time()
        self.bucket_vol = defaultdict(int)        # symbol -> volume traded in bucket
        self.bucket_open_ltp = {}                 # symbol -> ltp at bucket start
        self.vol_hist = defaultdict(lambda: deque(maxlen=HISTORY))
        self.chain_hist = deque(maxlen=HISTORY)   # total chain volume per bucket
        self.last_alert = {}                      # (symbol, kind) -> time

        # History for plotting (one point per bucket)
        self.times = []
        self.ce_vol_cum = []
        self.pe_vol_cum = []
        self._ce_total = 0
        self._pe_total = 0

        self.ws = None
        self.has_subscribed_options = False

        # Connection health
        self.last_tick = time.time()
        self._last_status_msg = 0

    # ------------------------------------------------------------
    def atm_strike(self, spot):
        return round(spot / 50) * 50

    def prepare_option_symbols(self):
        self.strikes = list(range(self.atm_strike_value - 500, self.atm_strike_value + 501, 50))
        symbols = []
        for strike in self.strikes:
            for opt_type in ("CE", "PE"):
                symbol = f"NSE:NIFTY{self.expiry}{strike}{opt_type}"
                symbols.append(symbol)
                self.symbol_info[symbol] = (strike, opt_type)
                self.ltp_data[symbol] = None
        return symbols

    def notify_status(self, text):
        """Connection-status Telegram message, rate limited so it can't spam."""
        now = time.time()
        if now - self._last_status_msg >= STATUS_COOLDOWN:
            self._last_status_msg = now
            send_telegram(text)

    # ------------------------------------------------------------
    def raise_alert(self, symbol, kind, detail, ltp=None):
        now = time.time()
        key = (symbol, kind)
        if now - self.last_alert.get(key, 0) < ALERT_COOLDOWN:
            return
        self.last_alert[key] = now

        stamp = datetime.now().strftime("%H:%M:%S")
        log(f"[{stamp}] !! {symbol:<28} {kind:<14} {detail}")

        new = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "symbol", "alert", "detail", "ltp", "nifty"])
            w.writerow([stamp, symbol, kind, detail, ltp, self.spot])

        short = symbol.replace("NSE:", "")
        send_telegram(f"{kind}\n{short}\n{detail}\n{stamp}")

    def close_bucket(self):
        ce_dv = pe_dv = 0
        for sym, (strike, typ) in self.symbol_info.items():
            dv = self.bucket_vol.get(sym, 0)
            if typ == "CE":
                ce_dv += dv
            else:
                pe_dv += dv
        chain_dv = ce_dv + pe_dv

        # How busy is the whole chain compared with its own normal?
        chain_ratio = 1.0
        if len(self.chain_hist) >= MIN_HISTORY:
            cm = statistics.mean(self.chain_hist)
            if cm > 0:
                chain_ratio = chain_dv / cm

        if chain_ratio >= CHAIN_SURGE and chain_dv >= SIDE_MIN_VOL:
            self.raise_alert("CHAIN", "MARKET_SURGE",
                             f"whole chain {chain_ratio:.1f}x normal | CE {ce_dv:,} vs PE {pe_dv:,}")

        jumps = []
        for sym, (strike, typ) in self.symbol_info.items():
            dv = self.bucket_vol.get(sym, 0)
            ltp = self.ltp_data.get(sym)
            hist = self.vol_hist[sym]

            # 1) Volume burst that stands out beyond any market-wide surge
            if dv >= MIN_BUCKET_VOL and len(hist) >= MIN_HISTORY:
                sd = statistics.pstdev(hist)
                mean = statistics.mean(hist)
                z = (dv - mean) / sd if sd > 0 else 0
                strike_ratio = dv / mean if mean > 0 else 0
                if z >= VOL_Z and strike_ratio >= RELATIVE_FACTOR * max(chain_ratio, 1.0) and (ltp or 0) >= MIN_BURST_PREMIUM:
                    o = self.bucket_open_ltp.get(sym)
                    move = f" | premium {o:.2f} -> {ltp:.2f}" if o and ltp else ""
                    self.raise_alert(sym, "VOLUME_BURST",
                                     f"{dv:,} in {BUCKET_SEC}s ({strike_ratio:.1f}x its normal, z={z:.1f}){move}", ltp)

            # 2) Sharp premium move inside one bucket
            open_ltp = self.bucket_open_ltp.get(sym)
            if ltp and open_ltp and open_ltp >= MIN_PREMIUM:
                pct = (ltp - open_ltp) / open_ltp * 100
                if abs(pct) >= PREM_JUMP_PCT:
                    jumps.append((sym, typ, open_ltp, ltp, pct, dv))

            hist.append(dv)
            self.bucket_open_ltp[sym] = ltp

        # Premium jumps: many at once = the index moved, so send one summary
        if len(jumps) >= MARKET_MOVE_COUNT:
            ce_up = sum(1 for j in jumps if j[1] == "CE" and j[4] > 0)
            ce_dn = sum(1 for j in jumps if j[1] == "CE" and j[4] < 0)
            pe_up = sum(1 for j in jumps if j[1] == "PE" and j[4] > 0)
            pe_dn = sum(1 for j in jumps if j[1] == "PE" and j[4] < 0)
            spot_txt = ""
            if self.spot and self.spot_bucket_open:
                spot_txt = f"Nifty {self.spot_bucket_open:.2f} -> {self.spot:.2f} ({self.spot - self.spot_bucket_open:+.2f}) | "
            self.raise_alert("CHAIN", "MARKET_MOVE",
                             f"{spot_txt}{len(jumps)} strikes jumped | CE up {ce_up}/down {ce_dn}, PE up {pe_up}/down {pe_dn}")
        else:
            for sym, typ, o, l, pct, dv in jumps:
                self.raise_alert(sym, "PREMIUM_JUMP", f"{o:.2f} -> {l:.2f} ({pct:+.1f}%) vol {dv:,}", l)

        # 3) CE vs PE flow imbalance (whole chain)
        if chain_dv >= SIDE_MIN_VOL:
            hi, lo = max(ce_dv, pe_dv), max(min(ce_dv, pe_dv), 1)
            if hi / lo >= SIDE_RATIO:
                side = "CE" if ce_dv > pe_dv else "PE"
                self.raise_alert(f"CHAIN-{side}", "SIDE_IMBALANCE",
                                 f"CE {ce_dv:,} vs PE {pe_dv:,} in {BUCKET_SEC}s")

        self.chain_hist.append(chain_dv)

        # Plot history (cumulative)
        self._ce_total += ce_dv
        self._pe_total += pe_dv
        self.times.append(datetime.now())
        self.ce_vol_cum.append(self._ce_total)
        self.pe_vol_cum.append(self._pe_total)

        self.spot_bucket_open = self.spot
        self.bucket_vol.clear()
        self.bucket_start = time.time()

    # ------------------------------------------------------------
    def on_message(self, msg):
        self.last_tick = time.time()

        symbol = msg.get("symbol")
        ltp = msg.get("ltp")
        # Fyers sends cumulative day volume as 'vol_traded_today'
        volume = msg.get("vol_traded_today", msg.get("volume"))

        if self.stage == 0:
            if symbol == "NSE:NIFTY50-INDEX" and ltp is not None:
                self.spot = ltp
                self.spot_bucket_open = ltp
                self.atm_strike_value = self.atm_strike(self.spot)
                log(f"Spot: {self.spot}, ATM strike: {self.atm_strike_value}")

                symbols = self.prepare_option_symbols()
                if not self.has_subscribed_options:
                    log(f"Subscribing to {len(symbols)} option symbols...")
                    self.ws.subscribe(symbols=symbols, data_type="SymbolUpdate")
                    self.has_subscribed_options = True
                self.stage = 1
                self.bucket_start = time.time()
            return

        if datetime.now().strftime("%H:%M") >= MARKET_CLOSE:
            return

        if symbol == "NSE:NIFTY50-INDEX" and ltp is not None:
            self.spot = ltp
            return

        if symbol in self.symbol_info:
            if ltp is not None:
                self.ltp_data[symbol] = ltp
                self.bucket_open_ltp.setdefault(symbol, ltp)

            if volume is not None:
                prev = self.last_vol.get(symbol)
                # first tick only sets the baseline, so no fake huge delta
                if prev is not None and volume > prev:
                    self.bucket_vol[symbol] += volume - prev
                self.last_vol[symbol] = volume

            if time.time() - self.bucket_start >= BUCKET_SEC:
                self.close_bucket()

    def on_error(self, error):
        log("WebSocket error:", error)
        self.notify_status(f"Feed error: {error}")

    def on_close(self, status_code, reason):
        log(f"WebSocket closed: {status_code} - {reason}")
        self.notify_status(f"Feed closed: {status_code} {reason}")

    def on_open(self):
        if self.has_subscribed_options:
            # Reconnect: options must be re-subscribed, and old volume
            # baselines dropped (volume kept growing while we were offline).
            log("Reconnected. Re-subscribing and resetting baselines...")
            self.last_vol.clear()
            self.bucket_vol.clear()
            self.bucket_open_ltp.clear()
            self.bucket_start = time.time()
            self.last_tick = time.time()
            self.ws.subscribe(
                symbols=["NSE:NIFTY50-INDEX"] + list(self.symbol_info.keys()),
                data_type="SymbolUpdate",
            )
            self.notify_status("Feed reconnected, options re-subscribed")
        else:
            log("WebSocket connected. Subscribing to NIFTY spot...")
            self.ws.subscribe(symbols=["NSE:NIFTY50-INDEX"], data_type="SymbolUpdate")
        self.ws.keep_running()

    def plot_sums(self):
        if not self.times:
            log("No data to plot.")
            return
        plt.figure(figsize=(14, 6))
        plt.plot(self.times, self.ce_vol_cum, label="CE volume (cumulative since start)", color="blue")
        plt.plot(self.times, self.pe_vol_cum, label="PE volume (cumulative since start)", color="orange")
        plt.xlabel("Time")
        plt.ylabel("Volume")
        plt.title("CE vs PE traded volume")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def start(self):
        self.ws = data_ws.FyersDataSocket(
            access_token=self.token,
            log_path="",
            litemode=False,
            write_to_file=False,
            reconnect=True,
            on_connect=self.on_open,
            on_close=self.on_close,
            on_error=self.on_error,
            on_message=self.on_message,
        )
        self.ws.connect()

    def stop(self):
        if self.ws:
            self.ws.close_connection()


def handle_exit(sig, frame):
    log("Exiting and plotting...")
    option_ws.stop()
    option_ws.plot_sums()
    sys.exit(0)


if __name__ == "__main__":
    base = SCRIPT_DIR
    config = configparser.ConfigParser()
    config.read(os.path.join(base, "s1.ini"))
    client_id = config["fyers"]["client_id"]
    expiry = EXPIRY
    token = open(os.path.join(base, "Ts.txt")).read().strip()

    # Telegram credentials: s1.ini [telegram] overrides environment variables
    if config.has_section("telegram"):
        TELEGRAM_BOT_TOKEN = config["telegram"].get("bot_token", TELEGRAM_BOT_TOKEN).strip()
        TELEGRAM_CHAT_ID = config["telegram"].get("chat_id", TELEGRAM_CHAT_ID).strip()

    option_ws = FyersOptionChainWebsocket(client_id, token, expiry)
    signal.signal(signal.SIGINT, handle_exit)

    threading.Thread(target=_tg_worker, daemon=True).start()
    log("Starting Nifty Option Chain WebSocket...")
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        log("Telegram not configured: alerts will only print and log to CSV.")
    send_telegram("Option chain monitor started")
    option_ws.start()

    # keep the main thread alive so Ctrl+C (and the final plot) still works;
    # also acts as a watchdog if the feed goes silent during market hours
    while True:
        time.sleep(1)
        now_hm = datetime.now().strftime("%H:%M")
        if (option_ws.stage == 1
                and MARKET_OPEN <= now_hm < MARKET_CLOSE
                and time.time() - option_ws.last_tick > NO_TICK_TIMEOUT):
            log(f"No ticks for {NO_TICK_TIMEOUT}s, feed may be dead")
            send_telegram(f"No ticks for {NO_TICK_TIMEOUT}s, feed may be dead")
            option_ws.last_tick = time.time()   # avoid spamming every second
