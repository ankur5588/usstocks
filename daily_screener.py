import datetime
import concurrent.futures
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import sys
import contextlib
import logging
import threading

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

class suppress_output:
    _lock = threading.Lock()
    _depth = 0
    _saved_stdout = None
    _saved_stderr = None
    _devnull = None

    def __enter__(self):
        with suppress_output._lock:
            if suppress_output._depth == 0:
                suppress_output._saved_stdout = sys.stdout
                suppress_output._saved_stderr = sys.stderr
                sys.stdout.flush()
                sys.stderr.flush()
                suppress_output._devnull = open(os.devnull, "w")
                sys.stdout = suppress_output._devnull
                sys.stderr = suppress_output._devnull
            suppress_output._depth += 1

    def __exit__(self, *args):
        with suppress_output._lock:
            suppress_output._depth -= 1
            if suppress_output._depth == 0:
                sys.stdout.flush()
                sys.stderr.flush()
                sys.stdout.close()
                sys.stdout = suppress_output._saved_stdout
                sys.stderr = suppress_output._saved_stderr
                suppress_output._devnull = None
                suppress_output._saved_stdout = None
                suppress_output._saved_stderr = None

# --- Configuration via environment variables ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "25"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def send_telegram_message(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            print(f"Telegram API error: {r.text}")
            return False
        return True
    except Exception as e:
        print(f"Telegram request failed: {e}")
        return False

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def get_all_us_tickers():
    print("Fetching US stock universe...")
    tickers = []
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            raw = response.text.split('\n')
            for symbol in raw:
                symbol = symbol.strip().upper()
                if symbol and symbol.isalpha() and len(symbol) <= 4:
                    tickers.append(symbol)
        tickers = sorted(set(tickers))
        print(f"Retrieved {len(tickers)} US equities.")
        return tickers
    except Exception as e:
        print(f"Fetch failed ({e}). Using fallback list.")
        return ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "NFLX", "TSLA", "AVGO", "QCOM", "COST"]

def check_criteria(df_daily):
    if len(df_daily) < 200:
        return False

    today = df_daily.iloc[-1]
    yesterday = df_daily.iloc[-2]

    # 1. Expanding range: today's range > yesterday's range
    current_range = today["High"] - today["Low"]
    yesterday_range = yesterday["High"] - yesterday["Low"]
    if current_range <= yesterday_range:
        return False

    # 2. Daily close > daily open
    if not (today["Close"] > today["Open"]):
        return False

    # 3. Daily close > 1 day ago close
    if not (today["Close"] > yesterday["Close"]):
        return False

    # 4. Yesterday volume > 500000
    if not (yesterday["Volume"] > 500000):
        return False

    # 5. SMA(20) > SMA(50) — short-term uptrend
    sma20 = df_daily["Close"].rolling(20).mean().iloc[-1]
    sma50 = df_daily["Close"].rolling(50).mean().iloc[-1]
    if not (sma20 > sma50):
        return False

    # 6. SMA(50) > SMA(200) — long-term uptrend
    sma200 = df_daily["Close"].rolling(200).mean().iloc[-1]
    if not (sma50 > sma200):
        return False

    # 7. RSI(14) > 30
    rsi14 = calculate_rsi(df_daily["Close"]).iloc[-1]
    if not (rsi14 > 30):
        return False

    # 8. Volume spike > 1.5x 20-day average
    vol_avg20 = df_daily["Volume"].rolling(20).mean().iloc[-2]
    if vol_avg20 > 0 and yesterday["Volume"] / vol_avg20 <= 1.5:
        return False

    return True

def evaluate_single_ticker(ticker):
    try:
        with suppress_output():
            stock = yf.Ticker(ticker)
            df_d = stock.history(period="1y", interval="1d")

        if df_d.empty or "Close" not in df_d.columns:
            return None

        df_d = df_d[["Open", "High", "Low", "Close", "Volume"]].copy()

        if check_criteria(df_d):
            return ticker
    except Exception:
        pass
    return None

def scan_full_us_market():
    tickers = get_all_us_tickers()
    matched_stocks = []

    print(f"\nScanning {len(tickers)} US stocks with {MAX_WORKERS} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(evaluate_single_ticker, tickers)
        for index, result in enumerate(results):
            if result:
                print(f"[MATCH] {result}")
                matched_stocks.append(result)
            if (index + 1) % 500 == 0:
                print(f"Progress: {index + 1}/{len(tickers)}")

    return matched_stocks

def save_results_to_file(matched_stocks, elapsed):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = os.path.join(RESULTS_DIR, f"screener_{date_str}.txt")
    count = len(matched_stocks)

    lines = [
        f"US Stocks Screener - {date_str}",
        f"{'='*50}",
        f"Runtime: {elapsed}",
        f"Total Matches: {count}",
        f"{'='*50}",
    ]
    if matched_stocks:
        lines.append("Matched Stocks:")
        lines.extend(matched_stocks)
    else:
        lines.append("No stocks matched the criteria today.")

    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Results saved to {filename}")
    return filename

def send_results_to_telegram(matched_stocks, elapsed):
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    count = len(matched_stocks)

    if count == 0:
        msg = (
            f"<b>US Stocks Screener — {date_str}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"No stocks matched the criteria today."
        )
        send_telegram_message(msg)
        return

    header = (
        f"<b>US Stocks Screener — {date_str}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Matches: {count}  |  Runtime: {elapsed}\n\n"
    )

    chunk_size = 30
    for i in range(0, count, chunk_size):
        chunk = matched_stocks[i:i + chunk_size]
        ticker_list = ", ".join(chunk)
        if i == 0:
            msg = header + ticker_list
        else:
            msg = f"<b>Cont'd ({i+1}-{min(i+chunk_size, count)})</b>\n{ticker_list}"
        send_telegram_message(msg)

def main():
    start_time = datetime.datetime.now()
    print(f"Screener started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Telegram configured: {bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)}")

    matched_stocks = scan_full_us_market()
    elapsed = datetime.datetime.now() - start_time

    print(f"\n{'='*50}")
    print(f"Total Matches: {len(matched_stocks)}")
    print(f"Runtime: {elapsed}")
    if matched_stocks:
        print(f"Stocks: {matched_stocks}")

    save_results_to_file(matched_stocks, str(elapsed).split('.')[0])
    send_results_to_telegram(matched_stocks, str(elapsed).split('.')[0])
    print("Done. Results saved and sent to Telegram.")

if __name__ == "__main__":
    main()
