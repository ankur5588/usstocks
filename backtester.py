import datetime
import concurrent.futures
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import os
import sys
import json
import time
import itertools
import pickle

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
INITIAL_CAPITAL = 1000.0
TARGET_PCT = 0.03
BACKTEST_MONTHS = 6
UNIVERSE_SIZE = 500

def get_all_us_tickers():
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
        return tickers
    except Exception as e:
        print(f"Fetch failed ({e}).")
        return ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "META", "GOOGL", "NFLX", "TSLA", "AVGO", "QCOM", "COST"]

def download_data(ticker, max_retries=3):
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y", interval="1d")
            if df.empty or "Close" not in df.columns:
                return None
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            if "Rate limited" in str(e) and attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            return None
    return None

def download_universe():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, "data.pkl")
    tickers = get_all_us_tickers()[:UNIVERSE_SIZE]
    print(f"Downloading data for {len(tickers)} tickers...")
    all_data = {}
    done = 0
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        fut_to_ticker = {ex.submit(download_data, t): t for t in tickers}
        for fut in concurrent.futures.as_completed(fut_to_ticker):
            ticker = fut_to_ticker[fut]
            try:
                df = fut.result()
                if df is not None:
                    all_data[ticker] = add_indicators(df)
                else:
                    errors += 1
            except Exception:
                errors += 1
            done += 1
            if done % 100 == 0:
                print(f"  Downloaded {done}/{len(tickers)} (errors: {errors})")
                time.sleep(2)
    if len(all_data) < 10:
        print("Too few real tickers, using cached data.")
        return {}
    print(f"Downloaded {len(all_data)} tickers ({errors} errors)")
    with open(cache_file, "wb") as f:
        pickle.dump(all_data, f)
    print(f"Saved to {cache_file}")
    return all_data

def load_data():
    cache_file = os.path.join(CACHE_DIR, "data.pkl")
    if os.path.exists(cache_file):
        print("Loading cached data...")
        with open(cache_file, "rb") as f:
            all_data = pickle.load(f)
        if len(all_data) < 10:
            print("Cache too small, re-downloading...")
            return download_universe()
        sample = next(iter(all_data.values()))
        if "atr14" not in sample.columns or "volume_ratio" not in sample.columns:
            print("Upgrading cache with new indicators...")
            for ticker in all_data:
                all_data[ticker] = add_indicators(all_data[ticker])
            with open(cache_file, "wb") as f:
                pickle.dump(all_data, f)
        print(f"Loaded {len(all_data)} tickers from cache")
        return all_data
    return download_universe()

def load_spy():
    cache_file = os.path.join(CACHE_DIR, "spy.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            spy = pickle.load(f)
        if "atr14" in spy.columns:
            return spy
    print("Loading SPY data...")
    spy = yf.download("SPY", period="2y", progress=False)
    if not spy.empty:
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.droplevel(1)
        spy.columns = [c.lower() for c in spy.columns]
        spy = spy[["open", "high", "low", "close", "volume"]].copy()
        spy = add_indicators(spy)
        with open(cache_file, "wb") as f:
            pickle.dump(spy, f)
        return spy
    return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    df = df.copy()
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["ema10"] = df["close"].ewm(span=10, adjust=False).mean()
    df["rsi14"] = calculate_rsi(df["close"])
    df["range"] = df["high"] - df["low"]
    df["atr14"] = df["range"].rolling(14).mean()
    df["volume_avg20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_avg20"]
    return df

def check_entry(df, idx, params, date=None, spy_df=None):
    if idx < 200:
        return False
    today = df.iloc[idx]
    yesterday = df.iloc[idx - 1]

    if params.get("require_range", True):
        range_days = params.get("range_days", 1)
        current_range = today["range"]
        for i in range(1, range_days + 1):
            if current_range <= df.iloc[idx - i]["range"]:
                return False

    if params.get("require_close_gt_open", True):
        if not (today["close"] > today["open"]):
            return False

    if params.get("require_close_gt_prev", True):
        if not (today["close"] > yesterday["close"]):
            return False

    if params.get("require_weekly", True):
        df_week = df.iloc[:idx + 1]["close"].resample("W").ohlc()
        if len(df_week) >= 2:
            if not (df_week.iloc[-1]["close"] > df_week.iloc[-1]["open"]):
                return False

    if params.get("require_monthly", True):
        df_month = df.iloc[:idx + 1]["close"].resample("ME").ohlc()
        if len(df_month) >= 2:
            if not (df_month.iloc[-1]["close"] > df_month.iloc[-1]["open"]):
                return False

    if params.get("require_volume", True):
        vol_threshold = params.get("volume_threshold", 500000)
        if not (yesterday["volume"] > vol_threshold):
            return False

    if params.get("require_sma20_gt_50", True):
        if not (today["sma20"] > today["sma50"]):
            return False

    if params.get("require_sma50_gt_200", False):
        if not (today["sma50"] > today["sma200"]):
            return False

    rsi_threshold = params.get("rsi_threshold", 30)
    if not (today["rsi14"] > rsi_threshold):
        return False

    if params.get("require_volume_spike", False):
        vol_mult = params.get("volume_spike_mult", 1.5)
        if not (yesterday["volume_ratio"] > vol_mult):
            return False

    if params.get("require_price_gt_sma50", False):
        if not (today["close"] > today["sma50"]):
            return False

    if params.get("require_price_gt_sma200", False):
        if not (today["close"] > today["sma200"]):
            return False

    if params.get("require_atr_min", False):
        atr_min = params.get("atr_min_val", 0.5)
        if not (today["atr14"] > atr_min):
            return False

    if params.get("require_market_uptrend", False) and spy_df is not None and date is not None:
        if date not in spy_df.index:
            return False
        spy_row = spy_df.loc[date]
        if not (float(spy_row["close"]) > float(spy_row["sma50"])):
            return False

    return True

def signal_score(df, idx):
    today = df.iloc[idx]
    yesterday = df.iloc[idx - 1]
    vol_score = min(yesterday["volume_ratio"], 5) if not np.isnan(yesterday["volume_ratio"]) else 1
    range_score = today["range"] / yesterday["range"] if yesterday["range"] > 0 else 1
    rsi_score = (today["rsi14"] - 30) / 20 if not np.isnan(today["rsi14"]) else 0.5
    return vol_score * range_score * max(rsi_score, 0.1)

def run_backtest(data, params, spy_df=None):
    positions = {}
    trades = []
    equity_curve = []
    cash = INITIAL_CAPITAL
    position_size = params.get("position_size", 0.25)
    top_n = params.get("top_n", 0)

    all_dates = sorted(set(
        date for df in data.values() for date in df.index
    ))

    lookback = pd.Timestamp.today() - pd.DateOffset(months=BACKTEST_MONTHS)
    all_dates_tz = all_dates[0].tz if hasattr(all_dates[0], 'tz') and all_dates[0].tz else None
    if all_dates_tz:
        lookback = lookback.tz_localize(all_dates_tz)
    backtest_dates = [d for d in all_dates if d >= lookback]

    stop_type = params.get("stop_type", "trail")
    target_mult = params.get("target_pct", 0)
    hard_stop_pct = params.get("hard_stop_pct", 0)
    hold_days = params.get("hold_days", 0)

    for date in backtest_dates:
        to_close = []
        for ticker, pos in positions.items():
            if ticker not in data:
                to_close.append(ticker)
                continue
            df = data[ticker]
            if date not in df.index:
                continue
            if pos["entry_date"] >= date:
                continue
            row = df.loc[date]
            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])

            # Update trailing stop if enabled (skip when using hard stop)
            hard_stop_pct = params.get("hard_stop_pct", 0)
            if stop_type == "trail" and not hard_stop_pct:
                trail_pct = params.get("trail_pct", 0.01)
                pos["stop_price"] = max(pos["stop_price"], high * (1 - trail_pct))
            elif stop_type == "ema10":
                ema_val = float(row["ema10"]) if not np.isnan(row["ema10"]) else pos["stop_price"]
                pos["stop_price"] = max(pos["stop_price"], ema_val * 0.98)
            elif stop_type == "ema20":
                ema20 = df["close"].ewm(span=20, adjust=False).mean().loc[date]
                ema_val = float(ema20) if not np.isnan(ema20) else pos["stop_price"]
                pos["stop_price"] = max(pos["stop_price"], ema_val * 0.98)

            exit_price = None
            exit_reason = ""

            # Target exit
            if pos["target_price"] is not None and high >= pos["target_price"]:
                exit_price = pos["target_price"]
                exit_reason = "target"

            # Hard/trailing stop exit
            if low <= pos["stop_price"] and (exit_price is None or low < pos["stop_price"]):
                exit_price = min(pos["stop_price"], close)
                exit_reason = "stop"

            # Time-based exit (hold_days)
            if hold_days > 0:
                held = (date - pos["entry_date"]).days
                if held >= hold_days:
                    if exit_price is None:
                        exit_price = close
                        exit_reason = "time"
                    elif close < exit_price and exit_reason != "stop":
                        exit_price = close
                        exit_reason = "time"

            if exit_price is not None:
                pos["ticker"] = ticker
                pos["exit_date"] = date
                pos["exit_price"] = exit_price
                pos["pnl"] = (exit_price - pos["entry_price"]) * pos["shares"]
                pos["pnl_pct"] = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
                pos["exit_reason"] = exit_reason
                cash += pos["shares"] * exit_price
                trades.append(pos)
                to_close.append(ticker)

        for t in to_close:
            positions.pop(t, None)

        positions_value = 0.0
        for ticker, pos in positions.items():
            if ticker in data and date in data[ticker].index:
                positions_value += pos["shares"] * float(data[ticker].loc[date]["close"])
        equity_curve.append({"date": date, "equity": cash + positions_value})

        candidates = []
        for ticker, df in data.items():
            if ticker in positions:
                continue
            if date not in df.index:
                continue
            idx = df.index.get_loc(date)
            if check_entry(df, idx, params, date, spy_df):
                score = signal_score(df, idx)
                candidates.append((ticker, idx, score))

        max_concurrent = params.get("max_concurrent", 0)
        if max_concurrent > 0 and len(positions) >= max_concurrent:
            candidates = []

        if top_n > 0 and len(candidates) > top_n:
            candidates.sort(key=lambda x: x[2], reverse=True)
            candidates = candidates[:top_n]

        for ticker, idx, _ in candidates:
            df = data[ticker]
            entry_date = df.index[idx]
            entry_price = float(df.iloc[idx]["close"])
            if entry_price <= 0 or np.isnan(entry_price):
                continue

            target_price = None
            if target_mult > 0:
                target_price = entry_price * (1 + target_mult)

            hard_stop_pct = params.get("hard_stop_pct", 0)
            if hard_stop_pct > 0:
                stop_price = entry_price * (1 - hard_stop_pct)
            elif stop_type == "trail":
                trail_pct = params.get("trail_pct", 0.01)
                stop_price = entry_price * (1 - trail_pct)
            elif stop_type == "ema10":
                ema10_val = float(df.iloc[idx]["ema10"])
                stop_price = ema10_val if not np.isnan(ema10_val) else entry_price * 0.95
            elif stop_type == "ema20":
                ema20 = df["close"].ewm(span=20, adjust=False).mean().iloc[idx]
                stop_price = float(ema20) if not np.isnan(ema20) else entry_price * 0.95
            elif stop_type == "atr":
                atr = (df["high"] - df["low"]).rolling(14).mean().iloc[idx]
                stop_price = entry_price - float(atr) * 1.5 if not np.isnan(atr) else entry_price * 0.95
            else:
                stop_price = entry_price * 0.95

            shares = cash * position_size / entry_price
            cost = shares * entry_price
            if cost > cash:
                shares = cash / entry_price
                cost = cash
            if shares <= 0:
                continue
            cash -= cost
            positions[ticker] = {
                "entry_date": entry_date,
                "entry_price": entry_price,
                "target_price": target_price,
                "stop_price": stop_price,
                "shares": shares,
                "cost": cost,
            }

    for ticker, pos in list(positions.items()):
        if ticker in data:
            last_date = data[ticker].index[-1]
            last_close = float(data[ticker].iloc[-1]["close"])
            pos["ticker"] = ticker
            pos["exit_date"] = last_date
            pos["exit_price"] = last_close
            pos["pnl"] = (last_close - pos["entry_price"]) * pos["shares"]
            pos["pnl_pct"] = (last_close - pos["entry_price"]) / pos["entry_price"] * 100
            pos["exit_reason"] = "end"
            cash += pos["shares"] * last_close
            trades.append(pos)

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "final_equity": cash,
        "total_return": (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
    }

def compute_metrics(result):
    trades = result["trades"]
    if not trades:
        return {
            "total_return": 0, "win_rate": 0, "num_trades": 0,
            "avg_win": 0, "avg_loss": 0, "max_drawdown": 0,
            "profit_factor": 0, "sharpe": 0,
        }

    df_equity = pd.DataFrame(result["equity_curve"])
    if df_equity.empty:
        return {"total_return": result["total_return"], "num_trades": len(trades)}

    df_equity["peak"] = df_equity["equity"].cummax()
    df_equity["drawdown"] = (df_equity["peak"] - df_equity["equity"]) / df_equity["peak"] * 100
    max_dd = df_equity["drawdown"].max()

    wins = [t for t in trades if t["pnl"] is not None and t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] is not None and t["pnl"] <= 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0

    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    returns = df_equity["equity"].pct_change().dropna()
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0

    return {
        "total_return": result["total_return"],
        "win_rate": win_rate,
        "num_trades": len(trades),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_dd,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
    }

PARAM_GRID = {
    "range_days": [1],
    "rsi_threshold": [30],
    "require_close_gt_open": [True],
    "require_close_gt_prev": [True],
    "require_weekly": [False],
    "require_monthly": [False],
    "require_volume": [True],
    "volume_threshold": [500000],
    "require_sma20_gt_50": [True],
    "require_sma50_gt_200": [True],
    "require_volume_spike": [True],
    "volume_spike_mult": [1.5],
    "require_price_gt_sma50": [False],
    "require_atr_min": [False],
    "require_market_uptrend": [False],
    "position_size": [0.66],
    "stop_type": ["trail"],
    "hold_days": [15],
    "hard_stop_pct": [0.08],
    "trail_pct": [0.002],
    "max_concurrent": [0],
    "target_pct": [0],
    "top_n": [0],
}

def flatten_param_grid(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def print_report(results, top_n=10):
    print("\n" + "=" * 80)
    print(f"BACKTEST RESULTS - Top {top_n} Configurations")
    print(f"Period: Last {BACKTEST_MONTHS} months")
    print(f"Initial Capital: ${INITIAL_CAPITAL}")
    print(f"Target: {TARGET_PCT*100}% | Stop: EMA10 (trailing)")
    print("=" * 80)

    for i, r in enumerate(results[:top_n]):
        p = r["params"]
        print(f"\n{'─' * 80}")
        print(f"#{i+1} | Return: {r['total_return']:+.2f}% | Sharpe: {r['sharpe']:.2f} | WinRate: {r['win_rate']:.1f}%")
        print(f"    Trades: {r['num_trades']} | AvgWin: ${r['avg_win']:.2f} | AvgLoss: ${r['avg_loss']:.2f}")
        print(f"    MaxDD: {r['max_drawdown']:.1f}% | ProfitFactor: {r['profit_factor']:.2f}")
        config_str = f"Config: range={p['range_days']}d rsi>{p['rsi_threshold']} sz={p['position_size']:.0%} stop=trail({p.get('trail_pct','')}) tgt={p['target_pct']:.0%}%"
        if p.get("require_volume_spike"):
            config_str += f" vol_spike={p['volume_spike_mult']}x"
        if p.get("require_weekly"):
            config_str += " weekly=True"
        if p.get("require_price_gt_sma50"):
            config_str += " price>sma50=True"
        if p.get("require_sma50_gt_200"):
            config_str += " sma50>200=True"
        if p.get("require_market_uptrend"):
            config_str += " market=True"
        if p.get("top_n", 0) > 0:
            config_str += f" top_n={p['top_n']}"
        print(f"    {config_str}")

def print_best(best, p):
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION")
    print("=" * 80)
    print(json.dumps(p, indent=2, default=str))
    print(f"Return: {best['total_return']:+.2f}%")
    print(f"Sharpe: {best['sharpe']:.2f}")
    print(f"Win Rate: {best['win_rate']:.1f}%")
    print(f"Trades: {best['num_trades']}")
    print(f"Max Drawdown: {best['max_drawdown']:.1f}%")
    print(f"Avg Win: ${best['avg_win']:.2f} | Avg Loss: ${best['avg_loss']:.2f}")
    print(f"Profit Factor: {best['profit_factor']:.2f}")

def main():
    data = load_data()
    spy_df = load_spy()

    total = 1
    for v in PARAM_GRID.values():
        total *= len(v)

    print("\n" + "=" * 80)
    print("FULL OPTIMIZATION")
    print("=" * 80)
    print(f"\nTesting {total} parameter combinations...")

    all_results = []
    for i, params in enumerate(flatten_param_grid(PARAM_GRID)):
        result = run_backtest(data, params, spy_df)
        metrics = compute_metrics(result)
        metrics["params"] = params
        all_results.append(metrics)
        if (i + 1) % 5 == 0 or i == 0 or i == total - 1:
            print(f"  [{i+1}/{total}] return={metrics['total_return']:+.2f}%  trades={metrics['num_trades']}  wr={metrics['win_rate']:.0f}%  dd={metrics['max_drawdown']:.1f}%")

    all_results.sort(key=lambda r: r["total_return"], reverse=True)

    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimization_results.json")
    serializable = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "params"}
        sr["params"] = dict(r["params"])
        serializable.append(sr)
    with open(output_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Full results saved to {output_file}")

    print_report(all_results)
    print_best(all_results[0], all_results[0]["params"])

    return all_results

if __name__ == "__main__":
    main()
