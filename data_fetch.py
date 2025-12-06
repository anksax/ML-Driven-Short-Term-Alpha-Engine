import ccxt
import pandas as pd
import argparse
import time
from datetime import datetime, timedelta

def fetch_ohlcv(symbol='BTC/USDT', timeframe='1m', since_days=7, limit=1000):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
    })
    now = exchange.milliseconds()
    since = now - since_days * 24 * 60 * 60 * 1000
    all_bars = []
    since_iter = since
    while since_iter < now:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_iter, limit=limit)
        except Exception as e:
            print("Error fetching OHLCV:", e)
            time.sleep(1)
            continue
        if not bars:
            break
        all_bars += bars
        since_iter = bars[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000.0)
    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('datetime').drop(columns=['timestamp'])
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='BTC/USDT')
    p.add_argument('--since_days', type=int, default=7)
    p.add_argument('--outfile', default='data/btc_usdt_1m.csv')
    args = p.parse_args()
    df = fetch_ohlcv(symbol=args.symbol, since_days=args.since_days)
    print("Fetched", len(df), "rows")
    df.to_csv(args.outfile)
    print("Saved to", args.outfile)
