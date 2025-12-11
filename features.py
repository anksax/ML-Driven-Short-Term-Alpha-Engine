# features.py (V2)
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator

def add_features(df):
    df = df.copy()
    # basic returns
    df['ret_1'] = df['close'].pct_change()
    df['logret_1'] = np.log(df['close']).diff()
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_15'] = df['close'].pct_change(15)

    # moving averages
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_15'] = df['close'].rolling(15).mean()
    df['ma_60'] = df['close'].rolling(60).mean()
    df['ma_5_15'] = df['ma_5'] - df['ma_15']

    # volatility
    df['vol_10'] = df['logret_1'].rolling(10).std()
    df['vol_60'] = df['logret_1'].rolling(60).std()

    # ATR
    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr_14'] = atr.average_true_range()

    # RSI
    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi_14'] = rsi.rsi()

    # Bollinger bands pct
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_percent'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-12)

    # target: next-minute direction (keep for supervised)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df = df.dropna()
    return df

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--infile', default='data/btc_usdt_1m.csv')
    p.add_argument('--outfile', default='data/btc_features.csv')
    args = p.parse_args()
    df = pd.read_csv(args.infile, index_col='datetime', parse_dates=True)
    df_feat = add_features(df)
    df_feat.to_csv(args.outfile)
    print("Saved features to", args.outfile)
