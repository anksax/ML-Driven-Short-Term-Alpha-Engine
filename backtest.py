# backtest.py
import pandas as pd
import joblib
import numpy as np
import argparse
from sklearn.metrics import accuracy_score

FEATURES = ['ret_1','logret_1','ma_5','ma_15','ma_60','ma_5_15','vol_10','vol_60','rsi_14','bb_percent']

def simple_backtest(feature_csv='data/btc_features.csv', model_path='models/lgb_model.pkl', initial_capital=10000, position_size=0.1, slippage=0.0005):
    df = pd.read_csv(feature_csv, index_col=0, parse_dates=True)
    model = joblib.load(model_path)
    X = df[FEATURES]
    preds = model.predict(X)
    df = df.iloc[len(df) - len(preds):].copy()
    df['pred_prob'] = preds
    df['pred'] = (df['pred_prob'] > 0.5).astype(int)
    # simulate: every minute we take a long if pred==1, else flat
    capital = initial_capital
    cash = capital
    pos = 0.0
    equity_curve = []
    for idx, row in df.iterrows():
        price = row['close']
        signal = row['pred']
        # close previous position at current price
        if pos != 0:
            pnl = pos * (price - prev_price) - abs(pos) * slippage * price
            cash += pnl
            pos = 0
        # open new pos if signal==1
        if signal == 1:
            trade_value = cash * position_size
            qty = trade_value / price
            # pay slippage round-trip approx
            cash -= trade_value
            pos = qty
            prev_price = price
        equity = cash + pos * price
        equity_curve.append({'datetime': idx, 'equity': equity, 'signal': signal})
    eq = pd.DataFrame(equity_curve).set_index('datetime')
    eq['returns'] = eq['equity'].pct_change().fillna(0)
    total_return = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
    sharpe = (eq['returns'].mean() / (eq['returns'].std() + 1e-12)) * np.sqrt(252 * 24 * 60)  # minute freq approx
    print("Total return:", total_return)
    print("Sharpe (est):", sharpe)
    eq.to_csv('results/equity_curve.csv')
    print("Saved equity curve to results/equity_curve.csv")

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_csv', default='data/btc_features.csv')
    parser.add_argument('--model_path', default='models/lgb_model.pkl')
    args = parser.parse_args()
    os.makedirs('results', exist_ok=True)
    simple_backtest(args.feature_csv, args.model_path)
