# backtest.py
import pandas as pd
import joblib
import numpy as np
import argparse

FEATURES = ['ret_1','logret_1','ma_5','ma_15','ma_60','ma_5_15','vol_10','vol_60','rsi_14','bb_percent']

def improved_backtest(feature_csv='data/btc_features.csv', model_path='models/lgb_model.pkl',
                      initial_capital=10000, position_size=0.1, slippage=0.0003):

    df = pd.read_csv(feature_csv, index_col=0, parse_dates=True)
    model = joblib.load(model_path)

    X = df[FEATURES]
    preds = model.predict(X)

    df = df.iloc[-len(preds):].copy()
    df['pred_prob'] = preds
    df['pred'] = (preds > 0.5).astype(int)

    cash = initial_capital
    position_qty = 0
    entry_price = None

    equity_curve = []

    for i in range(len(df)-1):
        price = df['close'].iloc[i]
        next_price = df['close'].iloc[i+1]
        signal = df['pred'].iloc[i]

        # If we have a position, update equity with mark-to-market
        if position_qty > 0:
            unrealized_pnl = position_qty * (price - entry_price)
        else:
            unrealized_pnl = 0

        # ENTRY LOGIC
        if signal == 1 and position_qty == 0:
            trade_value = cash * position_size
            position_qty = trade_value / price
            entry_price = price * (1 + slippage)  # pay slippage
            cash -= trade_value

        # EXIT LOGIC
        elif signal == 0 and position_qty > 0:
            cash += position_qty * price * (1 - slippage)
            position_qty = 0
            entry_price = None

        # Mark-to-market equity
        equity = cash
        if position_qty > 0:
            equity += position_qty * price

        equity_curve.append({"datetime": df.index[i], "equity": equity})

    eq = pd.DataFrame(equity_curve).set_index("datetime")
    eq['returns'] = eq['equity'].pct_change().fillna(0)

    total_return = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
    sharpe = eq['returns'].mean() / (eq['returns'].std() + 1e-12) * np.sqrt(1440)  # 1440 min/day

    print("Total return:", total_return)
    print("Sharpe:", sharpe)

    eq.to_csv("results/equity_curve.csv")
    print("Saved improved equity curve to results/equity_curve.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_csv', default='data/btc_features.csv')
    parser.add_argument('--model_path', default='models/lgb_model.pkl')
    args = parser.parse_args()

    improved_backtest(args.feature_csv, args.model_path)
