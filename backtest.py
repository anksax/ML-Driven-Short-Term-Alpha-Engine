# backtest_v2.py
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import argparse
import os

FEATURES = ['ret_1','logret_1','ret_5','ret_15','ma_5','ma_15','ma_60','ma_5_15','vol_10','vol_60','atr_14','rsi_14','bb_percent']

def train_model_from_df(df_train):
    dtrain = lgb.Dataset(df_train[FEATURES], label=df_train['target'])
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
    }
    model = lgb.train(params, dtrain, num_boost_round=200)
    return model

def walk_forward_backtest(feature_csv='data/btc_features.csv', initial_capital=10000,
                          base_position_fraction=0.05,  # fraction of capital used as baseline
                          target_vol=0.002,   # target per-minute vol (tunable)
                          retrain_minutes=1440, # retrain daily (1440 minutes)
                          train_window_minutes=1440*7, # train on last 7 days
                          slippage=0.0003,
                          stop_loss_pct=0.003, # 0.3%
                          take_profit_pct=0.006 # 0.6%
                          ):
    df = pd.read_csv(feature_csv, index_col=0, parse_dates=True)
    n = len(df)
    df = df.sort_index().copy()

    # prepare columns
    df['pred_prob'] = np.nan
    df['pred'] = 0

    cash = initial_capital
    position = 0.0
    entry_price = None
    equity_curve = []

    # determine retrain points: start after first train window
    start_idx = train_window_minutes
    model = None

    for t in range(start_idx, n-1):
        # retrain when needed
        if (t - start_idx) % retrain_minutes == 0:
            i0 = max(0, t - train_window_minutes)
            df_train = df.iloc[i0:t].dropna()
            if len(df_train) < 200:
                # not enough data yet
                continue
            model = train_model_from_df(df_train)
            # (optional) save model snapshot
            # joblib.dump(model, f"models/lgb_snapshot_{t}.pkl")

        if model is None:
            continue

        row = df.iloc[t]
        price = row['close']
        # predicted probability
        x = row[FEATURES].values.reshape(1, -1)
        prob = model.predict(x)[0]
        df.at[df.index[t], 'pred_prob'] = prob

        # volatility estimate (use vol_60)
        vol = max(row['vol_60'], 1e-8)

        # dynamic threshold: if market quiet -> lower threshold (more trades), if volatile -> raise threshold
        # normalize relative to median vol in training window if available
        med_vol = df['vol_60'].iloc[max(0, t-train_window_minutes):t].median() or vol
        vol_ratio = vol / (med_vol + 1e-12)
        threshold = 0.5 + 0.08 * (vol_ratio - 1)  # can range roughly [0.42, 0.58] but we'll clip
        threshold = float(np.clip(threshold, 0.51, 0.75))

        signal = int(prob > threshold)
        df.at[df.index[t], 'pred'] = signal

        # compute desired position size: volatility scaling
        # as vol increases, reduce position; as vol decreases increase up to cap
        scaling = float(np.clip((target_vol / vol), 0.2, 3.0))
        position_fraction = base_position_fraction * scaling
        position_fraction = float(np.clip(position_fraction, 0.005, 0.25))

        # ENTRY
        if signal == 1 and position == 0:
            # allocate position_fraction of cash
            allocation = cash * position_fraction
            qty = allocation / (price * (1 + slippage))
            entry_price = price * (1 + slippage)
            position = qty
            cash -= qty * entry_price

        # If we have a position, check for stop-loss / take-profit on next bar(s)
        if position > 0:
            # check next bar price movement for exit conditions
            next_price = df['close'].iloc[t+1]
            # realized pnl if exit at next_price
            unreal_pnl = position * (next_price - entry_price)
            # stop-loss triggered?
            if next_price <= entry_price * (1 - stop_loss_pct):
                # exit at next_price minus slippage
                cash += position * next_price * (1 - slippage)
                position = 0
                entry_price = None
            # take-profit triggered?
            elif next_price >= entry_price * (1 + take_profit_pct):
                cash += position * next_price * (1 - slippage)
                position = 0
                entry_price = None
            # otherwise allow position to remain (mark-to-market used for equity curve)

        # compute equity (mark-to-market)
        equity = cash + (position * price if position > 0 else 0.0)
        equity_curve.append({'datetime': df.index[t], 'equity': equity, 'cash': cash, 'position': position})

    eq = pd.DataFrame(equity_curve).set_index('datetime')
    eq['returns'] = eq['equity'].pct_change().fillna(0)
    total_return = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
    sharpe = (eq['returns'].mean() / (eq['returns'].std() + 1e-12)) * np.sqrt(1440)

    print("Total return:", total_return)
    print("Sharpe:", sharpe)
    os.makedirs('results', exist_ok=True)
    eq.to_csv('results/equity_curve_v2.csv')
    print("Saved equity curve to results/equity_curve_v2.csv")
    # also save predictions for inspection
    df[['pred_prob','pred']].to_csv('results/predictions_v2.csv')
    print("Saved predictions to results/predictions_v2.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_csv', default='data/btc_features.csv')
    args = parser.parse_args()
    walk_forward_backtest(args.feature_csv)
