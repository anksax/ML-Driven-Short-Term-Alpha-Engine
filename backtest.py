import pandas as pd
import numpy as np
import lightgbm as lgb
import argparse
import os

# -------------------------------------------
# FEATURES USED
# -------------------------------------------
FEATURES = [
    'ret_1','logret_1','ret_5','ret_15',
    'ma_5','ma_15','ma_60','ma_5_15',
    'vol_10','vol_60','atr_14','rsi_14','bb_percent'
]

# -------------------------------------------
# TRAIN MODEL ON A SLIDING WINDOW
# -------------------------------------------
def train_model_from_df(df_train):
    dtrain = lgb.Dataset(df_train[FEATURES], label=df_train['target'])
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'verbosity': -1
    }
    model = lgb.train(params, dtrain, num_boost_round=150)
    return model


# -------------------------------------------
# WALK-FORWARD BACKTEST CONFIGURED FOR ~10k ROWS
# -------------------------------------------
def walk_forward_backtest(
    feature_csv='data/btc_features.csv',
    initial_capital=10_000,
    train_window=500,       # ← Fits your 10k-row dataset (train on last ~1000 minutes)
    retrain_interval=100,    # retrain roughly every 200 minutes
    target_vol=0.002,
    base_position_fraction=0.05,
    slippage=0.0003,
    stop_loss_pct=0.003,
    take_profit_pct=0.006
):
    df = pd.read_csv(feature_csv, index_col=0, parse_dates=True)
    df = df.sort_index().copy()

    n = len(df)
    if n < train_window + 10:
        raise ValueError("Dataset too small for walk-forward test.")

    df['pred_prob'] = np.nan
    df['pred'] = 0

    cash = initial_capital
    position = 0
    entry_price = None
    equity_curve = []

    # Start walk-forward after we have enough data to train
    start_index = train_window
    model = None

    for t in range(start_index, n - 1):
        # -------------------------------------------
        # RETRAIN MODEL IF NECESSARY
        # -------------------------------------------
        if (t - start_index) % retrain_interval == 0:
            df_train = df.iloc[t - train_window : t].dropna()
            if len(df_train) > 200:
                model = train_model_from_df(df_train)
            else:
                continue

        if model is None:
            continue
        
        row = df.iloc[t]
        price = row['close']

        # -------------------------------------------
        # PREDICT SIGNAL
        # -------------------------------------------
        X = row[FEATURES].values.reshape(1, -1)
        prob = float(model.predict(X)[0])
        df.at[df.index[t], 'pred_prob'] = prob

        # Dynamic threshold: adjust based on recent volatility ratio
        vol = max(row['vol_60'], 1e-8)
        median_vol = df['vol_60'].iloc[max(0, t - train_window):t].median()
        vol_ratio = vol / (median_vol + 1e-8)

        threshold = 0.5 + 0.05 * (vol_ratio - 1)
        threshold = float(np.clip(threshold, 0.52, 0.70))

        signal = int(prob > threshold)
        df.at[df.index[t], 'pred'] = signal

        # -------------------------------------------
        # POSITION SIZING (VOLATILITY SCALED)
        # -------------------------------------------
        vol_scale = float(np.clip((target_vol / vol), 0.3, 3.0))
        pos_fraction = base_position_fraction * vol_scale
        pos_fraction = float(np.clip(pos_fraction, 0.01, 0.20))

        # -------------------------------------------
        # ENTRY
        # -------------------------------------------
        if signal == 1 and position == 0:
            allocation = cash * pos_fraction
            qty = allocation / (price * (1 + slippage))
            position = qty
            entry_price = price * (1 + slippage)
            cash -= qty * entry_price

        # -------------------------------------------
        # EXIT CONDITIONS (STOP-LOSS / TAKE-PROFIT)
        # -------------------------------------------
        if position > 0:
            next_price = df['close'].iloc[t + 1]

            if next_price <= entry_price * (1 - stop_loss_pct):
                cash += position * next_price * (1 - slippage)
                position = 0
                entry_price = None

            elif next_price >= entry_price * (1 + take_profit_pct):
                cash += position * next_price * (1 - slippage)
                position = 0
                entry_price = None

        # -------------------------------------------
        # MARK-TO-MARKET EQUITY
        # -------------------------------------------
        equity = cash + (position * price if position > 0 else 0)
        equity_curve.append({
            'datetime': df.index[t],
            'equity': equity,
            'cash': cash,
            'position': position
        })

    # -------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------
    eq = pd.DataFrame(equity_curve).set_index('datetime')
    eq['returns'] = eq['equity'].pct_change().fillna(0)

    total_ret = eq['equity'].iloc[-1] / eq['equity'].iloc[0] - 1
    sharpe = eq['returns'].mean() / (eq['returns'].std() + 1e-12) * np.sqrt(1440)

    print("Total Return:", total_ret)
    print("Sharpe:", sharpe)

    os.makedirs('results', exist_ok=True)
    eq.to_csv('results/equity_curve.csv')
    df[['pred_prob','pred']].to_csv('results/predictions.csv')

    print("Saved equity curve to results/equity_curve.csv")
    print("Saved predictions to results/predictions.csv")


# -------------------------------------------
# CLI
# -------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_csv", default="data/btc_features.csv")
    args = parser.parse_args()
    walk_forward_backtest(args.feature_csv)
