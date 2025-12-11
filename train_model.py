# train_model_v2.py
import pandas as pd
import lightgbm as lgb
import joblib
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score

FEATURES = ['ret_1','logret_1','ret_5','ret_15','ma_5','ma_15','ma_60','ma_5_15','vol_10','vol_60','atr_14','rsi_14','bb_percent']

def train(infile='data/btc_features.csv', model_out='models/lgb_model_v2.pkl'):
    df = pd.read_csv(infile, index_col=0, parse_dates=True)
    X = df[FEATURES]
    y = df['target']
    split = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    dtrain = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
    }
    bst = lgb.train(params, dtrain, num_boost_round=300)
    preds = bst.predict(X_test)
    preds_label = (preds > 0.5).astype(int)
    print("Accuracy:", accuracy_score(y_test, preds_label))
    print("AUC:", roc_auc_score(y_test, preds))
    joblib.dump(bst, model_out)
    print("Saved model to", model_out)

if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', default='data/btc_features.csv')
    parser.add_argument('--model_out', default='models/lgb_model_v2.pkl')
    args = parser.parse_args()
    os.makedirs('models', exist_ok=True)
    train(args.infile, args.model_out)
