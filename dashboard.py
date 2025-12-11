# dashboard.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Short-term ML Predictor Dashboard")

model_path = st.sidebar.text_input("Model path", "models/lgb_model.pkl")
eq_path = st.sidebar.text_input("Equity CSV", "results/equity_curve.csv")
feat_path = st.sidebar.text_input("Feature CSV", "data/btc_features.csv")

if st.sidebar.button("Load model & data"):
    model = joblib.load(model_path)
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    X = df[['ret_1','logret_1','ma_5','ma_15','ma_60','ma_5_15','vol_10','vol_60','rsi_14','bb_percent']]
    preds = model.predict(X)
    df['pred_prob'] = preds
    st.write("Sample predictions")
    st.dataframe(df[['close','pred_prob']].tail(50))
    if st.sidebar.checkbox("Show feature importances"):
        importances = model.feature_importance()
        feat_names = X.columns.tolist()
        imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False)
        st.bar_chart(imp_df.set_index('feature'))
try:
    eq = pd.read_csv('results/equity_curve_v2.csv', index_col=0, parse_dates=True)
    st.line_chart(eq['equity'])
except FileNotFoundError:
    st.warning("Run backtest_v2.py to generate V2 equity curve.")
