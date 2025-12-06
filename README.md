## Setup
1. Create virtualenv: `python -m venv venv && source venv/bin/activate`
2. Install: `pip install -r requirements.txt`

## Files
- data_fetch.py      # download historical OHLCV
- stream_data.py     # (optional) live candle streamer
- features.py        # feature engineering
- train_model.py     # train and save LightGBM model
- backtest.py        # quick backtest
- dashboard.py       # Streamlit UI

## Typical workflow
1. `python data_fetch.py --symbol BTC/USDT --since_days 7`
2. `python train_model.py --symbol BTC/USDT`
3. `python backtest.py --symbol BTC/USDT`
4. `streamlit run dashboard.py`

## Note
- Uses Binance REST for historical bars. For production/low-latency, use websocket orderbook streams and Parquet/Arrow storage.
