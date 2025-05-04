import logging
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from sklearn.preprocessing import StandardScaler
from cnn_lstm_model import CNN_LSTM
from data_prep import DataPrep, LoadingData, Engineering

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":

    # Step 1: Extract and concatenate the data
    # ==========================================
    # data_path = Path('data/nasdaq100/1min_taq/2017')
    # data_prep = DataPrep(data_path)
    # data_prep.extract_and_combine_data()
    #
    # # Step 2: Load and process the combined data
    # # ==========================================
    # data_file = data_path / 'algoseek_2017.pkl'
    # data_loader = LoadingData(data_file)
    # data_loader.load_data()
    # data_loader.date_column()
    # data_loader.save_data(data_path / '2017_data.pkl')
    # data_loader.show_data_info()

    # Step 3: Apply feature engineering on the loaded and processed data
    # ==================================================================
    # engineering = Engineering('data/nasdaq100/1min_taq/2016/2016_data.pkl')
    # engineering.feature_engineering()


    # Step 4: Train and test the model, and save the the predictions for backtesting purposes
    # =======================================================================================
    cnn_lstm = CNN_LSTM('2016_model_data.pkl')
    df = cnn_lstm.load_data()

    tickers = ['GOOGL', 'AAPL', 'TSLA', 'AMZN', 'INTC', 'NVDA']
    df = df[df.index.get_level_values('ticker').isin(tickers)]
    cols_to_keep = ['first', 'high', 'low', 'last', 'price', 'volume']
    df = df[cols_to_keep].copy()
    df['predicted'] = np.nan

    logging.info(f"Processing {len(tickers)} stock(s)...")

    for ticker in tickers:
        try:
            logging.info(f"Running pipeline for {ticker}...")
            y_pred, y_test, best_strategy = cnn_lstm.run_pipeline(stock_symbol=ticker)
            df_ticker = df.loc[ticker]
            df_ticker_index = df_ticker.tail(len(y_pred)).index
            df.loc[(ticker, df_ticker_index), 'predicted'] = y_pred
        except Exception as e:
            logging.warning(f"Failed for {ticker}: {e}")

    df = df[['predicted', 'first', 'high', 'low', 'last', 'price', 'volume']]
    df.to_pickle("backtest.pkl")
    print(df.info())
