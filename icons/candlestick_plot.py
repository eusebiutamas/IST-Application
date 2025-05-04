import pickle
import pandas as pd
from plotly import graph_objects as go

pickle_path = '2016_model_data.pkl'
with open(pickle_path, 'rb') as f:
    df = pickle.load(f)

ohlcv = ['first', 'high', 'low', 'last', 'price', 'volume']
data_cols = ohlcv + ['rup', 'rdown', 'atask', 'atbid']
idx = pd.IndexSlice

df = df.loc[idx['TSLA', '2016-09-29'], :].reset_index()
df.fillna(0, inplace=True)

fig = go.Figure(data=[go.Candlestick(x=df['date_time'], open=df['first'], high=df['high'],
                                     low=df['low'], close=df['last'])])
fig.update_layout(title='TSLA Candlestick Chart - 2016-09-29', width=1000, height=500,
                  xaxis=dict(type='date', tickformat='%H:%M', tickmode='array',
                             tickvals=pd.date_range(start=df['date_time'].min(),
                                                    end=df['date_time'].max(),
                                                    freq='30min'), tickangle=45),
                  yaxis_title='Price (USD)')
fig.update_xaxes(rangeslider_visible=True)
fig.show(renderer="browser")
