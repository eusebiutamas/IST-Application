import pickle
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

sns.set_style('whitegrid')
np.random.seed(42)

warnings.filterwarnings('ignore')

# This will load the dataset earlier saved for backtesting purposes
with open('backtest.pkl', 'rb') as f:
    data = pickle.load(f)

# This function is selecting only the last 30% of the entire dataset
def last_30_percent(group):
    n = int(len(group) * 0.3)
    return group.tail(n)

subset = data.groupby('ticker', group_keys=False).apply(last_30_percent)

# This will create the benchmark (portfolio) containing the six stock from below list
benchmark_tickers = ['GOOGL', 'AAPL', 'TSLA', 'AMZN', 'INTC', 'NVDA']
benchmark_prices = subset['last'][subset.index.get_level_values('ticker').isin(benchmark_tickers)]
benchmark = benchmark_prices.unstack('ticker').pct_change()
benchmark.name = 'Benchmark'

# This will compute forward returns based on the first traded bar since the loaded data will be simulated
# and also create a new column to the dataset as 'predicted' using the predictions
prices = data['first'].unstack('ticker').sort_index()
returns = prices.pct_change()
fwd_returns = returns.shift(-1)
predictions = data['predicted'].unstack('ticker')

# This will help generate Buy and Sell signals based on the top 30 positive and negative predictions
# and the computed forward returns
N_LONG = N_SHORT = 30
long_signals = ((predictions.where(predictions > 0).rank(axis=1, ascending=False) <= N_LONG).astype(int))
short_signals = ((predictions.where(predictions < 0).rank(axis=1) <= N_SHORT).astype(int))

long_returns = long_signals.mul(fwd_returns).mean(axis=1)
short_returns = short_signals.mul(-fwd_returns).mean(axis=1)
strategy = long_returns.add(short_returns).to_frame('Strategy')

# This will create the vectorized strategy to backtest it against the benchmark
strategy = strategy.loc[strategy.index.intersection(benchmark.index)]
benchmark = benchmark.loc[strategy.index]

# This allows to specify a certain date by knowing the test set dates
# and also generate plots of the cummulative returns of the strategy against the benchmark (portfolio)
test_day = ['2016-09-30']
test_day = pd.to_datetime(test_day)
strategy = strategy[strategy.index.normalize().isin(test_day)]
benchmark = benchmark[benchmark.index.normalize().isin(test_day)]

fig, ax = plt.subplots(figsize=(12, 6))
strategy.join(benchmark).add(1).cumprod().sub(1).plot(ax=ax, title='Cumulative Return on 2016-09-30')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
sns.despine()
fig.tight_layout()
plt.show()

# This sets the initial capical of $100,000 and calculate the metrics (Net profit and its percentage)
initial_capital = 100000
strategy['Portfolio'] = initial_capital * (1 + strategy['Strategy']).cumprod()
benchmark_cum = (1 + benchmark.mean(axis=1)).cumprod()
benchmark_value = initial_capital * benchmark_cum

final_value = strategy['Portfolio'].iloc[-1]
profit = final_value - initial_capital
profit_percent = (profit / initial_capital) * 100

print(f"Portfolio net profit on 2016-09-30: ${profit:,.2f}")
print(f"Portfolio net profit percentage on 2016-09-30: {profit_percent:.2f}%")

strategy_mean = strategy['Strategy'].mean()
strategy_std = strategy['Strategy'].std()
benchmark_mean = benchmark.mean().mean()
benchmark_std = benchmark.std().mean()

# This will set the risk free simply to 0 and calculate the Sharpe Ratio metric
risk_free_rate = 0
sharpe_ratio_strategy = (strategy_mean - risk_free_rate) / strategy_std
sharpe_ratio_benchmark = (benchmark_mean - risk_free_rate) / benchmark_std

print(f"Sharpe Ratio for Strategy on 2016-09-30: {sharpe_ratio_strategy:.2f}")
