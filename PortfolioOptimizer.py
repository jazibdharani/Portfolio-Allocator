import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def symbol_to_path(symbol, base_dir="StockData"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()


def daily_returns(df):
    d_returns = (df / df.shift(1)) - 1
    d_returns.ix[0, :] = 0
    return d_returns


def sharpe_ratio_fn(allocation, df):

    allocations = [allocation[0], 1-allocation[0]] # Sets the restriction (faster)

    # Calculate portfolio returns and volatility
    returns = (np.sum(df.mean() * allocations) * 252)
    risk = (np.sqrt(np.dot(np.asarray(allocations).T, np.dot(df.cov() * 252, allocations))))

    # Calculate sharpe ratio
    ''' Assuming Risk Free Rate is Zero
    '''
    sharpe = returns / risk

    return -1 * sharpe

################################################

def portfolio_optimizer(sharpe_ratio_fn):

# Read Data
    dates = pd.date_range('2017-01-01', '2017-01-31')
    symbols = ['GOOGL', 'AAPL']
    df = get_data(symbols, dates)
    plot_data(df)

#Scatterplot of daily returns -> comparing two stocks
    daily_return = daily_returns(df)
    daily_returns.plot(kind='scatter', x = 'GOOGL', y='AAPL')
    betaAAPL, alphaAPPL = np.polyfit(daily_returns['GOOGL'], daily_returns['APPL'], 1)
    print ("betaAAPL=", betaAPPL)
    print ("alphaAAPL=", alphaAPPL)
    plt.plot(daily_returns['GOGGL'], betaAAPL*daily_returns['GOOGL'] + alphaAAPL, '-',color='red')
    plt.show()

#Correlation Coefficient
    print (daily_returns.corr(method='pearson'))


# Minimizer -> Gradient Decent Algo
    Guess = [0.5]

    min_result = spo.minimize(sharpe_ratio_fn, Guess, method='SLSQP', bounds=[(0,1)],
                             args=(daily_returns,), options={'disp':True})

    optimal_allocations = [min_result.x, 1 - min_result.x]
    print (optimal_allocations)
    print(-sharpe_ratio_fn(min_result.x,daily_returns))


def main():
    portfolio_optimizer(sharpe_ratio_fn)

if __name__ == "__main__":
    main()
