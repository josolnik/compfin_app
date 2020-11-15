# Yahoo Finance ETF URL
# https://finance.yahoo.com/etfs

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')
import plotly.express as px
import yfinance as yf
from datetime import date, datetime, timedelta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.plotting import plot_covariance
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


etfs_meta = { 'SPY': 'SPDR S&P 500 ETF Trust'
       , 'XLF': 'Financial Select Sector SPDR Fund'
       , 'QQQ': 'Invesco QQQ Trust' 
       , 'XLE': 'Energy Select Sector SPDR Fund'
       , 'IAU': 'iShares Gold Trust'
       , 'KRE': 'SPDR S&P Regional Banking ETF'
       , 'XLI': 'Industrial Select Sector SPDR Fund'
       , 'IYR': 'iShares U.S. Real Estate ETF'
       , 'IEFA': 'iShares Core MSCI EAFE ETF'
       , 'XLP': 'Consumer Staples Select Sector SPDR Fund'}


etfs_options = list(etfs_meta.keys())

start_date = "2015-01-01"
# t-1
yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d") 
end_date = yesterday

@st.cache
def load_data(etfs, start, end):
    df = yf.download(" ".join(etfs), start=start, end=end)['Adj Close']
    return df

df = load_data(etfs_options, start_date, end_date)
returns = df.pct_change().dropna()

st.title('Portfolio Optimizer')

st.markdown(
        """
        
    We would like to optimize our portfolio's return while taking into account risk of individual assets. One of the best ways for that is to invest our funds into a set of [ETFs (exchange traded funds)](https://www.investopedia.com/terms/e/etf.asp) . ETFs are affordable and well established financial products
        used to invest in a diversified basket of assets.
        
    To go even further, we can optimize* our portfolio to invest across multiple ETFs.
    Below you can find a selection of [top 10 most traded ETFs](https://finance.yahoo.com/etfs/) (as of October 10th 2020) with a short 
    description. Besides that, you can find historical data of how their prices moved in the last 5 years.

    You define two inputs:

    - ETFs to potentially add into your portfolio (we determine which are included in the optimized portfolio with the model)
    - Investment amount (as ETFs are traded in discrete amounts you might have some residual amount)
    
    *To optimize performance we use MVO ([mean-variance optimization](https://en.wikipedia.org/wiki/Modern_portfolio_theory)) to maximize the portfolio's Sharpe ratio (risk-adjusted return). We show discrete quantities of how to distribute
    the investment amount.
    
    """
    )

st.markdown("---")
st.subheader("ETF list")

# Show ETF metadata
etfs_metadata_df = pd.DataFrame.from_dict(etfs_meta, orient='index').reset_index().rename(columns={"index": "Ticker", 0: "Short description"})
st.table(etfs_metadata_df)

st.markdown("---")
st.subheader("Historical prices")

# Visualize historical prices
# title = 'Historical Adj. Close Price of available ETFs'

df_temp = df.copy()
df_temp['Date'] = df_temp.index
df_plot = pd.melt(df_temp, id_vars='Date', value_vars=df_temp.columns[:-1]).rename(columns={'variable': 'Asset', 'value': 'Price ($)'})
fig = px.line(df_plot, x='Date', y='Price ($)', color='Asset')
st.plotly_chart(fig)


st.subheader("Parameters")

etfs_chosen = st.multiselect(
'Which ETFs would you like to potentially add into your portfolio? (recommended to include all)',
etfs_options, default=etfs_options)

investment_amount = st.number_input('Investment amount (between 1000 and 10000)', min_value=1000, max_value=10000)
st.write('Your chosen amount is ', investment_amount)

# returns['date'] = returns.index
# df_plot = pd.melt(returns, id_vars='date', value_vars=df.columns[:-1])
# df_plot.head()
# fig = px.line(df_plot, x='date', y='value', color='variable')
# st.show(fig)

# fig = px.line(df, x=returns.index, y=returns[returns.columns[0]], title=title, labels={'x': 'Date', 'y': 'Adj. Close Price USD ($)'})

# for asset_col in df.columns[1:]:
#     px.line(df, x=returns.index, y=returns[asset_col])
    # fig.add_scatter(x=returns.index, y=returns[asset_col], mode='lines')

if st.checkbox("All set, let's run the optimization model!"):

    df = df[etfs_chosen]

    # calculate expected return with CAPM
    # slightly more stable than the default mean historical return
    mu = expected_returns.mean_historical_return(df)
    # .capm_return(df)

    mu_df = pd.DataFrame(mu).reset_index().rename(columns={"index": "Ticker", 0: "Expected Return (%)"}).sort_values(by="Expected Return (%)", ascending=False)
    mu_df['Expected Return (%)'] = round(mu_df['Expected Return (%)']*100,2)

    st.subheader("Expected returns")
    st.markdown("Showing returns that we could expect when taking into account historical data.")
    fig = px.bar(mu_df, x='Ticker', y='Expected Return (%)', width=300, height=200)
    st.plotly_chart(fig)

    # calculate estimated covariance matrix (risk model) using Ledoit-Wolf shrinkage
    # reduces the extreme values in the covariance matrix 
    S = risk_models.CovarianceShrinkage(df).ledoit_wolf()

    st.subheader("Covariance matrix")
    st.markdown("Showing relationship in price movement between different ETFs.")
    sns.heatmap(S.corr())
    st.pyplot()

    # Optimize the portfolio performance
    # Sharpe ratio: portfolio's return less risk-free rate, per unit of risk (volatility)

    ef = EfficientFrontier(mu, S, weight_bounds=(0.01, 1))

    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    portfolio_performance = ef.portfolio_performance()
    st.markdown("---")
    st.subheader("Portfolio performance")
    st.markdown('Summary metrics:')
    st.markdown('Expected annual return: {:.2f}%'.format(portfolio_performance[0]*100))
    st.markdown('Annual volatility: {:.2f}%'.format(portfolio_performance[1]*100))
    st.markdown('Sharpe Ratio: {:.2f}'.format(portfolio_performance[2]))

    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=investment_amount)

    latest_prices_column = pd.DataFrame(latest_prices).columns[0]
    latest_prices_df = pd.DataFrame(latest_prices).reset_index().rename(columns={"index": "Ticker", latest_prices_column: "Latest Price"}).sort_values(by='Ticker')

    allocation, leftover = da.lp_portfolio()

    st.subheader("Portfolio allocation")

    allocation_df = pd.DataFrame.from_dict(allocation, orient='index').reset_index().rename(columns={"index": "Ticker", 0: "Shares"}).sort_values(by='Ticker')
    allocation_df = pd.merge(allocation_df, latest_prices_df, on='Ticker', how='left')
    allocation_df['Amount'] = allocation_df['Shares'] * allocation_df['Latest Price']
    allocation_df.sort_values(by='Amount', inplace=True, ascending=False)

    allocation_df['Amount'] = ['$' + str(round(item,2)) for item in allocation_df['Amount']]
    allocation_df['Latest Price'] = ['$' + str(round(item,2)) for item in allocation_df['Latest Price']]
    
    allocation_df.reset_index(inplace=True, drop=True)

    st.table(allocation_df)

    title = "Allocation visualization (# of shares)"
    fig = px.bar(allocation_df, x='Ticker', y='Shares', width=600, height=400,title=title)
    st.plotly_chart(fig)

    title = "Allocation visualization ($ invested)"
    fig = px.bar(allocation_df, x='Ticker', y='Amount', width=600, height=400,title=title)
    st.plotly_chart(fig)

    invested_amount = investment_amount - leftover
    st.markdown('Funds invested: ${:.2f}'.format(invested_amount))
    st.markdown('Funds remaining: ${:.2f}'.format(leftover))