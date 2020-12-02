import numpy as np
import pandas as pd
import streamlit as st
plt.style.use('seaborn-whitegrid')
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import date, datetime, timedelta
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.plotting import plot_covariance
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

  
def risk_estimator():

  portfolio_metadata = { 'FB': ['Facebook, Inc.', 1]
          , 'AMZN': ['Amazon.com', 1]
          , 'AAPL': ['Apple', 1]
          , 'NFLX': ['Netflix', 1]
          , 'GOOG': ['Alphabet Inc Class A',1]
              }

  securities = list(portfolio_metadata.keys())
  names = [item[0] for item in portfolio_metadata.values()]
  shares = [item[1] for item in portfolio_metadata.values()]

  # setting time frame for historical data
  START_DATE = '2018-01-01'
  # t-1
  END_DATE = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d") 

  n_sims = 10 ** 5

  percentiles = [0.01, 0.1, 1, 5, 10] 
  confidence_levels = [100 - value for value in percentiles]

  @st.cache
  def load_data(products, start, end):
      df = yf.download(" ".join(products), start=start, end=end)['Adj Close']
      return df

  df = load_data(securities, START_DATE, END_DATE)
  latest_prices = pd.DataFrame(df.columns, df.values[-1, :]).reset_index().rename(columns={'index': 'Price', 0: 'Ticker'})
  latest_prices['Price'] = ['$' + str('%.2f') % item for item in latest_prices['Price']]
  returns = df.pct_change().dropna()

  st.markdown("---")
  st.subheader("Initial portfolio information")

  portfolio_metadata_df = pd.DataFrame([securities, names, shares]).T
  portfolio_metadata_df.columns = ['Ticker', 'Name', 'Shares']
  portfolio_metadata_df = pd.merge(portfolio_metadata_df, latest_prices, on='Ticker')
  st.table(portfolio_metadata_df)

  st.markdown("---")
  st.subheader("Historical prices")

  df_temp = df.copy()
  df_temp['Date'] = df_temp.index
  df_plot = pd.melt(df_temp, id_vars='Date', value_vars=df_temp.columns[:-1]).rename(columns={'variable': 'Asset', 'value': 'Price ($)'})
  fig = px.line(df_plot, x='Date', y='Price ($)', color='Asset')
  st.plotly_chart(fig)

  st.markdown("---")
  st.subheader("Parameters")

  st.markdown(
          """ 

          With the parameters below, you can vary both confidence level and investment horizon and see how worst expected portfolio loss changes (and how much relative portfolio loss that entails). 
          
          Intuitively, as uncertainty grows with the investment horizon, the VaR metric does as well. The inverse holds true for confidence level - the higher the confidence level, the lower the VaR metric.

          """)

  T = st.number_input('Choose investment time frame (between 1 and 365 days)', min_value=1, max_value=365, step=1)

  confidence_level = st.selectbox(
      'Choose VaR (value-at-risk) confidence level'
      , (confidence_levels))

  if st.checkbox("All set, let's run the model!"):

    # calculations

    # calculate the covariance matrix
    cov_mat = returns.cov()

    # perform the Cholesky transformation of the covariance matrix
    chol_mat = np.linalg.cholesky(cov_mat)

    # draw the correlated random numbers from the Standard Normal distribution
    rv = np.random.normal(size=(n_sims, len(securities)))
    correlated_rv = np.transpose(np.matmul(chol_mat, np.transpose(rv)))

    # define the metrics that will be used for the simulations

    # individual security returns
    r = np.mean(returns, axis=0).values 
    # individual security dispersion
    sigma = np.std(returns, axis=0).values 

    # starting price of the securities
    S_0 = df.values[-1, :]
    # starting portfolio value
    P_0 = np.sum(shares * S_0)

    # calculate the terminal price of the considered securities
    S_T = S_0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * correlated_rv)

    # calculate the terminal portfolio metrics
    # portfolio value 
    P_T = np.sum(shares * S_T, axis=1) 
    # portfolio returns
    P_diff = P_T - P_0

    # calculate the VaR for the selected confidence levels
    P_diff_sorted = np.sort(P_diff) 
    var = np.percentile(P_diff_sorted, percentiles)

    var_index = confidence_levels.index(confidence_level)
    var_value = var[var_index]

    st.markdown("---") 

    st.subheader("VaR metrics")
    st.markdown(f"Initial portfolio value: ${P_0:.2f}")
    st.markdown(f"{T}-day VaR with {confidence_level}% confidence: ${var_value:.2f}")

    fig = px.histogram(P_diff)
    fig.update_traces(name='value', showlegend = True)
    fig.add_trace(
        go.Scatter(
            x = [var_value, var_value],
            y = [0, 1500],
            mode = "lines",
            line = go.scatter.Line(color = "red", width = 1),
            name = 'value-at-risk',
            showlegend = True
        )
    )

    fig.update_layout(title_text=f"Distribution of possible {T}-day changes in portfolio value", height=1000)
    st.plotly_chart(fig)

def portfolio_optimizer():

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
  
  st.markdown("---")
  st.subheader("ETF list")

  etfs_metadata_df = pd.DataFrame.from_dict(etfs_meta, orient='index').reset_index().rename(columns={"index": "Ticker", 0: "Short description"})
  st.table(etfs_metadata_df)

  st.markdown("---")
  st.subheader("Historical prices")

  df_temp = df.copy()
  df_temp['Date'] = df_temp.index
  df_plot = pd.melt(df_temp, id_vars='Date', value_vars=df_temp.columns[:-1]).rename(columns={'variable': 'Asset', 'value': 'Price ($)'})
  fig = px.line(df_plot, x='Date', y='Price ($)', color='Asset')
  st.plotly_chart(fig)


  st.subheader("Parameters")

  etfs_chosen = st.multiselect(
  'Which ETFs would you like to potentially add into your portfolio? (recommended to include all)',
  etfs_options, default=etfs_options)

  investment_amount = st.slider('Investment amount (between 1000 and 10000)', 1000, 10000, 1000)
  st.write('Your chosen amount is', '$' + str(investment_amount))


  if st.checkbox("All set, let's run the optimization model!"):

      df = df[etfs_chosen]

      # calculate expected returns
      mu = expected_returns.mean_historical_return(df)

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

      allocation_df['Allocation percentage'] = ((allocation_df['Amount'] / allocation_df['Amount'].sum())*100).round(2)
      allocation_df['Amount'] = ['$' + str(round(item,2)) for item in allocation_df['Amount']]
      allocation_df['Latest Price'] = ['$' + str(round(item,2)) for item in allocation_df['Latest Price']]
      
      allocation_df.reset_index(inplace=True, drop=True)

      st.table(allocation_df)

      title = "Allocation visualization (# of shares)"
      fig = px.bar(allocation_df, x='Ticker', y='Shares', width=600, height=400,title=title)
      st.plotly_chart(fig)

      title = "Allocation visualization (% invested)"
      fig = px.bar(allocation_df, x='Ticker', y='Allocation percentage', width=600, height=400,title=title)
      st.plotly_chart(fig)

      invested_amount = investment_amount - leftover
      st.markdown('Funds invested: ${:.2f}'.format(invested_amount))
      st.markdown('Funds remaining: ${:.2f}'.format(leftover))
