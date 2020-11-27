import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
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


def intro():

    st.sidebar.success("Select one of the options above.")

    st.markdown(
        """
        
        This is a collection of three data tools for you to explore modeling of risk in investing:

        - **Price Simulator** (Simulating asset prices with the Monte Carlo method)
        - **Risk Estimator** (Evaluating portfolio risk with the VaR metric)
        - **Portfolio optimizer** (Optimizing ETF portfolio with mean-variance analysis)

        **👈 Select the one you're interested in on the left**

      &nbsp;

      [Github repo with all the code](https://github.com/josolnik/compfin_app)
      
       Simulation code largely taken from [Python from Finance Cookbook](https://www.amazon.com/Python-Finance-Cookbook-libraries-financial/dp/1789618517)
    
    """
    )

def price_simulator():

    @st.cache
    def load_data(asset, start, end):
        df = yf.download(asset, start=start, end=end, adjusted=True)
        return df

    st.markdown("---")
    st.subheader("Parameters")

    st.markdown("""

    There are two inputs into the model:
    - Which asset's price to simulate
    - How many simulations to perform

    You can play with both and see how the output changes.

    Some of the questions we might ask:
    - Is the simulation model aligned with the simulated price levels? Or is too optimistic/pessimistic?
    - How does accuracy of the simulation change across different assets? Why is that?
    - How does the simulation results change when we change the number of simulations to run?

""")

    RISKY_ASSET = st.radio("Asset to simulate:",
    ('FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG'))
    N_SIM = st.number_input('Number of simulations to run (choose between 10 and 1000): ', min_value=10, max_value=1000)

    if st.checkbox("All set, let's run the model!"):

      START_DATE = '2020-01-01' 
      END_DATE = '2020-11-30'

      # data_load_state = st.text('Loading data...')
      data = load_data(RISKY_ASSET, START_DATE, END_DATE)
      # data_load_state.text("Data loaded!")

      adj_close = data['Adj Close'] 
      returns = adj_close.pct_change().dropna()

      train = returns['2020-01-01':'2020-10-31'] 
      test = returns['2020-11-01':'2020-11-30']

      T = len(test)
      N = len(test)
      S_0 = adj_close[train.index[-1].date()]
      mu = train.mean()
      sigma = train.std()

      @st.cache
      def simulate_gbm(s_0, mu, sigma, n_sims, T, N):

        dt = T/N
        dW = np.random.normal(scale = np.sqrt(dt), size=(n_sims, N))
        W = np.cumsum(dW, axis=1)

        time_step = np.linspace(dt, T, N) 
        time_steps = np.broadcast_to(time_step, (n_sims, N))

        S_t = s_0 * np.exp((mu - 0.5 * sigma ** 2) * time_steps + sigma * W) 
        S_t = np.insert(S_t, 0, s_0, axis=1) 
        
        return S_t

      gbm_simulations = simulate_gbm(S_0, mu, sigma, N_SIM, T, N)

      # prepare objects for plotting
      LAST_TRAIN_DATE = train.index[-1].date()
      FIRST_TEST_DATE = test.index.min().date()
      LAST_TEST_DATE = test.index.max().date()
      PLOT_TITLE = (f'{RISKY_ASSET} Simulation ' f'(from {FIRST_TEST_DATE} till {LAST_TEST_DATE})')

      selected_indices = adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE].index
      index = [date.date() for date in selected_indices]

      gbm_simulations_df = pd.DataFrame(np.transpose(gbm_simulations), index=index)

      fig = px.line(gbm_simulations_df, x=index, y=gbm_simulations_df.mean(axis=1), title=PLOT_TITLE, labels={'x': 'Date', 'y': 'Adj. Close Price USD ($)'})
      fig.update_traces(name='Average value', showlegend = True)

      fig.add_scatter(x=index, y=adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE], mode='lines', name='Realized value') #  showlegend=False)

      for sim_num in gbm_simulations_df.columns:
        fig.add_scatter(x=index, y=gbm_simulations_df[sim_num], mode='lines', opacity=0.05, showlegend=False)

      st.plotly_chart(fig)

      if st.checkbox('Show sample raw data'):
          st.subheader('Raw data')
          st.write(data.head(50))

      if st.checkbox('Show historical returns'):
        chart_data = pd.DataFrame(
            returns.values,
            returns.index.values)
        st.line_chart(chart_data)
        st.write(f'Average return: {100 * returns.mean():.2f}%')
  
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
  start_date = '2018-01-01'
  # t-1
  end_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d") 

  n_sims = 10 ** 5

  percentiles = [0.01, 0.1, 1, 5, 10] 
  confidence_levels = [100 - value for value in percentiles]

  @st.cache
  def load_data(products, start, end):
      df = yf.download(" ".join(products), start=start, end=end)['Adj Close']
      return df

  df = load_data(securities, start_date, end_date)
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
    var_portfolio_return = ((var_value + P_0)/ P_0)*100

    st.markdown("---")

    st.subheader("VaR metrics")
    st.markdown(f"Initial portfolio value: ${P_0:.2f}")
    st.markdown(f"{T}-day VaR with {confidence_level}% confidence: ${var_value:.2f}")
    st.markdown(f"VaR portfolio gain/loss: {var_portfolio_return :.2f}%")

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
