import numpy as np
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


def intro():

    st.sidebar.success("Select one of the options above.")

    st.markdown(
        """
        
        This is a collection of three data apps for you to explore the interaction between risk and market dynamics:

        - Price Simulator (Simulation of asset prices with the Monte Carlo method)
        - Portfolio optimizer (Optimizing ETF portfolio with mean-variance analysis)

        **👈 Select the one you're interested in on the left**
    
    """
    )

def price_simulator():

  if st.checkbox("Got it, let's run the model!"):

    @st.cache
    def load_data(asset, start, end):
        df = yf.download(asset, start=start, end=end, adjusted=True)
        return df

    RISKY_ASSET = st.radio("Asset to simulate:",
    ('FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG'))
    N_SIM = st.number_input('Number of simulations to run (choose between 10 and 1000): ', min_value=10, max_value=1000)

    START_DATE = '2020-01-01' 
    END_DATE = '2020-09-30'

    # data_load_state = st.text('Loading data...')
    data = load_data(RISKY_ASSET, START_DATE, END_DATE)
    # data_load_state.text("Data loaded!")

    adj_close = data['Adj Close'] 
    returns = adj_close.pct_change().dropna()

    train = returns['2020-01-01':'2020-08-31'] 
    test = returns['2020-09-01':'2020-10-01']

    T = len(test)
    N = len(test)
    S_0 = adj_close[train.index[-1].date()]
    # N_SIM = 100
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

    fig = px.line(gbm_simulations_df, x=index, y=gbm_simulations_df.mean(axis=1), title=PLOT_TITLE, labels={'x': 'Date', 'y': 'Adj. Close Price USD ($)'}) #, hover_name='Mean')

    fig.add_scatter(x=index, y=adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE], mode='lines', name='Realized value') #  showlegend=False)

    for sim_num in gbm_simulations_df.columns:
      fig.add_scatter(x=index, y=gbm_simulations_df[sim_num], mode='lines', opacity=0.1, showlegend=False)

    st.plotly_chart(fig)

    # fig, ax = plt.subplots(figsize=(20,10))

    # line_1 = ax.plot(index, gbm_simulations_df.mean(axis=1), color='red')
    # line_2 = ax.plot(gbm_simulations_df.index, adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE], color='blue')

    # for sim_num in gbm_simulations_df.columns:
    #   ax.plot(index, gbm_simulations_df[sim_num], label=sim_num, alpha=0.1, color = 'gray')

    # st.subheader(PLOT_TITLE)
    # ax.set_xlabel('Date', fontsize=18)
    # ax.set_ylabel('Adj. Price USD ($)', fontsize=18)
    # ax.legend((line_1, line_2), ('mean', 'actual'))
    # st.show(fig)

    if st.checkbox('Show sample raw data'):
        st.subheader('Raw data')
        st.write(data.head(50))

    if st.checkbox('Show historical returns'):
      chart_data = pd.DataFrame(
          returns.values,
          returns.index.values)
      st.line_chart(chart_data)
      st.write(f'Average return: {100 * returns.mean():.2f}%')

    # st.show(fig)
  

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
