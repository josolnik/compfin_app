import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import date, datetime, timedelta


def main():

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

  N_SIMS = 10 ** 5
  PERCENTILES = [0.01, 0.1, 1, 5, 10] 
  CONFIDENCE_LEVELS = [100 - value for value in PERCENTILES]

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

          With the parameters below, you can vary both confidence level and investment horizon and see how worst expected portfolio loss changes.

          """)

  T = st.number_input('Choose investment time frame (between 1 and 365 days)', min_value=1, max_value=365, step=1)

  confidence_level = st.selectbox(
      'Choose VaR (value-at-risk) confidence level'
      , (CONFIDENCE_LEVELS))

  if st.checkbox("All set, let's run the model!"):

    # calculations

    # calculate the covariance matrix
    cov_mat = returns.cov()

    # perform the Cholesky transformation of the covariance matrix
    chol_mat = np.linalg.cholesky(cov_mat)

    # draw the correlated random numbers from the Standard Normal distribution
    rv = np.random.normal(size=(N_SIMS, len(securities)))
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
    var = np.percentile(P_diff_sorted, PERCENTILES)

    var_index = CONFIDENCE_LEVELS.index(confidence_level)
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