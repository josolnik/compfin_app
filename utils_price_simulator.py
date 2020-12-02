import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import date, datetime, timedelta


def main():

    @st.cache
    def load_data(asset, start, end):
        df = yf.download(asset, start=start, end=end, adjusted=True)['Adj Close'] 
        return df

    START_DATE = '2020-01-01' 
    END_DATE = '2020-11-30'

    securities = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']

    df = load_data(securities, START_DATE, END_DATE)
    returns = df.pct_change().dropna()

    st.markdown("---")
    st.subheader("Historical prices")

    df_temp = df.copy()
    df_temp['Date'] = df_temp.index
    df_plot = pd.melt(df_temp, id_vars='Date', value_vars=df_temp.columns[:-1]).rename(columns={'variable': 'Asset', 'value': 'Price ($)'})
    fig = px.line(df_plot, x='Date', y='Price ($)', color='Asset')
    st.plotly_chart(fig)

    st.markdown("---")
    st.subheader("Parameters")

    st.markdown("""

    There are two parameters to set in the model:
    - Which asset's price to simulate
    - How many simulations to perform

    You can play with both and see how the output changes.

    Some of the questions we might ask:
    - Is the simulation model aligned with the simulated price levels? Or is it too optimistic/pessimistic?
    - How does accuracy of the simulation change across different assets? Why is that?
    - How does the simulation results change when we change the number of simulations to run?

    """)

    RISKY_ASSET = st.radio("Asset price to simulate:",
    (securities))
    N_SIM = st.number_input('Number of simulations to run (choose between 10 and 1000): ', min_value=10, max_value=1000)

    if st.checkbox("All set, let's run the model!"):

      train = returns[START_DATE:'2020-10-31'] 
      test = returns['2020-11-01':END_DATE]

      T = len(test)
      N = len(test)
      S_0 = df[train.index[-1].date()]
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

      selected_indices = df[LAST_TRAIN_DATE:LAST_TEST_DATE].index
      index = [date.date() for date in selected_indices]

      gbm_simulations_df = pd.DataFrame(np.transpose(gbm_simulations), index=index)

      fig = px.line(gbm_simulations_df, x=index, y=gbm_simulations_df.mean(axis=1), title=PLOT_TITLE, labels={'x': 'Date', 'y': 'Adj. Close Price USD ($)'})
      fig.update_traces(name='Average simulated value', showlegend = True)

      fig.add_scatter(x=index, y=df[LAST_TRAIN_DATE:LAST_TEST_DATE], mode='lines', name='Realized value')

      for sim_num in gbm_simulations_df.columns:
        fig.add_scatter(x=index, y=gbm_simulations_df[sim_num], mode='lines', opacity=0.05, showlegend=False)

      st.plotly_chart(fig)

      if st.checkbox('Show sample raw data'):
          st.subheader('Raw data')
          st.write(df.head(50))

      if st.checkbox('Show historical returns'):
        chart_data = pd.DataFrame(
            returns.values,
            returns.index.values)
        st.line_chart(chart_data)
        st.write(f'Average return: {100 * returns.mean():.2f}%')