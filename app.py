# Copyright 2018-2020 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import textwrap
from collections import OrderedDict

import streamlit as st
from streamlit.logger import get_logger
import utils

LOGGER = get_logger(__name__)

# Dictionary of
# app_name -> (app_function, app_description)
APPS = OrderedDict(
    [
        ("—", (utils.intro, None)),
        (
            "Price Simulator",
            (
                utils.price_simulator,
                """
This app visualizes price simulation of a chosen asset using [Geometric Brownian motion (GBM)](https://en.wikipedia.org/wiki/Geometric_Brownian_motion).

The model processes historical data and simulates different pathways of how the prices could move over time.

  We simulate asset prices in September 2020 and take into account historical data between January 2020 and August 2020. We then compare the simulated and realized prices (ground truth).
  For this reason we also don't include the last month's price data for training of the simulation model, only for validating the final results.

  There are two inputs into the model:
  - Which asset's price to simulate
  - How many simulations to perform

You can play with both and see how the output changes.

  Some of the questions we might ask:
  - Is the simulation model aligned with the simulated price levels? Or is too optimistic/pessimistic?
  - How does accuracy of the simulation change across different assets? Why is that?
  - How does the simulation results change when we change the number of simulations to run?

""",
            )
        ),
        (
            "Portfolio Optimizer",
            (
                utils.portfolio_optimizer,
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
""",
            ),
        )
    ]
)


def run():
    app_name = st.sidebar.selectbox("Choose one of the apps", list(APPS.keys()), 0)
    app = APPS[app_name][0]

    if app_name == "—":
        st.write("# Welcome to the CompFin data app! 👋")
    else:
        st.markdown("# %s" % app_name)
        description = APPS[app_name][1]
        if description:
            st.write(description)
        # Clear everything from the intro page.
        # We only have 4 elements in the page so this is intentional overkill.
        for i in range(10):
            st.empty()
    
    app()


if __name__ == "__main__":
    run()