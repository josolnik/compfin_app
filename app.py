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
            This tool visualizes price simulation of a chosen asset using [Geometric Brownian motion (GBM)](https://en.wikipedia.org/wiki/Geometric_Brownian_motion).

            The model processes historical data and simulates different pathways of how the prices could evolve over time.

            We simulate asset prices in November 2020 and take into account historical data between January 2020 and October 2020. We then compare the simulated and realized prices (ground truth).
            For this reason we also don't include the last month's price data for training of the simulation model, only for validating the final results.

""",
            )
        ),

        (
            "Risk Estimator",
            (
                utils.risk_estimator,

                """
                
            We would like to evaluate portfolio risk with a metric that estimates worst expected loss.

            For this we use [value-at-risk (VaR)](https://en.wikipedia.org/wiki/Value_at_risk) measure which reports this value at a given level of confidence over a certain time horizon and under normal market conditions.

            To get a better intuitive understanding, it's best to look at an example: If the 1-day 95% VaR of our portfolio is $100 this means that 95% of the time (under normal market conditions), we will not lose more than $100 by holding our portfolio over one day.

            There are various ways to calculate VaR. We will be focusing here on using Monte Carlo simulations.

            The portfolio is preset and assumes 1 share owned in each of the [FAANG](https://www.investopedia.com/terms/f/faang-stocks.asp) companies. For this portfolio, we evaluate risk using the VaR metric from individual security prices.

            For the simulations, we take into account historical price data since 2018-01-01 until the day before analysis (t-1).

            It's worth mentioning that the analysis is a toy example and should be taken with a fist of salt. The metric we're using doesn't take into account potential [Black Swan](https://en.wikipedia.org/wiki/Black_swan_theory) events and is in general not capturing enough [tail risks](https://www.investopedia.com/terms/t/tailrisk.asp).

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
            Below you can find a selection of some of the [most traded ETFs](https://finance.yahoo.com/etfs/) with a short 
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
    app_name = st.sidebar.selectbox("Choose one of the tools", list(APPS.keys()), 0)
    app = APPS[app_name][0]

    if app_name == "—":
        st.write("# Welcome to the CompFin app! 👋")
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