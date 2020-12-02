import streamlit as st


def main():

    st.sidebar.success("Select one of the options above.")

    st.markdown(
        """
        
        This is a collection of three data tools for you to explore risk modeling in finance.

        - **Price Simulator** (Simulating asset prices with the Monte Carlo method)
        - **Risk Estimator** (Evaluating portfolio risk with the VaR metric)
        - **Portfolio optimizer** (Optimizing ETF portfolio with mean-variance analysis)

      &nbsp;

      [Github repo](https://github.com/josolnik/compfin_app)
      
      Simulation methodology taken from [Python from Finance Cookbook](https://www.amazon.com/Python-Finance-Cookbook-libraries-financial/dp/1789618517)
    
    """
    )