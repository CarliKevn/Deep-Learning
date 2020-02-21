import pandas as pd
import numpy as np


# Including transaction costs
# The transaction cost term is subtracted directly from the expected return since the sharpe ratio is unitless
# Transaction costs are applicable only if we have changed our position between two timesteps.
# Positions are (-1,0,1) = long, neutral, short

class Returns:
    def __init__(self, closing_price: pd.Series, position, tc: float=0.05, num_shares_traded: float=1.0):
        self._closing_price = closing_price
        self._position = position
        self._num_shares_traded = num_shares_traded
        self._tc = tc

    # Log returns
    def getLogReturns(self, sharpe=True):
        #log_return = np.log(1 + self._closing_price.pct_change())
        log_return = np.log(self._closing_price/self._closing_price.shift(1))
        log_return.fillna(0, inplace=True)
        T = len(log_return)
        log_returns_with_tc =  self._position[0:T - 1] * log_return[1:T] - self._tc * np.abs(self._position[1:T] - self._position[0:T - 1])
        log_gain = np.concatenate([[0], log_returns_with_tc])
        if(sharpe):
            sharpe_ratio = pd.Series(log_gain).mean() / pd.Series(log_gain).std()
            return pd.Series(log_return), pd.Series(log_gain), sharpe_ratio
        else:
            return pd.Series(log_return), pd.Series(log_gain)

    # Additive profits
    def getAdditiveProfits(self, sharpe=True):
        additive_profits = self._closing_price.diff()
        additive_profits.fillna(0, inplace=True)
        T = len(additive_profits)
        add_returns_with_tc =  self._position[0:T - 1] * additive_profits[1:T] - self._tc * np.abs(self._position[1:T] - self._position[0:T - 1])
        gain = np.concatenate([[0], add_returns_with_tc])
        if(sharpe):
            sharpe_ratio = pd.Series(gain).mean() / pd.Series(gain).std()
            return pd.Series(additive_profits), pd.Series(gain), sharpe_ratio
        else:
            return pd.Series(additive_profits), pd.Series(gain)

    # Multiplicative profits
    def getMultiplicativeProfits(self, sharpe=True, initial_wealth=1):
        mul_profits = self._closing_price[1:T] / self._closing_price[0:T - 1]
        mul_profits.fillna(0, inplace=True)
        T = len(mul_profits)
        mul_profits_with_tc = (1 + self._position[0:T - 1] * mul_profits[1:T]) * (1 - self._tc * np.abs(self._position[1:T] - self._position[0:T - 1]))
        gain = np.concatenate([[0], initial_wealth * mul_profits_with_tc])
        if(sharpe):
            sharpe_ratio = pd.Series(gain).mean() / pd.Series(gain).std()
            return pd.Series(mul_profits), pd.Series(gain), sharpe_ratio
        else:
            return pd.Series(mul_profits), pd.Series(gain)

