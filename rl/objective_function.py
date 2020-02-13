import pandas as pd
import numpy as np

class SharpeRatio:
    def __init__(self, closing_price: pd.Series, position, tc: float = 0.002, num_shares_traded: float = 1.0):
        self._closing_price = closing_price
        self._position = position
        self._num_shares_traded = num_shares_traded
        self._tc = tc
        self._run()

    def _run(self):
        # Risk free rate = 0
        # Portfolio vector w (weight of each asset)
        # u = vector of expected returns
        # Σ = covariance. Σij the empirical covariance of the ith and the jth assets.
        # Expected  returns  (wTμ)  and  risk  level  (wTΣw)
        # Sharpe = maximize wTμ/√wTΣw
        # For simplicity we have only one asset in our portfolio so we can omit vector w as sum(wi) = 1
        # And  vector u has only one expected return.
        # We can omit covariance and take variance instead. (std because we root squared the variance)

        # Calculate the log return: rt+1 = ln(closing_price(t+1) / (closing_price(t)))
        log_return = np.log(1 + self._closing_price.pct_change())
        log_return.fillna(0, inplace=True)

        # Including transaction costs
        # The transaction cost term is subtracted directly from the expected return since the sharpe ratio is unitless
        # Transaction costs are applicable only if we have changed our position between two timesteps.
        # Positions are (-1,0,1) = long, neutral, short
        position_diff = self._position.diff()
        position_diff.fillna(0, inplace=True)
        t=0
        self._gain = []
        while t < self._closing_price.size - 1:
            if t == 0:
                self._gain.append(0)
            else:
                self._gain.append(self._num_shares_traded * (self._position[t-1] * log_return[t] - (self._tc * abs(position_diff[t]))))

            t = t + 1

        self._sharpe_ratio = pd.Series(self._gain).mean() / pd.Series(self._gain).std()

    def sharpe(self):
        return self._sharpe_ratio


# Additive returns, with a different way to include transaction cost into the calculation of sharpe ratio
class SharpeRatioVariante:
    def __init__(self, closing_price: pd.Series, position, tc: float = 0.002, num_shares_traded: float = 1.0):
        self._closing_price = closing_price
        self._position = position
        self._num_shares_traded = num_shares_traded
        self._tc = tc
        self._run()

    def _run(self):
        position_diff = self._position.diff()
        position_diff.fillna(0, inplace=True)
        t=0
        additive_return = self._closing_price.diff()
        additive_return.fillna(0, inplace=True)
        self._r = []
        while t < self._closing_price.size - 1:
            if t == 0:
                self._r.append(0)
            else:
                #self._r.append(self._num_shares_traded * (self._position[t-1] * additive_return[t] - (self._tc * abs(position_diff[t]))))
                self._r.append(self._num_shares_traded * (self._position[t-1] * additive_return[t]))

            t = t + 1

        self._sharpe_ratio = pd.Series(self._r).mean() / pd.Series(self._r).std()
    def sharpe(self):
        return self._sharpe_ratio

    def getReturns(self):
        return pd.Series(self._r)


class DifferentialSharpeRatio:
    def __init__(self, closing_price: pd.Series, position, tc: float = 0.002, num_shares_traded: float = 1.0):
        self._closing_price = closing_price
        self._position = position
        self._num_shares_traded = num_shares_traded
        self._tc = tc
        self._run()

    def _run(self):
        position_diff = self._position.diff()
        position_diff.fillna(0, inplace=True)
        t=0
        additive_return = self._closing_price.diff()
        additive_return.fillna(0, inplace=True)
        additive_return_grouped = []
        r = []
        A = []
        B = []
        n = 0.0004
        self._differential_sharpe_ratio = []
        while t < self._closing_price.size - 1:
            additive_return_grouped.append(additive_return[t])
            if t == 0:
                self._differential_sharpe_ratio.append(0)
                r.append(0)
                A.append(0)
                B.append(0)
            else:
                r.append(self._num_shares_traded * (self._position[t-1] * additive_return[t] - (self._tc * abs(position_diff[t]))))
                A.append(A[t-1] + n * (pd.Series(r).mean() - A[t-1]))
                B.append(B[t-1] + n * (pd.Series(r).std() - B[t-1]))
                if t < 20:
                    self._differential_sharpe_ratio.append(0)
                else:
                    self._differential_sharpe_ratio.append((B[t-1] * (pd.Series(r).mean() - A[t-1]) - A[t-1] / 2 * (pd.Series(r).std() - B[t-1])) / (B[t-1] - A[t-1]**2)**(3/2))
            t = t + 1

    def sharpe(self):
        return self._differential_sharpe_ratio

