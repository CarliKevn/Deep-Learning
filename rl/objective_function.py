import pandas as pd
import numpy as np

class SharpeRatio:
    def __init__(self, closing_price: pd.Series, position, tc: float = 0.002):
        self._closing_price = closing_price
        self._position = position
        self._tc = tc
        self._run()

    def _run(self):
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
        log_return_grouped = []
#        print(log_return)
        toto = {}
        self._sharpe_ratio = []
        while t < self._closing_price.size - 1:
        #while t < 10:
            if t == 0:
                self._sharpe_ratio.append(0)
            else:
                log_return_grouped.append(log_return[t-1])
                log_return_grouped.append(log_return[t])
                toto['Mean'] = pd.Series(log_return_grouped).mean()
                toto['Std'] = pd.Series(log_return_grouped).std()
                #print("Log return: ", log_return_grouped)
                #print("Mean: ", pd.Series(log_return_grouped).mean())
                #print("Std: ", pd.Series(log_return_grouped).std())
                s_r = (pd.Series(log_return_grouped).mean() - (self._tc * abs(position_diff[t]))) / pd.Series(log_return_grouped).std()
                toto['SR'] = s_r
                self._sharpe_ratio.append(s_r)
                log_return_grouped = []
                print(toto)

            t = t + 1
        #print(pd.Series(self._sharpe_ratio))

    def sharpe(self):
        return self._sharpe_ratio

