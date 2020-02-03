import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#from statsmodels.regression.linear_model import OLS
#from statsmodels.stats.outliers_influence import variance_inflation_factor
#import category_encoders as ce
#from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
#from statsmodels.tsa.stattools import adfuller
#from statsmodels.tsa.arima_model import ARIMA
#from sklearn.metrics import mean_squared_error
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()



import techindic.indicator as ti
import utils.stats as stats
import econometric.utils as model

def sharpe_ratio_tc(closing_price, position, tc=0.002):
    # Portfolio vector w (weight of each asset)
    # u = vector of expected returns
    # Σ = covariance. Σij the empirical covariance of the ith and the jth assets.
    # Expected  returns  (wTμ)  and  risk  level  (wTΣw)
    # Sharpe = maximize wTμ/√wTΣw
    # For simplicity we have only one asset in our portfolio so we can omit vector w as sum(wi) = 1
    # And  vector u has only one expected return.
    # We can omit covariance and take variance instead. (std because we root squared the variance)

    # Calculate the log return: rt+1 = ln(closing_price(t+1) / (closing_price(t)))
    #log_return = pd.DataFrame()
    #log_return['log_ret'] = np.log(closing_price / closing_price.shift(1))
    log_return = np.log(1 + closing_price.pct_change())
    log_return.fillna(0, inplace=True)

    # Including transaction costs
    # The transaction cost term is subtracted directly from the expected return since the sharpe ratio is unitless
    # Transaction costs are applicable only if we have changed our position between two timesteps.
    # Positions are (-1,0,1) = long, neutral, short
    position_diff = position.diff()
    position_diff.fillna(0, inplace=True)
    t=1
    log_return_grouped = pd.Series()
    sharpe_ratio = pd.Series()
    while t < closing_price.len() - 1:
        if t == 0:
            sharpe_ratio.append(0)
        else:
            log_return_grouped = log_return[t-1:t]
            s_r = (log_return_grouped.mean() - (tc * abs(position_diff[t]))) / log_return_grouped.std()
            sharpe_ration.append(s_r)
            log_return_grouped.drop[:]
        t = t + 1


# Import data, replace unwanted coma for float numbers, and convert to numeric number
#data = pd.read_csv("/home/kevin/DeepLearning/bitcoin.csv")
data = pd.read_csv("./bitcoin.csv")
data.iloc[:, 1:].replace(',','', regex=True, inplace=True)
data_ordered = data.iloc[::-1]
data_processed = pd.concat([data_ordered.iloc[:,0], data_ordered.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')], axis=1)


# Add Technical Indicators as features
rsi = ti.RSIInidcator(data_processed['Fermeture']).rsi()
macd = ti.MACDIndicator(data_processed['Fermeture']).macd()
bb = ti.BollingerBands(data_processed['Fermeture']).bollinger()
ma = ti.MovingAverage(data_processed['Fermeture']).movingAverage()
wr = ti.WilliamsRIndicator(data_processed['Haut'], data_processed['Bas'], data_processed['Fermeture']).wr()

data_with_TI = pd.concat([data_processed, rsi, macd, bb, ma, wr], axis=1)

# Check if evertything is good.
print(data_with_TI.head())

# Split train/test sets, without shuffle as it is a time serie.
data_train, data_test = train_test_split(data_with_TI, test_size=0.2, shuffle=False)
# Feature
X_train = data_train[['Ouverture', 'Haut', 'Bas', 'Cap', 'ma7', 'ma21', '26ema', '12ema', 'MACD', 'upper_band', 'lower_band', 'ema', 'wr']]
X_test = data_test[['Ouverture', 'Haut', 'Bas', 'Cap', 'ma7', 'ma21', '26ema', '12ema', 'MACD', 'upper_band', 'lower_band', 'ema', 'wr']]
# Target
y_train = data_train["Fermeture"]
y_test = data_test["Fermeture"]

#Make an ARIMA prediction
arima = model.FitARIMA(y_train, y_test).get_arima_values()
model.FitARIMA(y_train, y_test).plot_arima_predictions()

