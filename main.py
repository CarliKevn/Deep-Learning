import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from random import randint
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
import rl.objective_function as of

# Import data, replace unwanted coma for float numbers, and convert to numeric number
#data = pd.read_csv("/home/kevin/DeepLearning/bitcoin.csv")
data = pd.read_csv("./bitcoin.csv")
data.iloc[:, 1:].replace(',','', regex=True, inplace=True)
data_ordered = data.iloc[::-1].reset_index(drop=True)
data_processed = pd.concat([data_ordered.iloc[:,0], data_ordered.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')], axis=1)

# Add Technical Indicators as features
rsi = ti.RSIInidcator(data_processed['Fermeture']).rsi()
macd = ti.MACDIndicator(data_processed['Fermeture']).macd()
bb = ti.BollingerBands(data_processed['Fermeture']).bollinger()
ma = ti.MovingAverage(data_processed['Fermeture']).movingAverage()
wr = ti.WilliamsRIndicator(data_processed['Haut'], data_processed['Bas'], data_processed['Fermeture']).wr()

data_with_TI = pd.concat([data_processed, rsi, macd, bb, ma, wr], axis=1)

# Check if evertything is good.
#print(data_with_TI.head())

# Split train/test sets, without shuffle as it is a time serie.
data_train, data_test = train_test_split(data_with_TI, test_size=0.2, shuffle=False)
# Feature
X_train = data_train[['Ouverture', 'Haut', 'Bas', 'Cap', 'ma7', 'ma21', '26ema', '12ema', 'MACD', 'upper_band', 'lower_band', 'ema', 'wr']]
X_test = data_test[['Ouverture', 'Haut', 'Bas', 'Cap', 'ma7', 'ma21', '26ema', '12ema', 'MACD', 'upper_band', 'lower_band', 'ema', 'wr']]
# Target
y_train = data_train["Fermeture"]
y_test = data_test["Fermeture"]

random = [randint(-1, 1) for iter in range(data_processed['Fermeture'].size)]
position = pd.Series(random)
#sharpe_ratio = pd.Series(of.SharpeRatio(data_processed['Fermeture'], position).sharpe())
#sharpe_ratio_additive = pd.Series(of.SharpeRatioVariante(data_processed['Fermeture'], position).sharpe())
#sharpe_ratio_differential = pd.Series(of.DifferentialSharpeRatio(data_processed['Fermeture'], position).sharpe())
#print(sharpe_ratio_additive)

selected_feature = data_processed[['Ouverture', 'Haut', 'Bas', 'Fermeture']]

#plt.figure()
#sharpe_ratio_differential.plot()
#plt.legend(['Sharpe ratio'])
#plt.show()

#Make an ARIMA prediction
#arima = model.FitARIMA(y_train, y_test).get_arima_values()
#model.FitARIMA(y_train, y_test).plot_arima_predictions()

