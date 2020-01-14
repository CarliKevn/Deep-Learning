import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def check_for_stationarity(X, cutoff=0.01):
    # H_0 in adfuller is unit root exists (non-stationary)
    # We must observe significant p-value to convince ourselves that the series is stationary
    pvalue = adfuller(X)[1]
    print('p-value = {}' .format(pvalue))
    if pvalue < cutoff:
        return True
    else:
        return False

def make_stationary(X):
    if check_for_stationarity(X):
        print('The series is likely stationary.')
        return X
    else:
        print(' The series is not likely stationary.')
        log = np.log(X)
        rolling_mean = log.rolling(window=12).mean()
        df_log_minus_mean = log - rolling_mean
        df_log_minus_mean.dropna(inplace=True)
        if check_for_stationarity(df_log_minus_mean):
            print('The series is now stationary.')
            return df_log_minus_mean
        else:
            print(' The series is still not stationary.')

def plot_arima_predictions(X,Y):
    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(Y, label='Real')
    plt.plot(X, color='red', label='Predicted')
    plt.xlabel('Days')
    plt.ylabel('Closed Price')
    plt.title('Figure 5: ARIMA model on Bitcoin')
    plt.legend()
    plt.show()

def get_technical_indicators(dataset):
    TI = pd.DataFrame()
    # Create 7 and 21 days Moving Average
    TI['ma7'] = dataset['Fermeture'].rolling(window=7).mean()
    TI['ma21'] = dataset['Fermeture'].rolling(window=21).mean()

    # Create MACD
    TI['26ema'] = pd.DataFrame.ewm(dataset['Fermeture'], span=26).mean()
    TI['12ema'] = pd.DataFrame.ewm(dataset['Fermeture'], span=12).mean()
    TI['MACD'] = (TI['12ema']-TI['26ema'])

    # Create Bollinger Bands
    TI['21sd'] =dataset['Fermeture'].rolling(window=21).std()
    TI['upper_band'] = TI['ma21'] + (TI['21sd']*2)
    TI['lower_band'] = TI['ma21'] - (TI['21sd']*2)

    # Create Exponential moving average
    TI['ema'] = dataset['Fermeture'].ewm(com=0.5).mean()

    # Create Momentum
    TI['momentum'] = dataset['Fermeture']-1

    return TI

data = pd.read_csv("/home/kevin/DeepLearning/bitcoin.csv")
data.iloc[:, 1:].replace(',','', regex=True, inplace=True)
data_ordered = data.iloc[::-1]
data2 = pd.concat([data_ordered.iloc[:,0], data_ordered.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')], axis=1)

data_TI = get_technical_indicators(data2)
data_with_TI = pd.concat([data2, data_TI], axis=1)

#print(data_with_TI.head())
data_train, data_test = train_test_split(data_with_TI, test_size=0.2, shuffle=False)
X_train = data_train[['Ouverture', 'Haut', 'Bas', 'Cap', 'ma7', 'ma21', '26ema', '12ema', 'MACD', '21sd', 'upper_band', 'lower_band', 'ema', 'momentum']]
y_train = data_train["Fermeture"]
X_test = data_test[['Ouverture', 'Haut', 'Bas', 'Cap', 'ma7', 'ma21', '26ema', '12ema', 'MACD', '21sd', 'upper_band', 'lower_band', 'ema', 'momentum']]
y_test = data_test["Fermeture"]

print(y_train.head())
stationary_closed_price = make_stationary(y_train)
#
#closed_price_train = [x for x in stationary_closed_price]
closed_price_train = [x for x in y_train]
arima_predictions = list()
closed_price_test = [x for x in data_test['Fermeture']]
for t in range(len(closed_price_test)):
    model = ARIMA(closed_price_train, order=(2,1,2))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    arima_predictions.append(yhat)
    #Update known value as we move in time
    closed_price_train.append(closed_price_test[t])

error = mean_squared_error(closed_price_test, arima_predictions)
print('MSE: %.3f' % error)

plot_arima_predictions(arima_predictions, closed_price_test)

#print(data_train.info())
#print(data_test.head())


#plot closed Fermeture

#plt.figure(figsize=(14, 5), dpi=100)
#plt.plot(data2['Date'], data2['Fermeture'], label='Bitcoin closed Fermeture')
#plt.xlabel('Date')
#plt.ylabel('Fermeture')
#plt.title('Figure 1: Bitcoin closed Fermeture')
#plt.legend()
#plt.show()
