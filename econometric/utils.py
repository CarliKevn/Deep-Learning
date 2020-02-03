from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


class Stationarity:
    def __init__(self, closing_price: pd.Series, cutoff: float = 0.01):
        self._closing_price = closing_price
        self._cutoff = cutoff
        self._run()

    def _run(self):
        self._pvalue = adfuller(self._closing_price)[1]
        if self._pvalue < self._cutoff:
            self.stationary = True
        else:
            self.stationary = False

    def check_for_stationarity(self):
        # H_0 in adfuller is unit root exists (non-stationary)
        # We must observe significant p-value to convince ourselves that the series is stationary
        print('p-value = {}' .format(self._pvalue))
        return self.stationary


    # Deprecated - Need to confirm these results
    def make_stationary(self):
        if self.stationary:
            return self._closing_price
        else:
            print(' The series is not likely stationary.')
            log = np.log(self._closing_price)
            rolling_mean = log.rolling(window=12).mean()
            df_log_minus_mean = log - rolling_mean
            df_log_minus_mean.dropna(inplace=True)
            return df_log_minus_mean



class FitARIMA:
    def __init__(self, closing_price_train: pd.Series, closing_price_test: pd.Series):
        self._closing_price_train = closing_price_train
        self._closing_price_test = closing_price_test
        self._run()

    def _run(self):
        self._closed_price_train = [x for x in self._closing_price_train]
        self._arima_predictions = list()
        self._closed_price_test = [x for x in self._closing_price_test]
        for t in range(len(self._closed_price_test)):
            model = ARIMA(self._closed_price_train, order=(2,1,2))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            self._arima_predictions.append(yhat)
            #Update known value as we move in time
            self._closed_price_train.append(self._closed_price_test[t])

    def plot_arima_predictions(self):
        plt.figure(figsize=(12, 6), dpi=100)
        plt.plot(self._closed_price_test, label='Real')
        plt.plot(self._arima_predictions, color='red', label='Predicted')
        plt.xlabel('Days')
        plt.ylabel('Closed Price')
        plt.title('Figure 5: ARIMA model on Bitcoin')
        plt.legend()
        plt.show()

    def get_arima_values(self):
        error = mean_squared_error(self._closed_price_test, self._arima_predictions)
        print('MSE: %.3f' % error)
        return self._arima_predictions

