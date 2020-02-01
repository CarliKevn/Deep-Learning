import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import OLS
#from statsmodels.stats.outliers_influence import variance_inflation_factor
import category_encoders as ce
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller
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

def variance_inflation_factor(exog, exog_idx):
    """
    VIF quantifies the severity of multicollinearity in an ordinary least squares regression analysis.
    Measures how much the variance of an estimated regression coefficient is increased because of collinearity.
    """
    k_vars = exog.shape[1]
    x_i = exog[:, exog_idx]
    mask = np.arange(k_vars) != exog_idx
    x_noti = exog[:, mask]
    r_squared_i = OLS(x_i, x_noti, missing='drop').fit().rsquared
    vif = 1. / (1. - r_squared_i)
    return vif

def multicolinearity_check(X, treshold=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > treshold:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

def fit_arima(dataset):
    closed_price_train = [x for x in dataset]
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

def relative_strength_index(closing_price, window_length=14):
    RSI_TI = pd.DataFrame()
    #Get the closing price difference between two time steps
    closing_price_diff = closing_price.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    closing_price_diff = closing_price_diff[1:]

    # Make the positive gains (up) and negative gains (down) Series
    positive_gain, negative_gain = closing_price_diff.copy(), closing_price_diff.copy()
    positive_gain[positive_gain < 0] = 0
    negative_gain[negative_gain > 0] = 0

    # Calculate the EWMA
    roll_up_ewm = pd.DataFrame.ewm(positive_gain, window_length).mean()
    roll_down_ewm = pd.DataFrame.ewm(negative_gain.abs(), window_length).mean()

    # Calculate the RSI based on EWMA
    rs1 = roll_up_ewm / roll_down_ewm
    rsi_ewm = 100.0 - (100.0 / (1.0 + rs1))

    # Calculate the SMA
    roll_up_sma = positive_gain.rolling(window=window_length).mean()
    roll_down_sma = negative_gain.abs().rolling(window=window_length).mean()

    # Calculate the RSI based on SMA
    rs2 = roll_up_sma / roll_down_sma
    rsi_sma = 100.0 - (100.0 / (1.0 + rs2))

    # Compare graphically
    plt.figure()
    rsi_ewm.plot()
    rsi_sma.plot()
    plt.legend(['RSI via EWMA', 'RSI via SMA'])
    plt.show()
    RSI_TI['rsi_ewm'] = rsi_ewm
    RSI_TI['rsi_sma'] = rsi_sma

    return RSI_TI

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
    TI['21sd'] = dataset['Fermeture'].rolling(window=21).std()
    TI['upper_band'] = TI['ma21'] + (TI['21sd']*2)
    TI['lower_band'] = TI['ma21'] - (TI['21sd']*2)

    # Create Exponential moving average
    TI['ema'] = dataset['Fermeture'].ewm(com=0.5).mean()

    # Create Momentum
    TI['momentum'] = dataset['Fermeture']-1

    return TI

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
        if t = 0:
            sharpe_ratio.append(0)
        else
            log_return_grouped = log_return[t-1:t]
            s_r = (log_return_grouped.mean() - (tc * abs(position_diff[t]))) / log_return_grouped.std()
            sharpe_ration.append(s_r)
            log_return_grouped.drop[:]
        t = t + 1




# Import data, replace unwanted coma for float numbers, and convert to numeric number
data = pd.read_csv("/home/kevin/DeepLearning/bitcoin.csv")
data.iloc[:, 1:].replace(',','', regex=True, inplace=True)
data_ordered = data.iloc[::-1]
data2 = pd.concat([data_ordered.iloc[:,0], data_ordered.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')], axis=1)

data_RSI = relative_strength_index(data2['Fermeture'])
data_TI = get_technical_indicators(data2)
data_with_TI = pd.concat([data2, data_TI], axis=1)

# Check if evertything is good.
print(data_with_TI.head())

# Split train/test sets, without shuffle as it is a time serie.
data_train, data_test = train_test_split(data_with_TI, test_size=0.2, shuffle=False)
# Feature
X_train = data_train[['Ouverture', 'Haut', 'Bas', 'Cap', 'ma7', 'ma21', '26ema', '12ema', 'MACD', '21sd', 'upper_band', 'lower_band', 'ema', 'momentum']]
X_test = data_test[['Ouverture', 'Haut', 'Bas', 'Cap', 'ma7', 'ma21', '26ema', '12ema', 'MACD', '21sd', 'upper_band', 'lower_band', 'ema', 'momentum']]
# Target
y_train = data_train["Fermeture"]
y_test = data_test["Fermeture"]

print(y_train.head())
stationary_closed_price = make_stationary(y_train)
#
#closed_price_train = [x for x in stationary_closed_price]

#Make an ARIMA prediction
#fit_arima(y_train)

#print(X_train.loc[:, X_train.isna().any()])
#X_train.Cap = X_train.Cap.astype(np.float64)
#vif = multicolinearity_check(X_train)
#print(vif)


