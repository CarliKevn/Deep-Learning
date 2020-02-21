import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import techindic.indicator as ti
import utils.stats as stats
import econometric.utils as model
import rl.objective_function as of
import rl.policy as policy

# Import data, replace unwanted coma for float numbers, and convert to numeric number
data = pd.read_csv("./bitcoin.csv")
#data = pd.read_csv("./sp500.csv")
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

# Available Features
# [['Ouverture', 'Haut', 'Bas', 'ma7', 'ma21', '26ema', '12ema', 'MACD', 'upper_band', 'lower_band', 'ema', 'wr']]

selected_feature = data_with_TI[['Fermeture', 'MACD', 'ema', 'wr']]

epochs=5
past_timesteps=5
# nb features = all columns * (past_timesteps +1(because index start at 0))rows + Last_position
nb_features = (past_timesteps + 1) * selected_feature.shape[1] + 1

# Parameters initialization using Xavier initialization for tanh activation function
xavier_weights=np.random.randn(nb_features,1)*np.sqrt(2/(nb_features+1))
theta=xavier_weights.flatten()
# Parameters initialization using Basic all ones initialization
theta = np.ones(nb_features)

# Initialize sharpe ratios
sharpes = np.zeros(epochs)

#learning_rate = 0.1
learning_rate = 0.01

# Split train/test sets, without shuffle as it is a time serie.
selected_feature_train, selected_feature_test = train_test_split(selected_feature, test_size=0.2, shuffle=False)
#scaler = MinMaxScaler()
scaler = StandardScaler()
selected_feature[['Fermeture', 'MACD', 'ema', 'wr']] = scaler.fit_transform(selected_feature)
#selected_feature[['Fermeture']] = scaler.fit_transform(selected_feature)
selected_feature_train_scaled, selected_feature_test_scaled = train_test_split(selected_feature, test_size=0.2, shuffle=False)

for i in range(epochs):
    grad, sharpe, positions, returns = policy.DirectReinforcementLearning(selected_feature_train_scaled, past_timesteps, nb_features, theta).gradientAscent(diffSharpe=True)
    theta = theta + grad * learning_rate
    print("epochs:{} -> Gradients are:{} - Params:{}".format(i, grad, theta))
    print("Sharpe: {}".format(sharpe))
    sharpes[i] = sharpe

print("Training is over")

plt.figure()
pd.Series(sharpes).plot()
plt.legend(['Sharpe ratio'])
plt.show()

selected_feature_test_scaled.reset_index(drop=True, inplace=True)
selected_feature_test.reset_index(drop=True, inplace=True)

add_arima = False
if(add_arima):
    # Add an ARIMA prediction as a feature. For test set only as ARIMA is used for predicting test set's closed price
    # Warning: add too much time to be viable, in an online manner.
    y_train = selected_feature_train["Fermeture"]
    y_test = selected_feature_test["Fermeture"]
    arima = model.FitARIMA(y_train, y_test).get_arima_values()
    arima_scaled = pd.Series(scaler.fit_transform(arima))
    arima_scaled.reset_index(drop=True, inplace=True)

    # Add scaled ARIMA values to feature dataframe
    selected_feature_test_scaled['ARIMA'] = arima_scaled
    print(selected_feature_test_scaled['ARIMA'])

grad, sharpe, positions, returns = policy.DirectReinforcementLearning(selected_feature_test_scaled, past_timesteps, nb_features, theta).gradientAscent(diffSharpe=True)

theta = theta + grad * learning_rate

plt.figure()
plt.plot(pd.Series(returns).cumsum(), label="RLModel Add returns", linewidth=1)
plt.plot((selected_feature_test_scaled['Fermeture'].diff()).cumsum(), label="Buy and Hold", linewidth=1)
plt.plot(selected_feature_test_scaled['Fermeture'], label="Closing Price", linewidth=1)
plt.plot(pd.Series(positions).round(), label="positions", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns');
plt.legend()
plt.title("RL Model vs. Buy and Hold - Test Data");
plt.show()




