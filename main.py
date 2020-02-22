import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
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


# Parse arguments from command line
parser = argparse.ArgumentParser(description='Reinforcement Learning Model.')
parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=500, help='The number of epochs to train the model')
parser.add_argument('-t', '--timesteps', dest='past_timesteps', type=int, default=5, help='Numbers of past data to take into account')
parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, choices=[0.1, 0.01, 0.001], default=0.01, help='Learning rate')
parser.add_argument('--objective_function', dest='objective_function', choices=['Sharpe', 'Dsharpe'], default='Dsharpe', help='Objective function to maximize')
parser.add_argument('--weights_init', dest='weights_init', choices=['Ones', 'Xavier'], default='Ones', help='How to initialize features weights')
parser.add_argument('-v', '--verbose', action='store_true', help='Display more log messages')
parser.add_argument('--version', action='version', version='1.0')
#parser.add_argument('Windows Length', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')

args = parser.parse_args()

# Hyperparameter setting
epochs = args.epochs
past_timesteps = args.past_timesteps
learning_rate = args.learning_rate
verbose = args.verbose
objective_function = args.objective_function

print("Chosen hyperparameters: epochs:{}, windows:{}, learning rate:{}, objective function:{}, weights init scheme:{}, verbose:{}".format(epochs, past_timesteps, learning_rate, objective_function, args.weights_init, verbose))

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

# Available Features
# [['Ouverture', 'Haut', 'Bas', 'ma7', 'ma21', '26ema', '12ema', 'MACD', 'upper_band', 'lower_band', 'ema', 'wr']]

selected_feature = data_with_TI[['Fermeture', 'MACD', 'ema', 'wr']]

# Numbers of selected features * (past_timesteps + 1 (because index start at 0)) + 1=Last_position
nb_features = (past_timesteps + 1) * selected_feature.shape[1] + 1


# Parameters initialization using Xavier initialization for tanh activation function
# Or just using basic all ones initialization
if args.weights_init == "Xavier":
    xavier_weights=np.random.randn(nb_features,1)*np.sqrt(2/(nb_features+1))
    theta=xavier_weights.flatten()
elif args.weights_init == "Ones":
    theta = np.ones(nb_features)

# Initialize sharpe ratios
sharpes = np.zeros(epochs)

# Split train/test sets, without shuffle as it is a time serie.
selected_feature_train, selected_feature_test = train_test_split(selected_feature, test_size=0.2, shuffle=False)
# With Scaling
scaler = StandardScaler() # Or scaler = MinMaxScaler()
selected_feature_scaled = scaler.fit_transform(selected_feature)
selected_feature_scaled = pd.DataFrame(selected_feature_scaled, columns=selected_feature.columns)
selected_feature_train_scaled, selected_feature_test_scaled = train_test_split(selected_feature_scaled, test_size=0.2, shuffle=False)

# Train the model
for i in range(epochs):
    grad, sharpe, positions, returns = policy.DirectReinforcementLearning(selected_feature_train_scaled, past_timesteps, nb_features, theta).gradientAscent(objective_function)
    theta = theta + grad * learning_rate
    if verbose:
       print("epochs:{} -> Gradients are:{} - Params:{}".format(i, grad, theta))
       print("Sharpe: {}".format(sharpe))
    sharpes[i] = sharpe

print("------- Training is over -------")

# Display sharpe ratio improvements over epochs
if verbose:
    plt.figure()
    pd.Series(sharpes).plot()
    plt.legend(['Sharpe ratio'])
    plt.show()


# Adjust indexes for the test set
selected_feature_test_scaled.reset_index(drop=True, inplace=True)
selected_feature_test.reset_index(drop=True, inplace=True)


# Add an ARIMA prediction as a feature. For test set only as ARIMA is used for predicting test set's closed price
# Warning: add too much time to be viable, in an online manner.
# Need to find a way to add it. Currently we can't because we have not a theta for it.
add_arima = False
if(add_arima):
    y_train = selected_feature_train["Fermeture"]
    y_test = selected_feature_test["Fermeture"]
    arima = model.FitARIMA(y_train, y_test).get_arima_values()
    arima_scaled = scaler.fit_transform(arima)
    arima_scaled_series = pd.Series( (v[0] for v in arima_scaled) )
    print("Arima {}".format(arima_scaled_series))
    arima_scaled_series.reset_index(drop=True, inplace=True)

    # Add scaled ARIMA values to feature dataframe
    #selected_feature_test_scaled['ARIMA'] = arima_scaled_series
    selected_feature_test_scaled = selected_feature_test_scaled.assign(ARIMA=arima_scaled_series.values)
    print(selected_feature_test_scaled)

# Run the model with the found parameters, on the test set
grad, sharpe, positions, returns = policy.DirectReinforcementLearning(selected_feature_test_scaled, past_timesteps, nb_features, theta).gradientAscent(objective_function)

# Actualise parameters
theta = theta + grad * learning_rate

# Plot the changing positions
# Separate buy signals from sell signals
changing_positions = (pd.Series(positions).round()).diff()
changing_positions.fillna(0, inplace=True)
xbuy = [i for i in range(len(changing_positions)) if changing_positions[i] > 0]
xsell = [i for i in range(len(changing_positions)) if changing_positions[i] < 0]
ybuy = []
ysell = []
for i in range(len(selected_feature_test_scaled['Fermeture'])):
    for j in range(len(xbuy)):
        if i == xbuy[j]:
            ybuy.append(selected_feature_test_scaled['Fermeture'].iloc[i])

    for k in range(len(xsell)):
        if i == xsell[k]:
            ysell.append(selected_feature_test_scaled['Fermeture'].iloc[i])

# Plot the results
plt.figure()
plt.plot(pd.Series(returns).cumsum(), label="RLModel Add returns", linewidth=1)
plt.plot((selected_feature_test_scaled['Fermeture'].diff()).cumsum(), label="Buy and Hold", linewidth=1)
plt.plot(selected_feature_test_scaled['Fermeture'], label="Closing Price", linewidth=1)
plt.scatter(xbuy, ybuy, s=70, c='red', label="Buy Signal")
plt.scatter(xsell, ysell, s=70, c='blue', label="Sell Signal")
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns');
plt.legend()
plt.title("RL Model vs. Buy and Hold - Test Data");
plt.show()




