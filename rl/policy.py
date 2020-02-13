import numpy as np
import pandas as pd

import rl.objective_function as of

class DirectReinforcementLearning:
    def __init__(self, feature_vector: pd.DataFrame, window_length: int, nb_params: int, theta, tc: float = 0.002):
        self._feature_vector = feature_vector
        self._window_length = window_length
        self._theta = theta
        self._tc = tc
        self._nb_params = nb_params
        self._run()

    def _run(self):
        # position_at_t = tanh[<W.feature_vector> + b + u*position_at_t-1]
        # u*position_at_t-1 take the latest trading decision into account to discourage to change positions frequently
        # W and b = coefficietns for the feature regression
        # take the recent m return values as feature vector ?

        # Optimization process = learn(W,u,b) that can maximize global performance function
        # Ut(R1, R2, R3, R4, ...,RT)

        # position =(-1,0,1)
        # position_at_t0 = 0

        self._position = np.zeros(self._feature_vector.shape[0])
        t = self._window_length
        # for all timestep, take all last window_length feature, flatten them, and add last position
        while t < self._feature_vector.shape[0] - 1:
            flatten = (self._feature_vector.loc[t-self._window_length:t, :].to_numpy()).flatten()
            self._input_feature = np.concatenate([flatten, [self._position[t-1]]])
            self._position[t] = np.tanh(np.dot(self._theta, self._input_feature))
            #print("position:{}, dot:{}".format(self._position[t], np.dot(self._theta, self._input_feature)))
            t = t + 1

    def gradientAscent(self):

        sharpe_ratio = of.SharpeRatioVariante(self._feature_vector['Fermeture'], pd.Series(self._position)).sharpe()
        #print("Sharpe:{}".format(sharpe_ratio))

        self._grad = np.zeros(self._nb_params)
        prevTheta = np.zeros(self._nb_params)
        additive_return = self._feature_vector['Fermeture'].diff()
        additive_return.fillna(0, inplace=True)
        t = self._window_length
        while t < self._feature_vector.shape[0] - 1:
            flatten = (self._feature_vector.loc[t-self._window_length:t, :].to_numpy()).flatten()
            self._input_feature = np.concatenate([flatten, [self._position[t-1]]])
            dRdF = -self._tc * np.sign(self._position[t] - self._position[t-1])
            dRdFp = additive_return[t] + self._tc * np.sign(self._position[t] - self._position[t-1])
            dFdtheta = (1 - self._position[t] ** 2) * (self._input_feature + self._theta[-1] * prevTheta)
            dSdtheta = (dRdF * dFdtheta + dRdFp * prevTheta)
            self._grad = self._grad + dSdtheta
            prevTheta = dFdtheta
            t = t + 1

            #print("dRdF:{}, dRdFp:{}, dFdtheta:{}, dSdtheta:{}, self._grad:{}, prevTheta:{}".format(dRdF,dRdFp,dFdtheta, dSdtheta, self._grad, prevTheta))
        return self._grad, sharpe_ratio, self._position
