import numpy as np
import panda as pd

class DirectReinforcementLearning:
    def __init__(self, feature_vector: pd.Dataframe, past_position: int, theta):
        self._feature_vector = feature_vector
        self._past_position = past_position
        self._theta = theta
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
        t = self._past_position
        while t < self._feature_vector.shape[0] - 1:
            flatten = (self._feature_vector.loc[t-self._past_position:t, :].to_numpy()).flatten()
            self._input_feature = np.concatenate([flatten, [self._position[t-1]]])
            self._position = np.tanh(np.dot(self._theta, input_feature))
            t = t + 1

    def getPosition(self):
        return self._position


    def gradientAscent(self):

        sharpes = pd.Series(of.DifferentialSharpeRatio(self._feature_vector['Fermeture'], self._position).sharpe())

        self._grad = np.zeros(self._input_feature.size + 1)
        prevTheta = np.zeros(self._input_feature.size + 1)
        additive_return = self._feature_vector['Fermeture'].diff()
        additive_return.fillna(0, inplace=True)
        t = self._past_position
        while t < self._feature_vector.shape[0] - 1:
            flatten = (self._feature_vector.loc[t-self._past_position:t, :].to_numpy()).flatten()
            self._input_feature = np.concatenate([flatten, [self._position[t-1]]])
            dRdF = -self._tc * np.sign(self._position[t] - self._position[t-1])
            dRdFp = additive_return[t] + self._tc * np.sign(self._position[t] - self._position[t-1])
            dFdtheta = (1 - self._position[t] ** 2) * (self._input_feature + self._theta[-1] * prevTheta)
            dSdtheta = (dRdF * dFdtheta + dRdFp * prevTheta)
            self._grad = self._grad + dSdtheta
            prevTheta = dFdtheta

        return self._grad, sharpes
