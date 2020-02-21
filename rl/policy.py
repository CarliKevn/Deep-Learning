import numpy as np
import pandas as pd

import rl.objective_function as of

class DirectReinforcementLearning:
    def __init__(self, feature_vector: pd.DataFrame, window_length: int, nb_params: int, theta, tc: float = 0.5):
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

    def gradientAscent(self, diffSharpe=False, n=0.005):

       # profits, returns, sharpe_ratio = of.Returns(self._feature_vector['Fermeture'], pd.Series(self._position).round(), self._tc).getAdditiveProfits()
        profits, returns, sharpe_ratio = of.Returns(self._feature_vector['Fermeture'], pd.Series(self._position).round(), self._tc).getLogReturns()

        self._grad = np.zeros(self._nb_params)
        prevTheta = np.zeros(self._nb_params)
        t = self._window_length
        L = len(returns)
        if diffSharpe:
            A = np.zeros(self._feature_vector.shape[0] - 1)
            B = np.zeros(self._feature_vector.shape[0] - 1)
        else:
            A = np.mean(returns)
            B = np.mean(np.square(returns))
            K = np.sqrt(L / (L - 1))
            #sharpe_ratio = A / ((np.sqrt(B - A ** 2)) * K)

        while t < self._feature_vector.shape[0] - 1:
            flatten = (self._feature_vector.loc[t-self._window_length:t, :].to_numpy()).flatten()
            self._input_feature = np.concatenate([flatten, [self._position[t-1]]])
            dRdF = -self._tc * np.sign(self._position[t] - self._position[t-1])
            dRdFp = profits[t] + self._tc * np.sign(self._position[t] - self._position[t-1])
            dFdtheta = (1 - self._position[t] ** 2) * (self._input_feature + self._theta[-1] * prevTheta)
            dSdtheta = (dRdF * dFdtheta + dRdFp * prevTheta)
            if diffSharpe:
                A[t] = A[t-1] + (n * (returns[t] - A[t-1]))
                B[t] = B[t-1] + (n * (returns[t]**2  - B[t-1]))
                Op1 = B[t-1] - A[t-1] * returns[t]
                Op2 = (B[t-1] - (A[t-1]**2))**(3/2)
                if(Op2 > 0):
                    dDiffSharpe = Op1 / Op2
                else:
                    dDiffSharpe = 0
                mul = dDiffSharpe * dSdtheta
            else:
                dSharpe = (B - A*returns[t]) / (K * ((B - A ** 2)**(3/2)))
                mul = dSharpe * dSdtheta

            self._grad = self._grad + mul
            prevTheta = dFdtheta
            t = t + 1

        self._grad = self._grad / (t - self._window_length)

        return self._grad, sharpe_ratio, self._position, returns
