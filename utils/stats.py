from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

class VarianceInflationFactor:
    def __init__(self, exog, exog_idx):
        self._exog = exog
        self._exog_idx = exog_idx
        self._run()

    def run(self):
        """
        VIF quantifies the severity of multicollinearity in an ordinary least squares regression analysis.
        Measures how much the variance of an estimated regression coefficient is increased because of collinearity.
        """
        k_vars = self._exog.shape[1]
        x_i = self._exog[:, self._exog_idx]
        mask = np.arange(k_vars) != self._exog_idx
        x_noti = self._exog[:, mask]
        r_squared_i = OLS(x_i, x_noti, missing='drop').fit().rsquared
        self._vif = 1. / (1. - r_squared_i)

    def vif(self):
        return self._vif


class Multicolinearity:
    def __init__(self, X, treshold=5.0):
        self._X = X
        self._treshold = treshold
        self._run()

    def run(self):
        self._variables = list(range(self._X.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(self._X.iloc[:, self._variables].values, ix)
                   for ix in range(self._X.iloc[:, self._variables].shape[1])]

            maxloc = vif.index(max(vif))
            if max(vif) > self._treshold:
                print('dropping \'' + self._X.iloc[:, self._variables].columns[maxloc] +
                      '\' at index: ' + str(maxloc))
                del self._variables[maxloc]
                dropped = True

    def mc(self):
        return self._X.iloc[:, self._variables]


