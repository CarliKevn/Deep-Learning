import numpy as np
import pandas as pd

class RSIInidcator:
    def __init__(self, closing_price: pd.Series, window_length: int = 14):
        self._closing_price = closing_price
        self._window_length = window_length
        self._run()

    def _run(self):
        #Get the closing price difference between two time steps
        closing_price_diff = self._closing_price.diff()
        # Get rid of the first row, which is NaN since it did not have a previous
        # row to calculate the differences
        closing_price_diff = closing_price_diff[1:]

        # Make the positive gains (up) and negative gains (down) Series
        positive_gain = closing_price_diff.where(closing_price_diff > 0, 0.0)
        negative_gain = closing_price_diff.where(closing_price_diff < 0, 0.0)

        # Calculate the EWMA
        roll_up_ewm = positive_gain.ewm(alpha=1/self._window_length, adjust=False).mean()
        roll_down_ewm = negative_gain.ewm(alpha=1/self._window_length, adjust=False).mean()

        # Calculate the RSI based on EWMA
        rs1 = roll_up_ewm / roll_down_ewm
        self._rsi_ewm = 100.0 - (100.0 / (1.0 + rs1))

        # Calculate the SMA
        roll_up_sma = positive_gain.rolling(window=self._window_length).mean()
        roll_down_sma = negative_gain.abs().rolling(window=self._window_length).mean()

        # Calculate the RSI based on SMA
        rs2 = roll_up_sma / roll_down_sma
        self._rsi_sma = 100.0 - (100.0 / (1.0 + rs2))

        self._rsi_ewm.fillna(50, inplace=True)
        self._rsi_sma.fillna(50, inplace=True)

    def rsi(self) -> pd.DataFrame:
        RSI_TI = pd.DataFrame()
        RSI_TI['rsi_ewm'] = self._rsi_ewm
        RSI_TI['rsi_sma'] = self._rsi_sma
        return RSI_TI


class MACDIndicator:
    def __init__(self, closing_price: pd.Series, slow: int = 26, fast: int = 12):
        self._closing_price = closing_price
        self._slow = slow
        self._fast = fast
        self.run()

    def run(self):
        self._emafast = self._closing_price.ewm(span=self._fast).mean()
        self._emaslow = self._closing_price.ewm(span=self._slow).mean()
        self._macd = self._emafast - self._emaslow
        self._macd.fillna(0, inplace=True)

    def macd(self) -> pd.DataFrame:
        MACD_TI = pd.DataFrame()
        MACD_TI['26ema'] = self._emaslow
        MACD_TI['12ema'] = self._emafast
        MACD_TI['MACD'] = self._macd
        return MACD_TI

class BollingerBands:
    def __init__(self, closing_price: pd.Series, n: int = 21, ndev: int = 2):
        self._closing_price = closing_price
        self._n = n
        self._ndev = ndev
        self.run()

    def run(self):
        self._mavg = self._closing_price.rolling(window=self._n).mean()
        self._mstd = self._closing_price.rolling(window=self._n).std()
        self._hband = self._mavg + self._ndev * self._mstd
        self._lband = self._mavg - self._ndev * self._mstd

    def bollinger(self) -> pd.DataFrame:
        BB_TI = pd.DataFrame()
        BB_TI['upper_band'] = self._hband
        BB_TI['lower_band'] = self._lband
        return BB_TI

# Not generalized. But for now its not important as we are only interested about 21 and 7
class MovingAverage:
    def __init__(self, closing_price: pd.Series, up: int = 21, down: int = 7):
        self._closing_price = closing_price
        self._down = down
        self._up = up
        self.run()

    def run(self):
        self._mad = self._closing_price.rolling(window=self._down).mean()
        self._mau = self._closing_price.rolling(window=self._up).mean()
        self._ema = self._closing_price.ewm(com=0.5).mean()

    def movingAverage(self) -> pd.DataFrame:
        MA_TI = pd.DataFrame()
        MA_TI['ma7'] = self._mad
        MA_TI['ma21'] = self._mau
        MA_TI['ema'] = self._ema
        return MA_TI

class WilliamsRIndicator:
    def __init__(self, high: pd.Series, low: pd.Series, close: pd.Series, lbp: int = 14):
        self._high = high
        self._low = low
        self._close = close
        self._lbp = lbp
        self._run()

    def _run(self):
        hh = self._high.rolling(self._lbp, min_periods=0).max()  # highest high over lookback period lbp
        ll = self._low.rolling(self._lbp, min_periods=0).min()  # lowest low over lookback period lbp
        self._wr = -100 * (hh - self._close) / (hh - ll)

    def wr(self) -> pd.Series:
        self._wr.fillna(-50, inplace=True)
        return pd.Series(self._wr, name='wr')




