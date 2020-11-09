import pandas as pd
from ..node import Node


def RSI(node, period=14, basket=False):
    """Relative Strength Index.

    Args:
        node (Node): input node.
        period (int): RSI period
        basket (bool): given a list as input, return a list as output (as opposed to the last value)
    """

    def _rsi(node=node, period=period, basket=basket):
        delta = pd.Series(node.value()).diff().shift(-1)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        _gain = up.ewm(alpha=1.0 / period, adjust=False).mean()
        _loss = down.abs().ewm(alpha=1.0 / period, adjust=False).mean()
        RS = _gain / _loss
        rsi = pd.Series(100 - (100 / (1 + RS)))

        if basket:
            return rsi
        return rsi.iloc[-1]

    # make new node
    ret = node._gennode("RSI[{}]".format(period), _rsi, [node])
    return ret


def MACD(node, period_fast=12, period_slow=26, signal=9, basket=False):
    """Moving Average Convergence/Divergence

    Args:
        node (Node): input data
        period_fast (int): Fast moving average period
        period_slow (int): Slow moving average period
        signal (int): MACD moving average period
        basket (bool): given a list as input, return a list as output (as opposed to the last value)
    Returns:
        Node: node that emits tuple of (macd, macd_signal)
    """

    def _macd(
        node=node,
        period_fast=period_fast,
        period_slow=period_slow,
        signal=signal,
        basket=basket,
    ):
        EMA_fast = pd.Series(
            pd.Series(node.value())
            .ewm(ignore_na=False, span=period_fast, adjust=False)
            .mean(),
            name="EMA_fast",
        )
        EMA_slow = pd.Series(
            pd.Series(node.value())
            .ewm(ignore_na=False, span=period_slow, adjust=False)
            .mean(),
            name="EMA_slow",
        )
        MACD = pd.Series(EMA_fast - EMA_slow, name="MACD")
        MACD_signal = pd.Series(
            MACD.ewm(ignore_na=False, span=signal, adjust=False).mean(), name="SIGNAL"
        )
        macd = pd.concat([MACD, MACD_signal], axis=1)

        if basket:
            return macd.values
        return macd.iloc[-1]

    # make new node
    ret = node._gennode(
        "MACD[{},{},{}]".format(period_fast, period_slow, signal), _macd, [node]
    )
    return ret


Node.rsi = RSI
Node.macd = MACD
