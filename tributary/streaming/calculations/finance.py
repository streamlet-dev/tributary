from ..node import Node
from ..utils import Reduce
from ...base import StreamNone


def RSI(node, period=14):
    """Relative Strength Index

    Args:
        node (Node): input data
        period (int): RSI period
    Returns:
        Node: stream of RSI calculations
    """

    def _filter(up=True):
        def filter(val, up=up):
            if val is None:
                return StreamNone()

            if up:
                if val > 0:
                    return val
                else:
                    return 0
            if val < 0:
                return abs(val)
            return 0

        return filter

    diff = node.diff()

    ups = (
        diff.apply(_filter(up=True))
        .ema(window_width=period, alpha=1 / period, adjust=True)
        .print("up:")
    )
    downs = (
        diff.apply(_filter(up=False))
        .ema(window_width=period, alpha=1 / period, adjust=True)
        .print("down:")
    )

    RS = ups / downs

    rsi = 100 - (100 / (1 + RS))
    return rsi.abs()


def MACD(node, period_fast=12, period_slow=26, signal=9):
    """Moving Average Convergence/Divergence

    Args:
        node (Node): input data
        period_fast (int): Fast moving average period
        period_slow (int): Slow moving average period
        signal (int): MACD moving average period
    Returns:
        Node: node that emits tuple of (macd, macd_signal)
    """
    fast = node.ema(window_width=period_fast)
    slow = node.ema(window_width=period_slow)
    macd = fast - slow
    signal = macd.ema(window_width=signal)

    return Reduce(macd, signal)


Node.rsi = RSI
Node.macd = MACD
