from ..node import Node
from ...base import StreamNone


def RSI(node, rsi_periods=14):
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

    ups = diff.apply(_filter(up=True)).ema(window_width=rsi_periods, alpha=1 / rsi_periods, adjust=True).print('up:')
    downs = diff.apply(_filter(up=False)).ema(window_width=rsi_periods, alpha=1 / rsi_periods, adjust=True).print('down:')

    RS = ups / downs

    rsi = 100 - (100 / (1 + RS))
    return rsi.abs()


Node.rsi = RSI
