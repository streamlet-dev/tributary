import pandas as pd
import tributary.streaming as ts


class TestFinance:
    def test_rsi(self):
        vals = pd.DataFrame(pd.util.testing.getTimeSeriesData(20))
        adjust = False
        period = 14
        delta = vals['A'].diff().shift(-1)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        _gain = up.ewm(alpha=1.0 / period, adjust=adjust).mean()
        _loss = down.abs().ewm(alpha=1.0 / period, adjust=adjust).mean()
        RS = _gain / _loss
        rsi = pd.Series(100 - (100 / (1 + RS))).values

        curve = ts.Curve(vals['A'].tolist())
        ret = ts.run(
            ts.Print(ts.RSI(curve, rsi_periods=period), 'rsi:')
        )
        for x, y in zip(ret, rsi):
            if pd.isna(y):
                continue
            print('({}, {})'.format(x, y))
            assert (x - y) < 0.001
