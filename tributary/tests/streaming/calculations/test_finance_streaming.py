import pandas as pd
import tributary.streaming as ts


class TestFinance:
    def test_rsi(self):
        vals = pd.DataFrame(pd.util.testing.getTimeSeriesData(20))
        adjust = False
        period = 14
        delta = vals["A"].diff().shift(-1)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        _gain = up.ewm(alpha=1.0 / period, adjust=adjust).mean()
        _loss = down.abs().ewm(alpha=1.0 / period, adjust=adjust).mean()
        RS = _gain / _loss
        rsi = pd.Series(100 - (100 / (1 + RS))).values

        curve = ts.Curve(vals["A"].tolist())
        ret = ts.run(ts.Print(ts.RSI(curve, period=period), "rsi:"))
        for x, y in zip(ret, rsi):
            if pd.isna(y):
                continue
            print("({}, {})".format(x, y))
            assert abs(x - y) < 0.001

    def test_macd(self):
        vals = pd.DataFrame(pd.util.testing.getTimeSeriesData(20))
        period_fast = 12
        period_slow = 26
        signal = 9
        adjust = False

        EMA_fast = pd.Series(
            vals["A"].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
            name="EMA_fast",
        )
        EMA_slow = pd.Series(
            vals["A"].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
            name="EMA_slow",
        )
        MACD = pd.Series(EMA_fast - EMA_slow, name="MACD")
        MACD_signal = pd.Series(
            MACD.ewm(ignore_na=False, span=signal, adjust=adjust).mean(), name="SIGNAL"
        )

        expected = pd.concat([MACD, MACD_signal], axis=1).values

        curve = ts.Curve(vals["A"].tolist())
        ret = ts.run(ts.MACD(curve).print("macd:"))

        for i, (macd, signal) in enumerate(ret):
            assert abs(expected[i][0] - macd) < 0.001
            assert abs(expected[i][1] - signal) < 0.001
