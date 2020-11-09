import pandas as pd
import tributary.lazy as tl


class TestFinance:
    def test_rsi(self):
        df = pd.DataFrame(pd.util.testing.getTimeSeriesData(20))
        adjust = False
        period = 14
        delta = df["A"].diff().shift(-1)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        _gain = up.ewm(alpha=1.0 / period, adjust=adjust).mean()
        _loss = down.abs().ewm(alpha=1.0 / period, adjust=adjust).mean()
        RS = _gain / _loss
        rsi = pd.Series(100 - (100 / (1 + RS)))

        val = [df["A"].iloc[0]]
        n = tl.Node(value=val)
        n_rsi = n.rsi(period=period)

        for i, x in enumerate(df["A"][1:]):
            val.append(x)
            n._dirty = True
            print("data\t", i, x, n_rsi(), rsi[i])
            assert abs(n_rsi() - rsi[i]) < 0.003

        n = tl.Node(value=val)
        n_rsi = n.rsi(period=period, basket=True)
        assert n_rsi().tolist() == rsi.tolist()

    def test_macd(self):
        df = pd.DataFrame(pd.util.testing.getTimeSeriesData(20))

        period_fast = 12
        period_slow = 26
        signal = 9
        adjust = False

        EMA_fast = pd.Series(
            df["A"].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
            name="EMA_fast",
        )
        EMA_slow = pd.Series(
            df["A"].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
            name="EMA_slow",
        )
        MACD = pd.Series(EMA_fast - EMA_slow, name="MACD")
        MACD_signal = pd.Series(
            MACD.ewm(ignore_na=False, span=signal, adjust=adjust).mean(), name="SIGNAL"
        )

        expected = pd.concat([MACD, MACD_signal], axis=1)

        val = []
        n = tl.Node(value=val)
        n_macd = n.macd(period_fast=period_fast, period_slow=period_slow, signal=signal)

        for i, x in enumerate(df["A"]):
            val.append(x)
            n_macd._dirty = True
            ret = n_macd()
            assert expected.values[i][0] - ret[0] < 0.001
            assert expected.values[i][1] - ret[1] < 0.001

        n = tl.Node(value=val)
        n_macd = n.macd(
            period_fast=period_fast, period_slow=period_slow, signal=signal, basket=True
        )

        assert n_macd().tolist() == expected.values.tolist()
