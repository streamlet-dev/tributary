import sympy as sy
from sympy.stats import Normal as syNormal, cdf


class TestConfig:
    def setup(self):
        pass
        # setup() before each test method

    def test_1(self):
        # adapted from https://gist.github.com/raddy/bd0e977dc8437a4f8276
        spot, strike, vol, dte, rate, cp = sy.symbols('spot strike vol dte rate cp')

        T = dte / 260.
        N = syNormal('N', 0.0, 1.0)

        d1 = (sy.ln(spot / strike) + (0.5 * vol ** 2) * T) / (vol * sy.sqrt(T))
        d2 = d1 - vol * sy.sqrt(T)

        TimeValueExpr = sy.exp(-rate * T) * (cp * spot * cdf(N)(cp * d1) - cp * strike * cdf(N)(cp * d2))

        import tributary.symbolic as ts
        PriceClass = ts.construct_lazy(TimeValueExpr)

        price = PriceClass(spot=210.59, strike=205, vol=14.04, dte=4, rate=.2175, cp=-1)

        x = price.evaluate()()

        price.strike = 210

        assert x != price.evaluate()()

        ts.graphviz(TimeValueExpr)
        assert ts.traversal(TimeValueExpr)
        assert ts.symbols(TimeValueExpr)
