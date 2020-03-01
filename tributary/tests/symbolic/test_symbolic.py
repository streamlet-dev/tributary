import time
import tributary.symbolic as ts
import tributary.streaming as tss
import sympy as sy
import numpy as np
import sympy as sy
from IPython.display import display, HTML
from sympy.stats import Normal as syNormal, cdf
sy.init_printing()


class TestConfig:
    def setup(self):
        pass
        # setup() before each test method

    def test_construct_lazy(self):
        # adapted from https://gist.github.com/raddy/bd0e977dc8437a4f8276
        spot, strike, vol, dte, rate, cp = sy.symbols('spot strike vol dte rate cp')

        T = dte / 260.
        N = syNormal('N', 0.0, 1.0)

        d1 = (sy.ln(spot / strike) + (0.5 * vol ** 2) * T) / (vol * sy.sqrt(T))
        d2 = d1 - vol * sy.sqrt(T)

        TimeValueExpr = sy.exp(-rate * T) * (cp * spot * cdf(N)(cp * d1) - cp * strike * cdf(N)(cp * d2))

        PriceClass = ts.construct_lazy(TimeValueExpr)

        price = PriceClass(spot=210.59, strike=205, vol=14.04, dte=4, rate=.2175, cp=-1)

        x = price.evaluate()()

        assert price.evaluate()() == x

        price.strike = 210

        assert x != price.evaluate()()

    def test_others(self):
        # adapted from https://gist.github.com/raddy/bd0e977dc8437a4f8276
        spot, strike, vol, dte, rate, cp = sy.symbols('spot strike vol dte rate cp')
        T = dte / 260.
        N = syNormal('N', 0.0, 1.0)
        d1 = (sy.ln(spot / strike) + (0.5 * vol ** 2) * T) / (vol * sy.sqrt(T))
        d2 = d1 - vol * sy.sqrt(T)
        TimeValueExpr = sy.exp(-rate * T) * (cp * spot * cdf(N)(cp * d1) - cp * strike * cdf(N)(cp * d2))
        PriceClass = ts.construct_lazy(TimeValueExpr)
        price = PriceClass(spot=210.59, strike=205, vol=14.04, dte=4, rate=.2175, cp=-1)
        price.evaluate()()
        ts.graphviz(TimeValueExpr)
        assert ts.traversal(TimeValueExpr)
        assert ts.symbols(TimeValueExpr)

    def test_parse(self):
        from sympy.parsing.sympy_parser import parse_expr
        assert parse_expr('x^2') == ts.parse_expression('x^2')

    def test_construct_streaming(self):
        # adapted from https://gist.github.com/raddy/bd0e977dc8437a4f8276
        # spot, strike, vol, days till expiry, interest rate, call or put (1,-1)
        spot, strike, vol, dte, rate, cp = sy.symbols('spot strike vol dte rate cp')

        T = dte / 260.
        N = syNormal('N', 0.0, 1.0)

        d1 = (sy.ln(spot / strike) + (0.5 * vol ** 2) * T) / (vol * sy.sqrt(T))
        d2 = d1 - vol * sy.sqrt(T)

        TimeValueExpr = sy.exp(-rate * T) * (cp * spot * cdf(N)(cp * d1) - cp * strike * cdf(N)(cp * d2))

        PriceClass = ts.construct_streaming(TimeValueExpr)

        def strikes():
            strike = 205
            while strike <= 220:
                yield strike
                strike += 2.5

        price = PriceClass(spot=tss.Const(210.59),
                           #    strike=tss.Print(tss.Const(205), text='strike'),
                           strike=tss.Foo(strikes, interval=1),
                           vol=tss.Const(14.04),
                           dte=tss.Const(4),
                           rate=tss.Const(.2175),
                           cp=tss.Const(-1))

        ret = tss.run(tss.Print(price._node))
        time.sleep(2)
        assert len(ret) == 7
