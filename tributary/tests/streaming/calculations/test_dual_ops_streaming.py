import math
import numpy as np
import tributary.streaming as ts


rng = range(-10, 11)


def foo_range():
    for _ in rng:
        yield (_, 1)


pos_rng = range(1, 11)


def foo_pos():
    for _ in pos_rng:
        yield (_, 1)


neg_rng = range(-10, 0)


def foo_neg():
    for _ in neg_rng:
        yield (_, 1)


zero_one_rng = np.arange(0, 1, 0.05)  # [0,1)


def foo_zero_one():
    for _ in zero_one_rng:
        yield (_, 0.05)


class TestDualOps:
    def test_Noop(self):
        """
        No-op
        """
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Noop(t)
        assert ts.run(out) == list(foo_range())

    def test_Negate(self):
        """
        f = -x
        f' = -1
        """
        expected_pos = [(-1 * x, -1) for x in pos_rng]
        expected_neg = [(-1 * x, -1) for x in neg_rng]
        t_pos = ts.Timer(foo_pos, count=len(pos_rng), use_dual=True)
        t_neg = ts.Timer(foo_neg, count=len(neg_rng), use_dual=True)
        out_pos = ts.Negate(t_pos)
        out_neg = ts.Negate(t_neg)
        assert ts.run(out_pos) == expected_pos
        assert ts.run(out_neg) == expected_neg

    def test_Invert(self):
        """
        f = 1/x
        f' = -x^-2
        """
        expected_pos = [(1 / x, -1 * x ** (-2)) for x in pos_rng]
        expected_neg = [(1 / x, -1 * x ** (-2)) for x in neg_rng]
        t_pos = ts.Timer(foo_pos, count=len(pos_rng), use_dual=True)
        t_neg = ts.Timer(foo_neg, count=len(neg_rng), use_dual=True)
        out_pos = ts.Invert(t_pos)
        out_neg = ts.Invert(t_neg)
        assert ts.run(out_pos) == expected_pos
        assert ts.run(out_neg) == expected_neg

    def test_Add(self):
        """
        f = x+x
        f' = 2
        """
        expected = [(x + x, 2) for x in rng]
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Add(t, t)
        assert ts.run(out) == expected

    def test_Sub(self):
        """
        f = x-x
        f' = 0
        """
        expected = [(x - x, 0) for x in rng]
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Sub(t, t)
        assert ts.run(out) == expected

    def test_Mult(self):
        """
        f = x*x
        f' = 2x
        """
        expected = [(x * x, 2 * x) for x in rng]
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Mult(t, t)
        assert ts.run(out) == expected

    def test_Div(self):
        """
        f = x/x
        f' = 0
        """
        expected = [(1, 0) for x in pos_rng]
        t = ts.Timer(foo_pos, count=len(pos_rng), use_dual=True)
        out = ts.Div(t, t)
        assert ts.run(out) == expected

    def test_RDiv(self):
        """
        f = x/x
        f' = 0
        """
        expected = [(1, 0) for x in pos_rng]
        t = ts.Timer(foo_pos, count=len(pos_rng), use_dual=True)
        out = ts.RDiv(t, t)
        assert ts.run(out) == expected

    def test_Pow(self):
        """
        f = x^2
        f' = 2x
        """
        expected = [(x ** 2, 2 * x) for x in rng]
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Pow(t, 2)
        assert ts.run(out) == expected

    def test_Sum(self):
        """
        f = x+x+2
        f' = 2
        """
        expected = [(x + x + 2, 2) for x in rng]
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        t2 = ts.Timer(foo_range, count=len(rng), use_dual=True)
        c = ts.Const((2, 0), use_dual=True)
        out = ts.Sum(t, t2, c)
        assert ts.run(out) == expected

    def test_Average(self):
        """
        f = (x + x + 1)/3
        f' = 2/3
        """
        expected = [((x + x + 1) / 3, 2 / 3) for x in rng]
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        t2 = ts.Timer(foo_range, count=len(rng), use_dual=True)
        c = ts.Const((1, 0), use_dual=True)
        out = ts.Average(t, t2, c)
        assert ts.run(out) == expected

    def test_Not(self):
        t = ts.Timer(foo_range, count=2, use_dual=True)
        out = ts.Not(t)
        assert ts.run(out) == [False, False]

    def test_And(self):
        t = ts.Timer(foo_range, count=2, use_dual=True)
        out = ts.And(t, t)
        assert ts.run(out) == [(-10, 1), (-9, 1)]

    def test_Or(self):
        t = ts.Timer(foo_range, count=2, use_dual=True)
        out = ts.Or(t, t)
        assert ts.run(out) == [(-10, 1), (-9, 1)]

    def test_Equal(self):
        t = ts.Timer(foo_range, count=2, use_dual=True)
        out = ts.Equal(t, t)
        assert ts.run(out) == [True, True]

    def test_NotEqual(self):
        t = ts.Timer(foo_range, count=2, use_dual=True)
        c = ts.Const((-10, 1), use_dual=True)
        out = ts.NotEqual(t, c)
        assert ts.run(out) == [False, True]

    def test_Lt(self):
        t = ts.Timer(foo_range, count=2, use_dual=True)
        c = ts.Const((1, 1), use_dual=True)
        out = ts.Lt(c, t)
        assert ts.run(out) == [False, False]

    def test_Le(self):
        t = ts.Timer(foo_range, count=2, use_dual=True)
        c = ts.Const((-9, 1), use_dual=True)
        out = ts.Le(c, t)
        assert ts.run(out) == [False, True]

    def test_Gt(self):
        t = ts.Timer(foo_range, count=2, use_dual=True)
        c = ts.Const((-9, 1), use_dual=True)
        out = ts.Gt(t, c)
        assert ts.run(out) == [False, False]

    def test_Ge(self):
        t = ts.Timer(foo_range, count=2, use_dual=True)
        c = ts.Const((-9, 1), use_dual=True)
        out = ts.Ge(t, c)
        assert ts.run(out) == [False, True]

    def test_Log(self):
        """
        f = ln(x)
        f' = 1/x
        """
        expected = [(math.log(x), 1 / x) for x in pos_rng]
        t = ts.Timer(foo_pos, count=len(pos_rng), use_dual=True)
        print(t)
        out = ts.Log(t)
        assert ts.run(out) == expected

    def test_Sin(self):
        """
        f = sin(x)
        f' = cos(x)
        """
        expected = [(math.sin(x), math.cos(x)) for x in rng]
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Sin(t)
        assert ts.run(out) == expected

    def test_Cos(self):
        """
        f = cos(x)
        f' = -sin(x)
        """
        expected = [(math.cos(x), -1 * math.sin(x)) for x in rng]
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Cos(t)
        assert ts.run(out) == expected

    def test_Tan(self):
        """
        f = tan(x)
        f' = (1/cos(x))^2
        """
        expected = [(math.tan(x), (1 / math.cos(x)) ** 2) for x in rng]
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Tan(t)
        assert ts.run(out) == expected

    def test_Arcsin(self):
        """
        f = arcsin(x)
        f' = 1/sqrt(1-x^2)
        """
        expected = [(math.asin(x), 0.05 / math.sqrt(1 - x ** 2)) for x in zero_one_rng]
        t = ts.Timer(foo_zero_one, count=len(zero_one_rng), use_dual=True)
        out = ts.Arcsin(t)
        assert ts.run(out) == expected

    def test_Arccos(self):
        """
        f = arccos(x)
        f' = -1/sqrt(1-x^2)
        """
        expected = [(math.acos(x), -0.05 / math.sqrt(1 - x ** 2)) for x in zero_one_rng]
        t = ts.Timer(foo_zero_one, count=len(zero_one_rng), use_dual=True)
        out = ts.Arccos(t)
        assert ts.run(out) == expected

    def test_Arctan(self):
        """
        f = arctan(x)
        f' = 1/(1+x^2)
        """
        expected = [(math.atan(x), 0.05 / (1 + x ** 2)) for x in zero_one_rng]
        t = ts.Timer(foo_zero_one, count=len(zero_one_rng), use_dual=True)
        out = ts.Arctan(t)
        assert ts.run(out) == expected

    def test_Sqrt(self):
        """
        f = sqrt(x)
        f' = 0.5/sqrt(x)
        """
        expected = [(math.sqrt(x), 0.5 / math.sqrt(x)) for x in pos_rng]
        t = ts.Timer(foo_pos, count=len(pos_rng), use_dual=True)
        out = ts.Sqrt(t)
        assert ts.run(out) == expected

    def test_Abs(self):
        """
        f = abs(x)
        f' = x/abs(x)
        """
        expected = [(abs(x), x / abs(x)) for x in neg_rng]
        t = ts.Timer(foo_neg, count=len(neg_rng), use_dual=True)
        out = ts.Abs(t)
        assert ts.run(out) == expected

    def test_Exp(self):
        """
        f = exp(x)
        f' = exp(x)
        """
        expected = [(math.exp(x), math.exp(x)) for x in rng]
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Exp(t)
        assert ts.run(out) == expected

    def test_Erf(self):
        """
        f = erf(x)
        f' = (2/sqrt(pi))*e^(-x^2)
        """
        expected = [
            (math.erf(x), (2 / math.sqrt(math.pi)) * math.exp(-(x ** 2))) for x in rng
        ]
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Erf(t)
        assert ts.run(out) == expected

    def test_Floor(self):
        expected = [(math.floor(x), math.floor(0.05)) for x in zero_one_rng]
        t = ts.Timer(foo_zero_one, count=len(zero_one_rng), use_dual=True)
        out = ts.Floor(t)
        assert ts.run(out) == expected

    def test_Ceil(self):
        expected = [(math.ceil(x), math.ceil(0.05)) for x in zero_one_rng]
        t = ts.Timer(foo_zero_one, count=len(zero_one_rng), use_dual=True)
        out = ts.Ceil(t)
        assert ts.run(out) == expected

    def test_Round(self):
        expected = [(round(x, ndigits=1), round(0.05, ndigits=1)) for x in zero_one_rng]
        t = ts.Timer(foo_zero_one, count=len(zero_one_rng), use_dual=True)
        out = ts.Round(t, 1)
        assert ts.run(out) == expected

    def test_Int(self):
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Int(t)
        assert ts.run(out) == list(rng)

    def test_Float(self):
        t = ts.Timer(foo_range, count=len(rng), use_dual=True)
        out = ts.Float(t)
        assert ts.run(out) == [float(x) for x in rng]

    def test_Bool(self):
        t = ts.Timer(foo_range, count=2, use_dual=True)
        out = ts.Bool(t)
        assert ts.run(out) == [True, True]

    def test_Str(self):
        t = ts.Timer(foo_range, count=1, use_dual=True)
        out = ts.Str(t)
        assert ts.run(out) == ["-10+1ε"]

    def test_Len(self):
        t = ts.Timer(foo_range, count=1, use_dual=True)
        out = ts.Len(t)
        assert ts.run(out) == [2]
