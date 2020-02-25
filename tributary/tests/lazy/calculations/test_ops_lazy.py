import math
import tributary.lazy as tl


class TestOps:
    def test_Negate(self):
        t = tl.Node(value=5)
        out = tl.Negate(t)
        assert out() == -5

    def test_Invert(self):
        t = tl.Node(value=5)
        out = tl.Invert(t)
        assert out() == 1 / 5

    def test_Add(self):
        t = tl.Node(value=5)
        out = tl.Add(t, t)
        assert out() == 10

    def test_Sub(self):
        t = tl.Node(value=5)
        out = tl.Sub(t, t)
        assert out() == 0

    def test_Mult(self):
        t = tl.Node(value=5)
        out = tl.Mult(t, t)
        assert out() == 25

    def test_Div(self):
        t = tl.Node(value=15)
        t2 = tl.Node(value=5)
        out = tl.Div(t, t2)
        assert out() == 3

    def test_Mod(self):
        t = tl.Node(value=3)
        t2 = tl.Node(value=2)
        out = tl.Mod(t, t2)
        assert out() == 1

    def test_Pow(self):
        t = tl.Node(value=3)
        t2 = tl.Node(value=2)
        out = tl.Pow(t, t2)
        assert out() == 9

    def test_Sum(self):
        t = tl.Node(value=3)
        out = tl.Sum(t, t, t, 3)
        assert out() == 12

    def test_Average(self):
        t = tl.Node(value=3)
        out = tl.Average(t, 2, 1)
        assert out() == 2

    def test_Not(self):
        t = tl.Node(value=2)
        out = tl.Not(t)
        assert out() == False

    def test_And(self):
        t = tl.Node(value=2)
        out = tl.And(t, t)
        assert out() == 2

    def test_Or(self):
        t = tl.Node(value=2)
        out = tl.Or(t, t)
        assert out() == 2

    def test_Equal(self):
        t = tl.Node(value=2)
        out = tl.Equal(t, 2)
        assert out()

    def test_NotEqual(self):
        t = tl.Node(value=2)
        out = tl.NotEqual(t, 1)
        assert out()

    def test_Lt(self):
        t = tl.Node(value=2)
        out = tl.Lt(t, 1)
        assert out() == False

    def test_Le(self):
        t = tl.Node(value=2)
        out = tl.Le(t, 2)
        assert out()

    def test_Gt(self):
        t = tl.Node(value=2)
        out = tl.Gt(t, 2)
        assert out() == False

    def test_Ge(self):
        t = tl.Node(value=2)
        out = tl.Ge(t, 1)
        assert out()

    def test_Log(self):
        t = tl.Node(value=2)
        out = tl.Log(t)
        assert out() == math.log(2)

    def test_Sin(self):
        t = tl.Node(value=2)
        out = tl.Sin(t)
        assert out() == math.sin(2)

    def test_Cos(self):
        t = tl.Node(value=2)
        out = tl.Cos(t)
        assert out() == math.cos(2)

    def test_Tan(self):
        t = tl.Node(value=2)
        out = tl.Tan(t)
        assert out() == math.tan(2)

    def test_Arcsin(self):
        t = tl.Node(value=2)
        out = tl.Arcsin(tl.Div(t, 3))
        assert out() == math.asin(2 / 3)

    def test_Arccos(self):
        t = tl.Node(value=2)
        out = tl.Arccos(tl.Div(t, 3))
        assert out() == math.acos(2 / 3)

    def test_Arctan(self):
        t = tl.Node(value=2)
        out = tl.Arctan(t)
        assert out() == math.atan(2)

    def test_Sqrt(self):
        t = tl.Node(value=9)
        out = tl.Sqrt(t)
        assert out() == 3.0

    def test_Abs(self):
        t = tl.Node(value=-2)
        out = tl.Abs(t)
        assert out() == 2

    def test_Exp(self):
        t = tl.Node(value=2)
        out = tl.Exp(t)
        assert out() == math.exp(2)

    def test_Erf(self):
        t = tl.Node(value=2)
        out = tl.Erf(t)
        assert out() == math.erf(2)

    def test_Int(self):
        t = tl.Node(value=2.0)
        out = tl.Int(t)
        assert out() == 2

    def test_Float(self):
        t = tl.Node(value=2)
        out = tl.Float(t)
        assert out() == 2.0

    def test_Bool(self):
        t = tl.Node(value=2)
        out = tl.Bool(t)
        assert out()

    def test_Str(self):
        t = tl.Node(value=2)
        out = tl.Str(t)
        assert out() == '2'

    def test_Len(self):
        t = tl.Node(value=[1, 2, 3])
        out = tl.Len(t)
        assert out() == 3
