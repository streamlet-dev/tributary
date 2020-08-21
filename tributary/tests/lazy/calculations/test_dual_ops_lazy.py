import math
import tributary.lazy as tl

options = {'use_dual': True}

class TestOps:
    def test_Negate(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Negate(t)
        assert out() == (-5, -1)

    def test_Invert(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Invert(t)
        assert out() == (1/5, -5**(-2))

    def test_Add(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Add(t, t)
        assert out() == (10,2)

    def test_Sub(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Sub(t, t)
        assert out() == (0,0)

    def test_Mult(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Mult(t, t)
        assert out() == (25, 10)

    def test_Div(self):
        t = tl.Node(value=(15,1), **options)
        t2 = tl.Node(value=(5,1), **options)
        out = tl.Div(t, t2)
        assert out() == (3,-10/25)

    def test_Pow(self):
        t = tl.Node(value=(5,1), **options)
        t2 = tl.Node(value=2, **options)
        out = tl.Pow(t, t2)
        assert out() == (25,10)

    def test_Sum(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Sum(t, t, t, (3,1))
        assert out() == (18,4)

    def test_Average(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Average(t, (3,1), (1,1))
        assert out() == (3, 1)

    def test_Not(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Not(t)
        assert out() == False

    def test_And(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.And(t, t)
        assert out() == (5,1)

    def test_Or(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Or(t, t)
        assert out() == (5,1)

    def test_Equal(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Equal(t, (5,1))
        assert out()

    def test_NotEqual(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.NotEqual(t, (7,1))
        assert out()

    def test_Lt(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Lt(t, (10,1))
        assert out() == True

    def test_Le(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Le(t, (5,1))
        assert out()

    def test_Gt(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Gt(t, (5,1))
        assert out() == False

    def test_Ge(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Ge(t, (5,1))
        assert out()

    def test_Log(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Log(t)
        assert out() == (math.log(5), 1/5)

    def test_Sin(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Sin(t)
        assert out() == (math.sin(5), math.cos(5))

    def test_Cos(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Cos(t)
        assert out() == (math.cos(5), -math.sin(5))

    def test_Tan(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Tan(t)
        assert out() == (math.tan(5), (1/math.cos(5))**2)

    def test_Arcsin(self):
        t = tl.Node(value=(0.5,1), **options)
        out = tl.Arcsin(t)
        assert out() == (math.asin(0.5), 1/math.sqrt(1-0.5**2))

    def test_Arccos(self):
        t = tl.Node(value=(0.5,1), **options)
        out = tl.Arccos(t)
        assert out() == (math.acos(0.5), -1/math.sqrt(1-0.5**2))

    def test_Arctan(self):
        t = tl.Node(value=(0.5,1), **options)
        out = tl.Arctan(t)
        assert out() == (math.atan(0.5), 1/(1+0.5**2))

    def test_Sqrt(self):
        t = tl.Node(value=(9,1), **options)
        out = tl.Sqrt(t)
        assert out() == (3, 0.5/math.sqrt(9))

    def test_Abs(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Abs(t)
        assert out() == (abs(5), 5/abs(5))

    def test_Exp(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Exp(t)
        assert out() == (math.exp(5), math.exp(5))

    def test_Floor(self):
        t = tl.Node(value=(5.333,1.333), **options)
        out = tl.Floor(t)
        assert out() == (math.floor(5.333), math.floor(1.333))

    def test_Ceil(self):
        t = tl.Node(value=(5.333,1.333), **options)
        out = tl.Ceil(t)
        assert out() == (math.ceil(5.333), math.ceil(1.333))

    def test_Round(self):
        t = tl.Node(value=(5.333,1.333), **options)
        out = tl.Round(t, ndigits=2)
        assert out() == (5.33,1.33)

    def test_Int(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Int(t)
        assert out() == 5

    def test_Float(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Float(t)
        assert out() == 5.0

    def test_Bool(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Bool(t)
        assert out()

    def test_Str(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Str(t)
        assert out() == '5+1ε'

    def test_Len(self):
        t = tl.Node(value=(5,1), **options)
        out = tl.Len(t)
        assert out() == 2
