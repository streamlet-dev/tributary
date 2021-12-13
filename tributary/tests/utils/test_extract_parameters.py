import inspect
from tributary.utils import extractParameters, Parameter


def func(a, b=1, *c, **d):
    ...


class Test:
    def meth(self, a, b=1, *c, **d):
        ...


class TestLazyToStreaming:
    def test_function_parses(self):
        params = extractParameters(func)
        assert len(params) == 4

    def test_method_parses(self):
        t = Test()
        params = extractParameters(t.meth)
        assert len(params) == 4

    def test_function_all_are_parameters(self):
        params = extractParameters(func)
        assert isinstance(params[0], Parameter)
        assert isinstance(params[1], Parameter)
        assert isinstance(params[2], Parameter)
        assert isinstance(params[3], Parameter)

    def test_method_all_are_parameters(self):
        t = Test()
        params = extractParameters(t.meth)
        assert isinstance(params[0], Parameter)
        assert isinstance(params[1], Parameter)
        assert isinstance(params[2], Parameter)
        assert isinstance(params[3], Parameter)

    def test_function_names(self):
        params = extractParameters(func)
        assert params[0].name == "a"
        assert params[1].name == "b"
        assert params[2].name == "c"
        assert params[3].name == "d"

    def test_method_names(self):
        t = Test()
        params = extractParameters(t.meth)
        assert params[0].name == "a"
        assert params[1].name == "b"
        assert params[2].name == "c"
        assert params[3].name == "d"

    def test_function_defaults(self):
        params = extractParameters(func)
        assert params[0].default == inspect._empty
        assert params[1].default == 1
        assert params[2].default == ()
        assert params[3].default == {}

    def test_method_defaults(self):
        t = Test()
        params = extractParameters(t.meth)
        assert params[0].default == inspect._empty
        assert params[1].default == 1
        assert params[2].default == ()
        assert params[3].default == {}
