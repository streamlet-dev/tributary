from mock import MagicMock, patch


class TestConfig:
    def setup(self):
        pass
        # setup() before each test method

    def teardown(self):
        pass
        # teardown() after each test method

    @classmethod
    def setup_class(cls):
        pass
        # setup_class() before any methods in this class

    @classmethod
    def teardown_class(cls):
        pass
        # teardown_class() after any methods in this class

    def test_grid(self):
        with patch('time.sleep') as m:
            from tributary._depr.sources.random import RandomSource, RandomSource2
            r = RandomSource()
            r2 = RandomSource2()

            def foo(arg):
                raise Exception('Test')
            m.side_effect = foo

            try:
                r.run()
            except Exception:
                pass

            try:
                r2.run()
            except Exception:
                pass
