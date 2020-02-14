
class TestBase:
    def setup(self):
        pass
        # setup() before each test method

    def test_1(self):
        from tributary.base import StreamNone
        assert not StreamNone('test')
