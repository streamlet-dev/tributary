class TestBase:
    def test_1(self):
        from tributary.base import StreamNone

        assert not StreamNone()
