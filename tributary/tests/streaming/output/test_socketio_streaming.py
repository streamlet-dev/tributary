import tributary.streaming as ts


class TestSocketIO:
    def test_socketio(self):
        '''Test socketio streaming'''
        def foo():
            yield 'a'
            yield 'b'
            yield 'c'

        out = ts.SocketIOSink(ts.Foo(foo), url='http://localhost:8069')
        assert ts.run(out) == ['a', 'b', 'c']

