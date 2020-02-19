from ..base import Node


class Count(Node):
    '''Node to count inputs'''

    def __init__(self, node, text=''):
        self._count = 0

        def foo(val):
            self._count += 1
            return self._count

        super().__init__(foo=foo, foo_kwargs=None, name='Count', inputs=1)

        node._downstream.append((self, 0))
        self._upstream.append(node)


class Sum(Node):
    '''Node to sum inputs

    If stream type is iterable, will do += sum(input). If input
    stream type is not iterable, will do += input.
    '''

    def __init__(self, node, text=''):
        self._sum = 0

        def foo(val):
            try:
                # iterable, sum with sum function
                iter(val)
                self._sum += sum(val)
            except TypeError:
                # not iterable, sum by value
                self._sum += val
            return self._sum

        super().__init__(foo=foo, foo_kwargs=None, name='Sum', inputs=1)

        node._downstream.append((self, 0))
        self._upstream.append(node)
