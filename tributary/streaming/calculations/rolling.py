from ..base import Node


class Count(Node):
    '''Node to count inputs'''

    def __init__(self, node):
        self._count = 0

        def foo(val):
            self._count += 1
            return self._count

        super().__init__(foo=foo, foo_kwargs=None, name='Count', inputs=1)

        node._downstream.append((self, 0))
        self._upstream.append(node)


class Max(Node):
    '''Node to take rolling max of inputs'''

    def __init__(self, node):
        self._max = None

        def foo(val):
            self._max = max(self._max, val) if self._max is not None else val
            return self._max

        super().__init__(foo=foo, foo_kwargs=None, name='Max', inputs=1)

        node._downstream.append((self, 0))
        self._upstream.append(node)


class Min(Node):
    '''Node to take rolling min of inputs'''

    def __init__(self, node):
        self._min = None

        def foo(val):
            self._min = min(self._min, val) if self._min is not None else val
            return self._min

        super().__init__(foo=foo, foo_kwargs=None, name='Min', inputs=1)

        node._downstream.append((self, 0))
        self._upstream.append(node)


class Sum(Node):
    '''Node to take rolling sum inputs

    If stream type is iterable, will do += sum(input). If input
    stream type is not iterable, will do += input.
    '''

    def __init__(self, node):
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


class Average(Node):
    '''Node to take the running average

    If stream type is iterable, will do (average + sum(input))/(count+len(input)).
    If input stream type is not iterable, will do (average + input)/count
    '''

    def __init__(self, node):
        self._sum = 0
        self._count = 0

        def foo(val):
            try:
                # iterable, sum with sum function
                iter(val)
                self._sum += sum(val)
                self._count += len(val)
            except TypeError:
                # not iterable, sum by value
                self._sum += val
                self._count += 1
            return self._sum / self._count if self._count > 0 else float('nan')

        super().__init__(foo=foo, foo_kwargs=None, name='Average', inputs=1)

        node._downstream.append((self, 0))
        self._upstream.append(node)


Node.rollingCount = Count
Node.rollingMin = Min
Node.rollingMax = Max
Node.rollingSum = Sum
Node.rollingAverage = Average
