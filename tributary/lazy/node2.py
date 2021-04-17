import inspect
import uuid
from collections import namedtuple


def extractParameters(callable):
    """Given a function, extract the arguments and defaults

    Args:
        value [callable]: a callable
    """

    # TODO handle generators as lambda g=g: next(g)
    if inspect.isgeneratorfunction(callable):
        raise NotImplementedError()

    # wrap args and kwargs of function to node
    try:
        signature = inspect.signature(callable)

    except ValueError:
        # https://bugs.python.org/issue20189
        signature = namedtuple("Signature", ["parameters"])({})

    # extract all args. args/kwargs become tuple/dict input
    return [
        Parameter(p.name, i, p.default, p.kind)
        for i, p in enumerate(signature.parameters.values())
    ]


class Parameter(object):
    def __init__(self, name, position, default, kind):
        self.name = name
        self.position = position
        self.kind = kind

        if kind == inspect._ParameterKind.VAR_POSITIONAL:
            # default is empty tuple
            self.default = tuple()

        elif kind == inspect._ParameterKind.VAR_KEYWORD:
            # default is empty dict
            self.default = {}
        else:
            # default can be inspect._empty
            self.default = default


class CallingCtx(object):
    def __init__(self, base, value, args, kwargs):
        """Create a calling context from callable and args

        Args:
            base (Node): the base node attached to the callable
            value [callable]: A callable
            args [list]: a list of nodes representing parameters
            kwargs [dict]: a dict of name:node representing parameters
        """
        if callable(value):
            self.value = value
        else:
            self.value = lambda: value

        # base params
        self.parameters = extractParameters(self.value)

        # go through parameters and wrap in node, or used provided default
        self.args = []
        self.kwargs = {}
        for i, param in enumerate(self.parameters):
            if i < len(args):
                # use arg as node
                node = args[i]

            elif param.name in kwargs:
                # use kwargs as node
                node = kwargs[param.name]

            else:
                # create new node
                node = Node2(param.name, param.default)  # TODO check defaults

            # store
            self.args.append(node)
            self.kwargs[param.name] = node

        # initialize execution cache
        self.exec_cache = {}

    def dependencies(self):
        """upstream dependencies"""
        return self.kwargs

    def arg(self, position):
        """get arg at position"""
        return self.args[position]

    def kwarg(self, name):
        """get arg by keyword"""
        return self.kwargs[name]

    def changed(self, ctx):
        ...
        return True

    def __repr__(self):
        return "{} {}".format(self.callable.__name__, self.kwargs)

    def __call__(self, ctx):
        # TODO execution context
        if self.changed(ctx):
            # recalculate
            val = self.value(
                **{k: v._call_with_context(ctx) for k, v in self.kwargs.items()}
            )
            self.exec_cache[hash(ctx)] = val
            return val

        # return cached value
        return self.exec_cache[hash(ctx)]


class ExecutionCtx(object):
    def __init__(self, origin, force=False):
        self.origin = origin
        self.force = force

    def __hash__(self):
        # uniquely identify this context
        return 0


class Node2(object):
    def __init__(self, name="?", value=None, args=None, kwargs=None):
        # Name is a string for display
        self._name = name

        # ID is unique identifier of the node
        self._id = str(uuid.uuid4())

        # extract the callable into a calling context
        self._calling_ctx = CallingCtx(self, value, args, kwargs)

    def name(self):
        return self._name

    def fullname(self, truncate=True):
        if truncate:
            return "{}#{}".format(self._name, self._id[:4])
        return "{}#{}".format(self._name, self._id)

    def __repr__(self):
        return "Node[{}]".format(self.fullname(truncate=True))

    def _call_with_context(self, ctx):
        return self._calling_ctx(ctx)

    def __call__(self, *args, **kwargs):
        ctx = ExecutionCtx(origin=self)
        return self._calling_ctx(ctx)

    # def isDirty(self):
    #     return self._calling_ctx.changed()

    def dependencies(self):
        return self._calling_ctx.dependencies()

    def print(node, cache=None):
        if cache is None:
            cache = {}

        if node.fullname() in cache:
            # loop, return None
            return node

        cache[node.fullname()] = node

        ret = {node: []}

        for dep in node.dependencies().values():
            # callable node
            ret[node].append(dep.print(cache))

        return ret
