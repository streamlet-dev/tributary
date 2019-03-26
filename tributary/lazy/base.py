import six
import functools


class Node(object):
    def __init__(self,
                 name,
                 derived=False,
                 dependencies=None,
                 readonly=False,
                 nullable=False,
                 default_or_starting_value=None,
                 trace=False,
                 ):
        self._name = name
        self._value = default_or_starting_value
        self._readonly = readonly
        self._trace = trace
        self._dependencies = dependencies or {}

        if derived:
            self._dirty = True
        else:
            self._dirty = False

    def value(self):
        self._recompute()
        return self._value

    def _compute_from_dependencies(self):
        if self._dependencies:
            for deps in six.itervalues(self._dependencies):
                for dep in deps:
                    dep._recompute()
            k = list(self._dependencies.keys())[0]
            self._value = k(self._dependencies[k])
        return self._value

    def _subtree_dirty(self):
        for deps in six.itervalues(self._dependencies):
            for dep in deps:
                if dep._dirty or dep._subtree_dirty():
                    return True
        return False

    def _recompute(self):
        self._dirty = self._dirty or self._subtree_dirty()
        if self._dirty:
            if self._trace:
                print('recomputing: %s-%d' % (self._name, id(self)))
            self._value = self._compute_from_dependencies()
        self._dirty = False

    def __add__(self, other):
        return Node(name='_add_{lhs}_{rhs}'.format(lhs=self._name, rhs=other._name),
                    derived=True,
                    dependencies={(lambda x: x[0]._value + x[1]._value): [self, other]},
                    trace=self._trace or other._trace)

    def __sub__(self, other):
        return Node(name='_sub_{lhs}_{rhs}'.format(lhs=self._name, rhs=other._name),
                    derived=True,
                    dependencies={(lambda x: x[0]._value - x[1]._value): [self, other]},
                    trace=self._trace or other._trace)

    def __repr__(self):
        self._recompute()
        return '%s-%d-%d' % (self._name, id(self), self._value)


class BaseClass(object):
    def __init__(self, *args, **kwargs):
        pass

    def node(self, name, readonly=False, nullable=True, default_or_starting_value=None, trace=False):
        if not hasattr(self, '__nodes'):
            self.__nodes = {}

        self.__nodes[name] = Node(name=name,
                                  derived=False,
                                  readonly=readonly,
                                  nullable=nullable,
                                  default_or_starting_value=default_or_starting_value,
                                  trace=trace)
        setattr(self, name, self.__nodes[name])

        return self.__nodes[name]

    def __getattribute__(self, name):
        if name == '_BaseClass__nodes' or name == '__nodes':
            return super(BaseClass, self).__getattribute__(name)
        elif hasattr(self, '_BaseClass__nodes') and name in super(BaseClass, self).__getattribute__('_BaseClass__nodes'):
            # return super(BaseClass, self).__getattribute__('_BaseClass__nodes')[name]._value
            return super(BaseClass, self).__getattribute__('_BaseClass__nodes')[name]
        else:
            return super(BaseClass, self).__getattribute__(name)

    def __setattr__(self, name, value):
        if hasattr(self, '_BaseClass__nodes') and name in super(BaseClass, self).__getattribute__('_BaseClass__nodes'):
            node = super(BaseClass, self).__getattribute__('_BaseClass__nodes')[name]
            if node == value:
                return
            elif isinstance(value, Node):
                raise Exception('Cannot set to node')
            else:
                node._value = value
                node._dirty = True
        else:
            super(BaseClass, self).__setattr__(name, value)


def _either_type(f):
    @functools.wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)
    return new_dec


@_either_type
def node(meth, trace=False):
    def meth_wrapper(self, *args, **kwargs):
        return meth(self, *args, **kwargs)
    return meth_wrapper
