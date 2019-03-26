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

    def print(self, counter=0):
        key = str(self) + '-' + str(counter)
        ret = {key: []}
        counter += 1
        if self._dependencies:
            for deps in six.itervalues(self._dependencies):
                for dep in deps:
                    ret[key].append(dep.print(counter))
                    counter += 1
        return ret

    def graph(self):
        return self.print()

    def graphviz(self):
        d = self.graph()
        from graphviz import Digraph
        dot = Digraph(self._name, strict=True)
        dot.format = 'png'

        def rec(nodes, parent):
            for d in nodes:
                if not isinstance(d, dict):
                    dot.node(d)
                    dot.edge(d, parent)

                else:
                    for k in d:
                        dot.node(k)
                        rec(d[k], k)
                        dot.edge(k, parent)

        for k in d:
            dot.node(k)
            rec(d[k], k)
        return dot

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
                print('recomputing: %s' % (self._name))
            self._value = self._compute_from_dependencies()
        self._dirty = False

    def _gennode(self, other, name, foo, foo_args):
        return Node(name='{name}_{lhs}_{rhs}'.format(name=name, lhs=self._name, rhs=other._name),
                    derived=True,
                    dependencies={foo: foo_args},
                    trace=self._trace or other._trace)

    def _tonode(self, other):
        if isinstance(other, Node):
            return other
        return Node(name='gen_' + str(other)[:5],
                    derived=True,
                    dependencies={},
                    trace=self._trace)

    def __add__(self, other):
        other = self._tonode(other)
        return self._gennode(other, 'add', (lambda x: x[0]._value + x[1]._value), [self, other])

    def __sub__(self, other):
        other = self._tonode(other)
        return self._gennode(other, 'sub', (lambda x: x[0]._value - x[1]._value), [self, other])

    def __bool__(self):
        self._recompute()
        return self._value

    __nonzero__ = __bool__  # Py2 compat

    def __eq__(self, other):
        if isinstance(other, Node) and super(Node, self).__eq__(other):
            return True

        other = self._tonode(other)
        return self._gennode(other, 'eq', (lambda x: x[0]._value == x[1]._value), [self, other])

    def __ne__(self, other):
        other = self._tonode(other)
        return self._gennode(other, 'ne', (lambda x: x[0]._value != x[1]._value), [self, other])

    def __ge__(self, other):
        other = self._tonode(other)
        return self._gennode(other, 'ge', (lambda x: x[0]._value >= x[1]._value), [self, other])

    def __gt__(self, other):
        other = self._tonode(other)
        return self._gennode(other, 'gt', (lambda x: x[0]._value > x[1]._value), [self, other])

    def __le__(self, other):
        other = self._tonode(other)
        return self._gennode(other, 'le', (lambda x: x[0]._value <= x[1]._value), [self, other])

    def __lt__(self, other):
        other = self._tonode(other)
        return self._gennode(other, 'lt', (lambda x: x[0]._value < x[1]._value), [self, other])

    def __repr__(self):
        return '%s' % (self._name)


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
            if isinstance(value, Node) and node == value:
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
