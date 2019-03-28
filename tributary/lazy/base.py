import six
import math
import inspect
import functools
import numpy as np
import scipy as sp


class Node(object):
    def __init__(self,
                 name,
                 derived=False,
                 readonly=False,
                 nullable=False,
                 default_or_starting_value=None,
                 callable=None,
                 callable_args=None,
                 callable_kwargs=None,
                 callable_is_method=False,
                 always_dirty=False,
                 trace=False,
                 ):
        self._callable = callable
        self._name = name
        self._value = default_or_starting_value
        self._readonly = readonly
        self._trace = trace
        self._callable = callable
        self._callable_args = self._transform_args(callable_args or [])
        self._callable_kwargs = self._transform_kwargs(callable_kwargs or {})
        self._callable_is_method = callable_is_method
        self._always_dirty = always_dirty

        self._self_reference = self

        if callable:
            self._callable._node_wrapper = None  # not known until program start
            self._dependencies = {self._callable: (self._callable_args, self._callable_kwargs)}
        else:
            self._dependencies = {}

        if derived:
            self._dirty = True
        else:
            self._dirty = False

    def _transform_args(self, args):
        return args

    def _transform_kwargs(self, kwargs):
        return kwargs

    def _with_self(self, other_self):
        self._self_reference = other_self
        return self

    def _compute_from_dependencies(self):
        if self._dependencies:
            for deps in six.itervalues(self._dependencies):
                # recompute args
                for arg in deps[0]:
                    arg._recompute()

                # recompute kwargs
                for kwarg in six.itervalues(deps[1]):
                    kwarg._recompute()

            k = list(self._dependencies.keys())[0]

            if self._callable_is_method:
                new_value = k(self._self_reference, *self._dependencies[k][0], **self._dependencies[k][1])
            else:
                new_value = k(*self._dependencies[k][0], **self._dependencies[k][1])

            if isinstance(new_value, Node):
                k._node_wrapper = new_value
                new_value = new_value()  # get value

            if self._trace:
                if new_value != self._value:
                    print('recomputing: %s#%d' % (self._name, id(self)))
            self._value = new_value
        return self._value

    def _subtree_dirty(self):
        for deps in six.itervalues(self._dependencies):
            # check args
            for arg in deps[0]:
                if arg._dirty or arg._subtree_dirty() or arg._always_dirty:
                    return True

            # check kwargs
            for kwarg in six.itervalues(deps[1]):
                if kwarg._dirty or kwarg._subtree_dirty() or kwarg._always_dirty:
                    return True
        return False

    def _recompute(self):
        self._dirty = self._dirty or self._subtree_dirty() or self._always_dirty
        if self._dirty:
            self._value = self._compute_from_dependencies()
        self._dirty = False

    @staticmethod
    def _gennode(name, foo, foo_args, trace=False):
        return Node(name=name,
                    derived=True,
                    callable=foo,
                    callable_args=foo_args,
                    trace=trace)

    @staticmethod
    def _tonode(other, trace=False):
        if isinstance(other, Node):
            return other
        return Node(name='var(' + str(other)[:5] + ')',
                    derived=True,
                    default_or_starting_value=other,
                    trace=trace)

    def set(self, **kwargs):
        for k, v in six.iteritems(kwargs):
            _set = False
            for deps in six.itervalues(self._dependencies):
                # try to set args
                for arg in deps[0]:
                    if arg._name == k:
                        arg._dirty = arg._value != v
                        arg._value = v
                        _set = True
                        break

                if _set:
                    continue

                # try to set kwargs
                for kwarg in six.itervalues(deps[1]):
                    if kwarg._name == k:
                        kwarg._dirty = kwarg._value != v
                        kwarg._value = v
                        _set = True
                        break

    def value(self):
        return self._value

    def print(self, counter=0):
        key = str(self) + ' (#' + str(counter) + ')'
        ret = {key: []}
        counter += 1
        if self._dependencies:
            for call, deps in six.iteritems(self._dependencies):
                # callable node
                if hasattr(call, '_node_wrapper') and \
                   call._node_wrapper is not None:
                    ret[key].append(call._node_wrapper.print(counter))
                    counter += 1

                # args
                for arg in deps[0]:
                    ret[key].append(arg.print(counter))
                    counter += 1

                # kwargs
                for kwarg in six.itervalues(deps[1]):
                    ret[key].append(kwarg.print(counter))
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

    def __call__(self):
        self._recompute()
        return self.value()

    def __add__(self, other):
        other = Node._tonode(other)
        if isinstance(self._self_reference, Node):
            return Node._gennode(self._name + '+' + other._name, (lambda x, y: x.value() + y.value()), [self._self_reference, other], self._trace or other._trace)
        return Node._gennode(self._name + '+' + other._name, (lambda x, y: x.value() + y.value()), [self, other], self._trace or other._trace)

    __radd__ = __add__

    def __sub__(self, other):
        other = Node._tonode(other)
        if isinstance(self._self_reference, Node):
            return Node._gennode(self._name + '-' + other._name, (lambda x, y: x.value() - y.value()), [self._self_reference, other], self._trace or other._trace)
        return Node._gennode(self._name + '-' + other._name, (lambda x, y: x.value() - y.value()), [self, other], self._trace or other._trace)

    __rsub__ = __sub__

    def __mul__(self, other):
        other = Node._tonode(other)
        if isinstance(self._self_reference, Node):
            return Node._gennode(self._name + '*' + other._name, (lambda x, y: x.value() * y.value()), [self._self_reference, other], self._trace or other._trace)
        return Node._gennode(self._name + '*' + other._name, (lambda x, y: x.value() * y.value()), [self, other], self._trace or other._trace)

    __rmul__ = __mul__

    def __div__(self, other):
        other = Node._tonode(other)
        if isinstance(self._self_reference, Node):
            return Node._gennode(self._name + '/' + other._name, (lambda x, y: x.value() / y.value()), [self._self_reference, other], self._trace or other._trace)
        return Node._gennode(self._name + '/' + other._name, (lambda x, y: x.value() / y.value()), [self, other], self._trace or other._trace)

    __rdiv__ = __div__

    def __truediv__(self, other):
        other = Node._tonode(other)
        if isinstance(self._self_reference, Node):
            return Node._gennode(self._name + '/' + other._name, (lambda x, y: x.value() / y.value()), [self._self_reference, other], self._trace or other._trace)
        return Node._gennode(self._name + '/' + other._name, (lambda x, y: x.value() / y.value()), [self, other], self._trace or other._trace)

    __rtruediv__ = __truediv__

    def __pow__(self, other):
        other = Node._tonode(other)
        if isinstance(self._self_reference, Node):
            return Node._gennode(self._name + '^' + other._name, (lambda x, y: x.value() ** y.value()), [self._self_reference, other], self._trace or other._trace)
        return Node._gennode(self._name + '^' + other._name, (lambda x, y: x.value() ** y.value()), [self, other], self._trace or other._trace)

    def sin(self):
        return Node._gennode('sin(' + self._name + ')', (lambda x: math.sin(self.value())), [self], self._trace)

    def tan(self):
        return Node._gennode('tan(' + self._name + ')', (lambda x: math.tan(self.value())), [self], self._trace)

    def sqrt(self):
        return Node._gennode('sqrt(' + self._name + ')', (lambda x: math.sqrt(self.value())), [self], self._trace)

    def log(self):
        return Node._gennode('log(' + self._name + ')', (lambda x: math.log(self.value())), [self], self._trace)

    def exp(self):
        return Node._gennode('exp(' + self._name + ')', (lambda x: math.exp(self.value())), [self], self._trace)

    def erf(self):
        return Node._gennode('erf(' + self._name + ')', (lambda x: math.erf(self.value())), [self], self._trace)

    def __float__(self):
        return Node._gennode('float(' + self._name + ')', (lambda x: float(self.value())), [self], self._trace)

    def __int__(self):
        return Node._gennode('int(' + self._name + ')', (lambda x: int(self.value())), [self], self._trace)

    def __len__(self):
        return Node._gennode('len(' + self._name + ')', (lambda x: len(self.value())), [self], self._trace)

    def __getitem__(self):
        import ipdb; ipdb.set_trace()

    # def __array__(self, *args):
    #     import ipdb; ipdb.set_trace()
    #     return self.value()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc == np.add:
            if isinstance(inputs[0], Node): return inputs[0].__add__(inputs[1])
            else: return inputs[1].__add__(inputs[0])
        elif ufunc == np.subtract:
            if isinstance(inputs[0], Node): return inputs[0].__sub__(inputs[1])
            else: return inputs[1].__sub__(inputs[0])
        elif ufunc == np.divide:
            if isinstance(inputs[0], Node): return inputs[0].__truedivide__(inputs[1])
            else: return inputs[1].__truedivide__(inputs[0])
        elif ufunc == np.multiply:
            if isinstance(inputs[0], Node): return inputs[0].__mul__(inputs[1])
            else: return inputs[1].__mul__(inputs[0])

        elif ufunc == np.exp:
            return inputs[0].exp()
        elif ufunc == sp.special.erf:
            return inputs[0].erf()
        else:
            import ipdb; ipdb.set_trace()

    def __neg__(self):
        return Node._gennode('(-' + self._name + ')', (lambda x: -self.value()), [self], self._trace)

    def __bool__(self):
        if self.value() is None:
            return False
        return self.value()

    __nonzero__ = __bool__  # Py2 compat

    def __eq__(self, other):
        if isinstance(other, Node) and super(Node, self).__eq__(other):
            return True

        other = Node._tonode(other)
        return Node._gennode(self._name + '==' + other._name, (lambda x, y: x() == y()), [self._self_reference, other], self._trace or other._trace)

    def __ne__(self, other):
        if isinstance(other, Node) and super(Node, self).__eq__(other):
            return False

        other = Node._tonode(other)
        return Node._gennode(self._name + '!=' + other._name, (lambda x, y: x() != y()), [self, other], self._trace or other._trace)

    def __ge__(self, other):
        if isinstance(other, Node) and super(Node, self).__eq__(other):
            return True

        other = Node._tonode(other)
        return Node._gennode(self._name + '>=' + other._name, (lambda x, y: x() >= y()), [self, other], self._trace or other._trace)

    def __gt__(self, other):
        if isinstance(other, Node) and super(Node, self).__eq__(other):
            return False

        other = Node._tonode(other)
        return Node._gennode(self._name + '>' + other._name, (lambda x, y: x() > y()), [self, other], self._trace or other._trace)

    def __le__(self, other):
        if isinstance(other, Node) and super(Node, self).__eq__(other):
            return True

        other = Node._tonode(other)
        return Node._gennode(self._name + '<=' + other._name, (lambda x, y: x() <= y()), [self, other], self._trace or other._trace)

    def __lt__(self, other):
        if isinstance(other, Node) and super(Node, self).__eq__(other):
            return False
        other = Node._tonode(other)
        return Node._gennode(self._name + '<' + other._name, (lambda x, y: x() < y()), [self, other], self._trace or other._trace)

    def __repr__(self):
        return self._name


class BaseClass(object):
    def __init__(self, *args, **kwargs):
        pass

    def node(self, name, readonly=False, nullable=True, default_or_starting_value=None, trace=False):
        if not hasattr(self, '_BaseClass__nodes'):
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
                node._dirty = node._value != value
                node._value = value
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
def node(meth, memoize=False, trace=False):
    argspec = inspect.getfullargspec(meth)
    # args = argspec.args
    # varargs = argspec.varargs
    # varkw = argspec.varkw
    # defaults = argspec.defaults
    # kwonlyargs = argspec.kwonlyargs
    # kwonlydefaults = args.kwonlydefaults

    if argspec.varargs:
        raise Exception('varargs not supported yet!')
    if argspec.varkw:
        raise Exception('varargs not supported yet!')

    node_args = []
    node_kwargs = {}
    is_method = False

    for i, arg in enumerate(argspec.args):
        if arg == 'self':
            # TODO
            is_method = True
            continue

        if (is_method and len(argspec.defaults or []) >= i) or \
           (not is_method and len(argspec.defaults or []) > i):
            default_or_starting_value = argspec.defaults[0]
            nullable = True
        else:
            default_or_starting_value = None
            nullable = False

        node_args.append(Node(name=arg,
                              derived=True,
                              readonly=False,
                              nullable=nullable,
                              default_or_starting_value=default_or_starting_value,
                              trace=trace))

    for k, v in six.iteritems(argspec.kwonlydefaults or {}):
        node_kwargs[k] = Node(name=k,
                              derived=True,
                              readonly=False,
                              nullable=True,
                              default_or_starting_value=v,
                              trace=trace)

    def meth_wrapper(self, *args, **kwargs):
        if len(args) > len(node_args):
            raise Exception('Missing args (call or preprocessing error has occurred)')

        if len(kwargs) > len(node_kwargs):
            raise Exception('Missing kwargs (call or preprocessing error has occurred)')

        # move to __call__
        # # update args
        # for i, value in args:
        #     node = node_args[i]
        #     if node._value != value:
        #         node._dirty = True
        #     node._value = value

        # # update kwargs
        # for k, value in six.iteritems(kwargs):
        #     node = node_kwargs[k]
        #     if node._value != value:
        #         node._dirty = True
        #     node._value = value

        if is_method:
            val = meth(self, *(arg.value() for arg in args), **kwargs)
        else:
            val = meth(*(arg.value() for arg in args), **kwargs)
        return val

    new_node = Node(name=meth.__name__,
                    derived=True,
                    callable=meth_wrapper,
                    callable_args=node_args,
                    callable_kwargs=node_kwargs,
                    callable_is_method=is_method,
                    always_dirty=not memoize,
                    trace=trace)

    if is_method:
        ret = lambda self, *args, **kwargs: new_node._with_self(self)  # noqa: E731
    else:
        ret = lambda *args, **kwargs: new_node  # noqa: E731

    ret._node_wrapper = new_node
    return ret
