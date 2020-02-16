import gc
import six
import inspect
from ..utils import _either_type


class BaseNode(object):
    '''Class to represent an operation that is lazy'''
    def __init__(self,
                 name="?",
                 derived=False,
                 readonly=False,
                 nullable=False,
                 value=None,
                 callable=None,
                 callable_args=None,
                 callable_kwargs=None,
                 callable_is_method=False,
                 always_dirty=False,
                 trace=False,
                 ):
        '''Construct a new lazy node, wrapping a callable or a value

        Args:
            name (str): name to use to represent the node
            derived (bool):
            readonly (bool): whether a node is settable
            nullable (bool): whether a node can have value None
            value (any): initial value of the node
            callable (callable): function or other callable that the node is wrapping
            callable_args (tuple): args for the wrapped callable
            callable_kwargs (dict): kwargs for the wrapped callable
            callable_is_method (bool): is the callable a method of an object
            always_dirty (bool): node should not be lazy - always access underlying value
            trace (bool): trace when the node is called
        '''
        self._name = name
        self._callable = callable
        self._value = value
        self._readonly = readonly
        self._trace = trace
        self._callable = callable
        self._callable_args = self._transform_args(callable_args or [])
        self._callable_kwargs = self._transform_kwargs(callable_kwargs or {})
        self._callable_is_method = callable_is_method
        self._always_dirty = always_dirty

        self._parents = []
        self._self_reference = self

        # cache node operations that have already been done
        self._node_op_cache = {}

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

                    # Set yourself as parent
                    if self not in arg._parents:
                        arg._parents.append(self)

                # recompute kwargs
                for kwarg in six.itervalues(deps[1]):
                    kwarg._recompute()

                    # Set yourself as parent
                    if self not in kwarg._parents:
                        kwarg._parents.append(self)

            k = list(self._dependencies.keys())[0]

            if self._callable_is_method:
                new_value = k(self._self_reference, *self._dependencies[k][0], **self._dependencies[k][1])
            else:
                new_value = k(*self._dependencies[k][0], **self._dependencies[k][1])

            if isinstance(new_value, BaseNode):
                k._node_wrapper = new_value
                new_value = new_value()  # get value

            if self._trace:
                if new_value != self._value:
                    print('recomputing: %s#%d' % (self._name, id(self)))
            self._value = new_value
        return self._value

    def _subtree_dirty(self):
        for call, deps in six.iteritems(self._dependencies):
            # callable node
            if hasattr(call, '_node_wrapper') and \
               call._node_wrapper is not None:
                if call._node_wrapper._dirty or call._node_wrapper._subtree_dirty() or call._node_wrapper._always_dirty:
                    # CRITICAL
                    # always set self to be dirty if subtree is dirty
                    self._dirty = True
                    return True

            # check args
            for arg in deps[0]:
                if arg._dirty or arg._subtree_dirty() or arg._always_dirty:
                    # CRITICAL
                    # always set self to be dirty if subtree is dirty
                    self._dirty = True
                    return True

            # check kwargs
            for kwarg in six.itervalues(deps[1]):
                if kwarg._dirty or kwarg._subtree_dirty() or kwarg._always_dirty:
                    # CRITICAL
                    # always set self to be dirty if subtree is dirty
                    self._dirty = True
                    return True
        return False

    def _recompute(self):
        self._dirty = self._dirty or self._subtree_dirty() or self._always_dirty
        if self._dirty:
            if self._parents:
                for parent in self._parents:
                    # let your parents know you were dirty!
                    parent._dirty = True
            self._value = self._compute_from_dependencies()
        self._dirty = False

    def _gennode(self, name, foo, foo_args, trace=False):
        if name not in self._node_op_cache:
            self._node_op_cache[name] = \
                BaseNode(name=name,
                      derived=True,
                      callable=foo,
                      callable_args=foo_args,
                      trace=trace)
        return self._node_op_cache[name]

    def _tonode(self, other, trace=False):
        if isinstance(other, BaseNode):
            return other
        if str(other) not in self._node_op_cache:
            self._node_op_cache[str(other)] = \
                BaseNode(name='var(' + str(other)[:5] + ')',
                      derived=True,
                      value=other,
                      trace=trace)
        return self._node_op_cache[str(other)]

    def setValue(self, value):
        if value != self._value:
            self._dirty = True
        self._value = value

    def set(self, *args, **kwargs):
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

    def _print(self, counter=0, cache=None):
        if cache is None:
            cache = {}

        key = cache.get(id(self), str(self) + ' (#' + str(counter) + ')')
        cache[id(self)] = key

        if self._dirty or self._subtree_dirty() or self._always_dirty:
            key += '(dirty)'

        ret = {key: []}
        counter += 1

        if self._dependencies:
            for call, deps in six.iteritems(self._dependencies):
                # callable node
                if hasattr(call, '_node_wrapper') and \
                   call._node_wrapper is not None:
                    val, counter = call._node_wrapper._print(counter, cache)
                    ret[key].append(val)

                # args
                for arg in deps[0]:
                    val, counter = arg._print(counter, cache)
                    ret[key].append(val)

                # kwargs
                for kwarg in six.itervalues(deps[1]):
                    val, counter = kwarg._print(counter, cache)
                    ret[key].append(val)

        return ret, counter

    def print(self):
        return self._print(0, {})[0]

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
                    if '(dirty)' in d:
                        dot.node(d.replace('(dirty)', ''), color='red')
                        dot.edge(d.replace('(dirty)', ''), parent.replace('(dirty)', ''), color='red')
                    else:
                        dot.node(d)
                        dot.edge(d, parent.replace('(dirty)', ''))
                else:
                    for k in d:
                        if '(dirty)' in k:
                            dot.node(k.replace('(dirty)', ''), color='red')
                            rec(d[k], k)
                            dot.edge(k.replace('(dirty)', ''), parent.replace('(dirty)', ''), color='red')
                        else:
                            dot.node(k)
                            rec(d[k], k)
                            dot.edge(k, parent.replace('(dirty)', ''))

        for k in d:
            if '(dirty)' in k:
                dot.node(k.replace('(dirty)', ''), color='red')
            else:
                dot.node(k)
            rec(d[k], k)
        return dot

    def networkx(self):
        d = self.graph()
        # FIXME deduplicate
        from pygraphviz import AGraph
        import networkx as nx
        dot = AGraph(strict=True, directed=True)

        def rec(nodes, parent):
            for d in nodes:
                if not isinstance(d, dict):
                    if '(dirty)' in d:
                        d = d.replace('(dirty)', '')
                        dot.add_node(d, label=d, color='red')
                        dot.add_edge(d, parent, color='red')
                    else:
                        dot.add_node(d, label=d)
                        dot.add_edge(d, parent)
                else:
                    for k in d:
                        if '(dirty)' in k:
                            k = k.replace('(dirty)', '')
                            dot.add_node(k, label=k, color='red')
                            rec(d[k], k)
                            dot.add_edge(k, parent, color='red')
                        else:
                            dot.add_node(k, label=k)
                            rec(d[k], k)
                            dot.add_edge(k, parent)

        for k in d:
            dot.add_node(k, label=k)
            rec(d[k], k)
        return nx.nx_agraph.from_agraph(dot)

    def __call__(self):
        self._recompute()
        return self.value()

    def __repr__(self):
        return self._name


@_either_type
def node(meth, memoize=True, trace=False):
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
            value = argspec.defaults[0]
            nullable = True
        else:
            value = None
            nullable = False

        node_args.append(BaseNode(name=arg,
                               derived=True,
                               readonly=False,
                               nullable=nullable,
                               value=value,
                               trace=trace))

    for k, v in six.iteritems(argspec.kwonlydefaults or {}):
        node_kwargs[k] = BaseNode(name=k,
                               derived=True,
                               readonly=False,
                               nullable=True,
                               value=v,
                               trace=trace)

    def meth_wrapper(self, *args, **kwargs):
        if len(args) > len(node_args):
            raise Exception('Missing args (call or preprocessing error has occurred)')

        if len(kwargs) > len(node_kwargs):
            raise Exception('Missing kwargs (call or preprocessing error has occurred)')

        if is_method:
            val = meth(self, *(arg.value() for arg in args), **kwargs)
        else:
            val = meth(*(arg.value() for arg in args), **kwargs)
        return val

    new_node = BaseNode(name=meth.__name__,
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
