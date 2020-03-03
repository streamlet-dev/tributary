import six
import inspect
from ..utils import _either_type


class Node(object):
    '''Class to represent an operation that is lazy'''
    _id_ref = 0

    def __init__(self,
                 value=None,
                 name="?",
                 derived=False,
                 readonly=False,
                 nullable=False,
                 callable=None,
                 callable_args=None,
                 callable_kwargs=None,
                 callable_is_method=False,
                 always_dirty=False,
                 trace=False,
                 **kwargs
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
        # Instances get an id but one id tracker for all nodes so we can
        # uniquely identify them
        # TODO different scheme
        self._id = Node._id_ref
        Node._id_ref += 1

        # Every node gets a name so it can be uniquely identified in the graph
        self._name = '{}#{}'.format(name or self.__class__.__name__, self._id)

        if isinstance(value, Node):
            raise Exception('Cannot set value to be itself a node')

        # Graphviz shape
        self._graphvizshape = kwargs.get('graphvizshape', 'box')  # default is box instead of ellipse
        # because all lazy nodes are i/o nodes
        # by default

        # if using dagre-d3, this will be set
        self._dd3g = None

        # starting value
        self._value = value

        # can be set
        self._readonly = readonly

        # trace calls/executions
        self._trace = trace

        # callable and args
        self._callable = callable if not inspect.isgeneratorfunction(callable) else lambda gen=callable(*(callable_args or []), **(callable_kwargs or {})): next(gen)
        self._callable_args = self._transform_args(callable_args or [])
        self._callable_kwargs = self._transform_kwargs(callable_kwargs or {})
        self._callable_is_method = callable_is_method

        # if always dirty, always reevaluate
        self._always_dirty = always_dirty

        # parent nodes in graph
        self._parents = []

        # self reference for method calls
        self._self_reference = self

        # cache node operations that have already been done
        self._node_op_cache = {}

        # dependencies can be nodes
        if self._callable:
            self._callable._node_wrapper = None  # not known until program start
            self._dependencies = {self._callable: (self._callable_args, self._callable_kwargs)}
        else:
            self._dependencies = {}

        # if derived node, default to dirty to start
        if derived:
            self._dirty = True
        else:
            self._dirty = False

    def _get_dirty(self):
        return self._is_dirty

    def _set_dirty(self, val):
        if val:
            self._reddd3g()
        else:
            self._whited3g()
        self._is_dirty = val

    _dirty = property(_get_dirty, _set_dirty)

    def _name_no_id(self):
        return self._name.rsplit('#', 1)[0]

    def _transform_args(self, args):
        return args

    def _transform_kwargs(self, kwargs):
        return kwargs

    def _with_self(self, other_self):
        self._self_reference = other_self
        return self

    def _greendd3g(self):
        if self._dd3g:
            self._dd3g.setNode(self._name, tooltip=str(self.value()), style='fill: #0f0')

    def _yellowdd3g(self):
        if self._dd3g:
            self._dd3g.setNode(self._name, tooltip=str(self.value()), style='fill: #ff0')

    def _reddd3g(self):
        if self._dd3g:
            self._dd3g.setNode(self._name, tooltip=str(self.value()), style='fill: #f00')

    def _whited3g(self):
        if self._dd3g:
            self._dd3g.setNode(self._name, tooltip=str(self.value()), style='fill: #fff')

    def _compute_from_dependencies(self):
        if self._dependencies:
            self._greendd3g()
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

            if isinstance(new_value, Node):
                k._node_wrapper = new_value
                new_value = new_value()  # get value

            if isinstance(new_value, Node):
                raise Exception('Value should not itself be a node!')

            if self._trace:
                if new_value != self._value:
                    print('recomputing: %s#%d' % (self._name, id(self)))

            self._value = new_value

        self._whited3g()
        return self._value

    def _subtree_dirty(self):
        for call, deps in six.iteritems(self._dependencies):
            # callable node
            if hasattr(call, '_node_wrapper') and \
               call._node_wrapper is not None:
                if call._node_wrapper.isDirty():
                    # CRITICAL
                    # always set self to be dirty if subtree is dirty
                    self._dirty = True
                    return True

            # check args
            for arg in deps[0]:
                if arg.isDirty():
                    # CRITICAL
                    # always set self to be dirty if subtree is dirty
                    self._dirty = True
                    return True

            # check kwargs
            for kwarg in six.itervalues(deps[1]):
                if kwarg.isDirty():
                    # CRITICAL
                    # always set self to be dirty if subtree is dirty
                    self._dirty = True
                    return True
        return False

    def isDirty(self):
        self._dirty = self._dirty or self._subtree_dirty() or self._always_dirty
        return self._dirty

    def _recompute(self):
        self.isDirty()
        if self._dirty:
            if self._parents:
                for parent in self._parents:
                    # let your parents know you were dirty!
                    parent._dirty = True
            self._value = self._compute_from_dependencies()
        self._dirty = False

    def _gennode(self, name, foo, foo_args, trace=False, **kwargs):
        if name not in self._node_op_cache:
            self._node_op_cache[name] = \
                Node(name=name,
                     derived=True,
                     callable=foo,
                     callable_args=foo_args,
                     trace=trace,
                     **kwargs)
        return self._node_op_cache[name]

    def _tonode(self, other, trace=False):
        if isinstance(other, Node):
            return other
        if str(other) not in self._node_op_cache:
            self._node_op_cache[str(other)] = \
                Node(name='var(' + str(other)[:5] + ')',
                     derived=True,
                     value=other,
                     trace=trace)
        return self._node_op_cache[str(other)]

    def setValue(self, value):
        if value != self._value:
            self._value = value  # leave for dagre
            self._dirty = True
        self._value = value

    def set(self, *args, **kwargs):
        for k, v in six.iteritems(kwargs):
            _set = False
            for deps in six.itervalues(self._dependencies):
                # try to set args
                for arg in deps[0]:
                    if arg._name_no_id() == k:
                        arg._dirty = (arg._value != v)
                        arg._value = v
                        _set = True
                        break

                if _set:
                    continue

                # try to set kwargs
                for kwarg in six.itervalues(deps[1]):
                    if kwarg._name_no_id() == k:
                        kwarg._dirty = (kwarg._value != v)
                        kwarg._value = v
                        _set = True
                        break

    def value(self):
        return self._value

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

        node_args.append(Node(name=arg,
                              derived=True,
                              readonly=False,
                              nullable=nullable,
                              value=value,
                              trace=trace))

    for k, v in six.iteritems(argspec.kwonlydefaults or {}):
        node_kwargs[k] = Node(name=k,
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
