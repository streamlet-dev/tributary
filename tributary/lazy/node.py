import six
import inspect

# from boltons.funcutils import wraps
from ..utils import _either_type
from ..base import TributaryException


class Node(object):
    """Class to represent an operation that is lazy"""

    _id_ref = 0

    def __init__(
        self,
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
        override_callable_dirty=False,
        **kwargs
    ):
        """Construct a new lazy node, wrapping a callable or a value

        Args:
            name (str): name to use to represent the node
            derived (bool): node is note instantiated directly,
                            e.g. via n + 10 where n is a preexisting node.
                            These default to dirty state
            readonly (bool): whether a node is settable
            nullable (bool): whether a node can have value None
            value (any): initial value of the node
            callable (callable): function or other callable that the node is wrapping
            callable_args (tuple): args for the wrapped callable
            callable_kwargs (dict): kwargs for the wrapped callable
            callable_is_method (bool): is the callable a method of an object
            always_dirty (bool): node should not be lazy - always access underlying value
            override_callable_dirty (bool): treat callable as dirty-able. This is an override of the default
                                                behavior where a callable will be re-evaluated
        """
        # Instances get an id but one id tracker for all nodes so we can
        # uniquely identify them
        # TODO different scheme
        self._id = Node._id_ref
        Node._id_ref += 1

        # Every node gets a name so it can be uniquely identified in the graph
        self._name = "{}#{}".format(name or self.__class__.__name__, self._id)

        if isinstance(value, Node):
            raise TributaryException("Cannot set value to be itself a node")

        # Graphviz shape
        self._graphvizshape = kwargs.get(
            "graphvizshape", "box"
        )  # default is box instead of ellipse
        # because all lazy nodes are i/o nodes
        # by default

        # if using dagre-d3, this will be set
        self._dd3g = None

        # starting value
        self._value = value

        # use dual number operators
        self._use_dual = kwargs.get("use_dual", False)

        # can be set
        self._readonly = readonly

        # callable and args
        self._callable_args = callable_args or []
        self._callable_kwargs = callable_kwargs or {}
        self._callable_is_method = callable_is_method

        # map positional to kw
        if callable is not None and not inspect.isgeneratorfunction(callable):
            args = inspect.getfullargspec(callable).args
            if "self" in args:
                args.remove("self")
            self._callable_args_mapping = {i: arg for i, arg in enumerate(args)}
        else:
            self._callable_args_mapping = {}

        if not inspect.isgeneratorfunction(callable):
            self._callable = callable
        else:
            # FIXME this wont work for attribute inputs
            def _callable(gen=callable(*self._callable_args, **self._callable_kwargs)):
                try:
                    ret = next(gen)
                    return ret
                except StopIteration:
                    self._always_dirty = False
                    self._dirty = False
                    return self._value

            self._callable = _callable

        # if always dirty, always reevaluate
        self._always_dirty = always_dirty or (
            not override_callable_dirty and self._callable is not None
        )

        # parent nodes in graph
        self._parents = []

        # self reference for method calls
        self._self_reference = self

        # cache node operations that have already been done
        self._node_op_cache = {}

        # setup dependency dirty map
        self._dirty_dependency = {}

        # dependencies can be nodes
        if self._callable:
            self._callable._node_wrapper = None  # not known until program start
            self._dependencies = {
                self._callable: (self._callable_args, self._callable_kwargs)
            }
        else:
            self._dependencies = {}
            self._inputs = {}

        # if derived node, default to dirty to start
        if derived:
            self._dirty = True
        else:
            self._dirty = False

    def inputs(self, name=""):
        """get node inputs, optionally by name"""
        dat = {n._name_no_id(): n for n in self._callable_args}
        return dat if not name else dat.get(name)

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
        return self._name.rsplit("#", 1)[0]

    def _install_args(self, *args):
        kwargs = []
        for i, arg in enumerate(args):
            if (
                i < len(self._callable_args)
                and self._callable_args[i]._name_no_id()
                == self._callable_args_mapping[i]
            ):
                self._callable_args[i].setValue(arg)
            else:
                kwargs.append((self._callable_args_mapping[i], arg))

        for k, v in kwargs:
            self._callable_kwargs[k].setValue(v)

    def _install_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            self._callable_kwargs[k].setValue(v)

    def _with_self(self, other_self, *args, **kwargs):
        self._self_reference = other_self
        self._install_args(*args)
        self._install_kwargs(**kwargs)
        return self

    def _without_self(self, *args, **kwargs):
        self._install_args(*args)
        self._install_kwargs(**kwargs)
        return self

    def _greendd3g(self):
        if self._dd3g:
            self._dd3g.setNode(
                self._name, tooltip=str(self.value()), style="fill: #0f0"
            )

    def _yellowdd3g(self):
        if self._dd3g:
            self._dd3g.setNode(
                self._name, tooltip=str(self.value()), style="fill: #ff0"
            )

    def _reddd3g(self):
        if self._dd3g:
            self._dd3g.setNode(
                self._name, tooltip=str(self.value()), style="fill: #f00"
            )

    def _whited3g(self):
        if self._dd3g:
            self._dd3g.setNode(
                self._name, tooltip=str(self.value()), style="fill: #fff"
            )

    def dependencyIsDirty(self, dep):
        """In a callable, use this method to determine if a dependency WAS dirty.
        Since the dependency will be reevaluated before the callable is executed,
        you would otherwise lose the information"""
        return self._dirty_dependency.get(dep, False)

    def _compute_from_dependencies(self):
        if self._dependencies:
            self._greendd3g()
            for deps in six.itervalues(self._dependencies):
                # recompute args
                for arg in deps[0]:
                    self._dirty_dependency[arg] = arg._recompute()

                    # Set yourself as parent
                    if self not in arg._parents:
                        arg._parents.append(self)

                # recompute kwargs
                for kwarg in six.itervalues(deps[1]):
                    self._dirty_dependency[kwarg] = kwarg._recompute()

                    # Set yourself as parent
                    if self not in kwarg._parents:
                        kwarg._parents.append(self)

            k = list(self._dependencies.keys())[0]

            if self._callable_is_method:
                new_value = k(
                    self._self_reference,
                    *self._dependencies[k][0],
                    **self._dependencies[k][1]
                )
            else:
                new_value = k(*self._dependencies[k][0], **self._dependencies[k][1])

            if isinstance(new_value, Node):
                k._node_wrapper = new_value
                new_value = new_value()  # get value

            if isinstance(new_value, Node):
                raise TributaryException("Value should not itself be a node!")

            self._value = new_value

        # reset to empty
        self._dirty_dependency = {}

        self._whited3g()
        return self._value

    def _subtree_dirty(self):
        for call, deps in six.iteritems(self._dependencies):
            # callable node
            if hasattr(call, "_node_wrapper") and call._node_wrapper is not None:
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
        ret = False
        self.isDirty()
        if self._dirty:
            ret = True
            if self._parents:
                for parent in self._parents:
                    # let your parents know you were dirty!
                    parent._dirty = True
            self._value = self._compute_from_dependencies()
        self._dirty = False
        return ret

    def _gennode(self, name, foo, foo_args, **kwargs):
        if name not in self._node_op_cache:
            self._node_op_cache[name] = Node(
                name=name,
                derived=True,
                callable=foo,
                callable_args=foo_args,
                override_callable_dirty=True,
                **kwargs
            )
        return self._node_op_cache[name]

    def _tonode(self, other):
        if isinstance(other, Node):
            return other
        if str(other) not in self._node_op_cache:
            self._node_op_cache[str(other)] = Node(
                name="var(" + str(other)[:5] + ")", derived=True, value=other
            )
        return self._node_op_cache[str(other)]

    def setValue(self, value):
        if value != self._value:
            self._value = value  # leave for dagre
            self._dirty = True
        self._value = value

    def append(self, value):
        # TODO is this better or worse than
        # lst = []
        # n = Node(value=lst)
        # lst.append(x)
        # n._dirty = True
        iter(self._value)
        self._value.append(value)
        self._dirty = True

    def set(self, *args, **kwargs):
        for k, v in six.iteritems(kwargs):
            _set = False
            for deps in six.itervalues(self._dependencies):
                # try to set args
                for arg in deps[0]:
                    if arg._name_no_id() == k:
                        arg._dirty = arg._value != v
                        arg._value = v
                        _set = True
                        break

                if _set:
                    continue

                # try to set kwargs
                for kwarg in six.itervalues(deps[1]):
                    if kwarg._name_no_id() == k:
                        kwarg._dirty = kwarg._value != v
                        kwarg._value = v
                        _set = True
                        break

    def value(self):
        return self._value

    def __call__(self, *args, **kwargs):
        self._install_args(*args)
        self._install_kwargs(**kwargs)

        self._recompute()
        return self.value()

    def __repr__(self):
        return self._name


@_either_type
def node(meth, memoize=True, **default_attrs):
    """Convert a method into a lazy node

    Since `self` is not defined at the point of method creation, you can pass in
    extra kwargs which represent attributes of the future `self`. These will be
    converted to node args during instantiation

    The format is:
        @node(my_existing_attr_as_an_arg="_attribute_name"):
        def my_method(self):
            pass

    this will be converted into a graph of the form:
        self._attribute_name -> my_method
    e.g. as if self._attribute_name was passed as an argument to my_method, and converted to a node in the usual manner
    """

    argspec = inspect.getfullargspec(meth)
    # args = argspec.args
    # varargs = argspec.varargs
    # varkw = argspec.varkw
    # defaults = argspec.defaults
    # kwonlyargs = argspec.kwonlyargs
    # kwonlydefaults = args.kwonlydefaults

    if argspec.varargs:
        raise TributaryException("varargs not supported yet!")

    if argspec.varkw:
        raise TributaryException("varargs not supported yet!")

    if inspect.isgeneratorfunction(meth) and default_attrs:
        raise TributaryException("Not a supported pattern yet!")

    node_args = []
    node_kwargs = {}
    is_method = False

    # iterate through method's args and convert them to nodes
    for i, arg in enumerate(argspec.args):
        if arg == "self":
            # TODO
            is_method = True
            continue

        if (is_method and len(argspec.defaults or []) >= i) or (
            not is_method and len(argspec.defaults or []) > i
        ):
            default = True
            value = (
                argspec.defaults[i] if not is_method else argspec.defaults[i - 1]
            )  # account for self
            nullable = True
        else:
            default = False
            value = None
            nullable = False

        if arg not in default_attrs and not default:
            node_args.append(
                Node(
                    name=arg,
                    derived=True,
                    readonly=False,
                    nullable=nullable,
                    value=value,
                )
            )
        elif default:
            node_kwargs[arg] = Node(
                name=arg, derived=True, readonly=False, nullable=nullable, value=value
            )

    for k, v in six.iteritems(argspec.kwonlydefaults or {}):
        node_kwargs[k] = Node(
            name=k, derived=True, readonly=False, nullable=True, value=v
        )

    # add all attribute args to the argspec
    # see the docstring for more details

    # argspec.args.extend(list(default_attrs.keys()))
    node_kwargs.update(default_attrs)

    if (
        len([arg for arg in argspec.args if arg != "self"])
        + len(argspec.kwonlydefaults or {})
    ) != (len(node_args) + len(node_kwargs)):
        raise TributaryException(
            "Missing args (call or preprocessing error has occurred)"
        )

    def meth_wrapper(self, *args, **kwargs):
        if is_method:
            # val = meth(self, *(arg.value() if isinstance(arg, Node) else getattr(self, arg).value() for arg in args if arg not in default_attrs), **
            #            {k: v.value() if isinstance(v, Node) else getattr(self, v).value() for k, v in kwargs.items() if k not in default_attrs})
            val = meth(
                self,
                *(
                    arg.value() if isinstance(arg, Node) else getattr(self, arg).value()
                    for arg in args
                ),
                **{
                    k: v.value() if isinstance(v, Node) else getattr(self, v).value()
                    for k, v in kwargs.items()
                }
            )
        else:
            val = meth(
                *(
                    arg.value() if isinstance(arg, Node) else getattr(self, arg).value()
                    for arg in args
                ),
                **{
                    k: v.value() if isinstance(v, Node) else getattr(self, v).value()
                    for k, v in kwargs.items()
                }
            )
        return val

    new_node = Node(
        name=meth.__name__,
        derived=True,
        callable=meth_wrapper,
        callable_args=node_args,
        callable_kwargs=node_kwargs,
        callable_is_method=is_method,
        always_dirty=not memoize,
    )

    if is_method:
        ret = lambda self, *args, **kwargs: new_node._with_self(  # noqa: E731
            self, *args, **kwargs
        )
    else:
        ret = lambda *args, **kwargs: new_node._without_self(  # noqa: E731
            *args, **kwargs
        )

    ret._node_wrapper = new_node
    # ret = wraps(meth)(ret)
    return ret
