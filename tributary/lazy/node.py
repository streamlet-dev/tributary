import inspect
import uuid
from collections import namedtuple

from ..base import TributaryException

# from boltons.funcutils import wraps
from ..utils import _compare, _either_type, _ismethod
from .dd3 import _DagreD3Mixin


class Node(_DagreD3Mixin):
    """Class to represent an operation that is lazy"""

    def __init__(
        self,
        value=None,
        name="?",
        derived=False,
        callable=None,
        callable_args=None,
        callable_kwargs=None,
        dynamic=False,
        **kwargs,
    ):
        """Construct a new lazy node, wrapping a callable or a value

        Args:
            name (str): name to use to represent the node
            derived (bool): node is note instantiated directly,
                            e.g. via n + 10 where n is a preexisting node.
                            These default to dirty state
            value (any): initial value of the node
            callable (callable): function or other callable that the node is wrapping
            callable_args (tuple): args for the wrapped callable
            callable_kwargs (dict): kwargs for the wrapped callable
            dynamic (bool): node should not be lazy - always access underlying value
        """
        # ID is unique identifier of the node
        self._id = str(uuid.uuid4())

        # Name is a string for display
        self._name = "{}#{}".format(
            name
            or (callable.__name__ if callable else None)
            or self.__class__.__name__,
            self._id[:5],
        )

        if isinstance(value, Node):
            raise TributaryException("Cannot set value to be itself a node")

        # Graphviz shape
        self._graphvizshape = kwargs.get("graphvizshape", "box")
        # default is box instead of ellipse
        # because all lazy nodes are i/o nodes
        # by default

        # if using dagre-d3, this will be set
        self._dd3g = None

        # starting value
        self._values = []

        # use dual number operators
        self._use_dual = kwargs.get("use_dual", False)

        # threshold for calculating difference
        self._compare = _compare

        # callable and args
        self._callable_is_method = _ismethod(callable)
        self._callable = callable

        # map arguments of callable to nodes
        self._callable_args = callable_args or []
        self._callable_kwargs = callable_kwargs or {}

        # callable_args_mapping maps the wrapped functions'
        # arguments to nodes. It does so in 2 ways, either
        # via the argument node's name, or the wrapped
        # function's argument name
        #
        # e.g. if i have lambda x, y: x + y
        # where x is set to a Node(name="One")
        # and y is set to a Node(name="Two"),
        # callable_args_mapping looks like:
        # {0: {"node": "One", "arg": "x"},
        #  1: {"node": "Two", "arg": "y"}}
        #
        # this way i can pass (x=5) or (One=5)
        # to modify the node's value
        self._callable_args_mapping = {}

        # map positional to kw
        if callable is not None and not inspect.isgeneratorfunction(callable):
            # wrap args and kwargs of function to node
            try:
                signature = inspect.signature(callable)

            except ValueError:
                # https://bugs.python.org/issue20189
                signature = namedtuple("Signature", ["parameters"])({})

            parameters = [
                p
                for p in signature.parameters.values()
                if p.kind
                not in (
                    inspect._ParameterKind.VAR_POSITIONAL,
                    inspect._ParameterKind.VAR_KEYWORD,
                )
            ]

            # map argument index to name of argument, for later use
            self._callable_args_mapping = {
                i: {"arg": arg.name} for i, arg in enumerate(parameters)
            }

            # first, iterate through callable_args and callable_kwargs and convert to nodes
            for i, arg in enumerate(self._callable_args):
                # promote all args to nodes
                if not isinstance(arg, Node):
                    # see if arg in argspec
                    if i < len(parameters):
                        name = parameters[i].name
                    else:
                        name = "vararg"

                    self._callable_args[i] = Node(name=name, value=arg)

                # ensure arg can be passed by either node name, or arg name
                if i not in self._callable_args_mapping:
                    # varargs, disallow by arg
                    self._callable_args_mapping[i] = {}
                self._callable_args_mapping[i]["node"] = self._callable_args[
                    i
                ]._name_no_id()

            # first, iterate through callable_args and callable_kwargs and convert to nodes
            for name, kwarg in self._callable_kwargs.items():
                if not isinstance(kwarg, Node):
                    self._callable_kwargs[name] = Node(name=name, value=kwarg)

            # now iterate through callable's args and ensure
            # everything is matched up
            for i, arg in enumerate(parameters):
                if arg.name == "self":
                    # skip
                    continue

                # passed in as arg
                if i < len(self._callable_args) or arg.name in self._callable_kwargs:
                    # arg is passed in args/kwargs, continue
                    continue

                # arg not accounted for, see if it has a default in the callable
                # convert to node
                node = Node(name=arg.name, derived=True, value=arg.default)

                # set in kwargs
                self._callable_kwargs[arg.name] = node

            # compare filtered parameters to original
            if len(parameters) != len(signature.parameters):
                # if varargs, can have more callable_args + callable_kwargs than listed arguments
                failif = len([arg for arg in parameters if arg.name != "self"]) > (
                    len(self._callable_args) + len(self._callable_kwargs)
                )
            else:
                # should be exactly equal
                failif = len([arg for arg in parameters if arg.name != "self"]) != (
                    len(self._callable_args) + len(self._callable_kwargs)
                )

            if failif:
                # something bad happened trying to align
                # callable's args/kwargs with the provided
                # callable_args and callable_kwargs, and we're
                # now in an unrecoverable state.
                raise TributaryException(
                    "Missing args (call or preprocessing error has occurred)"
                )

            try:
                self._callable._node_wrapper = None  # not known until program start
            except AttributeError:
                # can't set attributes on certain functions, so wrap with lambda
                if self._callable_is_method:
                    self._callable = lambda self, *args, **kwargs: callable(
                        *args, **kwargs
                    )
                else:
                    self._callable = lambda *args, **kwargs: callable(*args, **kwargs)
                self._callable._node_wrapper = None  # not known until program start

        elif callable is not None:
            self._callable_args = callable_args or []
            self._callable_kwargs = callable_kwargs or {}

            # FIXME this wont work for attribute inputs
            def _callable(gen=callable(*self._callable_args, **self._callable_kwargs)):
                try:
                    ret = next(gen)
                    return ret
                except StopIteration:
                    self._dynamic = False
                    self._dirty = False
                    return self.value()

            self._callable = _callable

        # list out all upstream nodes
        self._upstream = list(self._callable_args) + list(
            self._callable_kwargs.values()
        )

        # if always dirty, always reevaluate
        # self._dynamic = dynamic  # or self._callable is not None
        self._dynamic = (
            dynamic
            or (self._callable and inspect.isgeneratorfunction(callable))
            or False
        )

        # parent nodes in graph
        self._parents = []

        # self reference for method calls
        self._self_reference = self

        # cache node operations that have already been done
        self._node_op_cache = {}

        # tweaks
        self._tweaks = None

        # dependencies can be nodes
        if self._callable:
            self._dependencies = {
                self._callable: (self._callable_args, self._callable_kwargs)
            }
        else:
            self._dependencies = {}

            # insert initial value
            self._setValue(value)

        # use this variable when manually overriding
        # a callable to have a fixed value
        self._dependencies_stashed = {}

        # if derived node, default to dirty to start
        if derived or self._callable is not None:
            self._dirty = True
        else:
            self._dirty = False

    def inputs(self, name=""):
        """get node inputs, optionally by name"""
        dat = {n._name_no_id(): n for n in self._upstream}
        return dat if not name else dat.get(name)

    def _get_dirty(self):
        return self._is_dirty

    def _set_dirty(self, val):
        self._reddd3g() if val else self._whited3g()
        self._is_dirty = val

    _dirty = property(_get_dirty, _set_dirty)

    def _name_no_id(self):
        return self._name.rsplit("#", 1)[0]

    def _install_args(self, *args):
        """set arguments' values to those given. this is a permanent operation"""
        kwargs = []
        for i, arg in enumerate(args):
            if (
                i < len(self._callable_args)
                and self._callable_args[i]._name_no_id()
                in self._callable_args_mapping[i].values()
            ):
                self._callable_args[i].setValue(arg)
            else:
                kwargs.append((self._callable_args_mapping[i]["node"], arg))

        for k, v in kwargs:
            self._callable_kwargs[k].setValue(v)

    def _install_kwargs(self, **kwargs):
        """set arguments' values to those given. this is a permanent operation"""
        for k, v in kwargs.items():
            self._callable_kwargs[k].setValue(v)

    def _get_arg(self, i):
        return self._callable_args[i]

    def _get_kwarg(self, keyword):
        print(self._callable_args)
        return self._callable_kwargs[keyword]

    def _bind(self, other_self=None, *args, **kwargs):
        if other_self is not None:
            self._self_reference = other_self
        self._install_args(*args)
        self._install_kwargs(**kwargs)
        return self

    def _tweak(self, node_tweaks):
        # TODO context manager
        self._tweaks = node_tweaks

    def _untweak(self):
        # TODO context manager
        self._tweaks = None

        # mark myself as dirt for tweak side-effects
        # TODO another way of doing this?
        self._dirty = True

    def _compute_from_dependencies(self, node_tweaks):
        """recompute node's value from its dependencies, applying any temporary tweaks as necessary"""

        # if i'm the one being tweaked, just return tweaked value
        if self in node_tweaks:
            return node_tweaks[self]

        # if i have upstream dependencies
        if self._dependencies:
            # mark graph as calculating
            self._greendd3g()

            # iterate through upstream deps
            for deps in self._dependencies.values():
                # recompute args
                for arg in deps[0]:
                    # recompute
                    arg._recompute(node_tweaks)

                    # Set yourself as parent if not set
                    if self not in arg._parents:
                        arg._parents.append(self)

                    # mark as tweaking
                    if node_tweaks:
                        arg._tweak(node_tweaks)

                # recompute kwargs
                for kwarg in deps[1].values():
                    # recompute
                    kwarg._recompute(node_tweaks)

                    # Set yourself as parent if not set
                    if self not in kwarg._parents:
                        kwarg._parents.append(self)

                    # mark as tweaking
                    if node_tweaks:
                        kwarg._tweak(node_tweaks)

            # fetch the callable
            kallable = list(self._dependencies.keys())[0]

            if self._callable_is_method:
                # if the callable is a method,
                # pass this node as self
                new_value = kallable(
                    self._self_reference,
                    *self._dependencies[kallable][0],
                    **self._dependencies[kallable][1],
                )
            else:
                # else just call on deps
                new_value = kallable(
                    *self._dependencies[kallable][0], **self._dependencies[kallable][1]
                )

            if isinstance(new_value, Node):
                # extract numerical value from node, if it is a node
                kallable._node_wrapper = new_value
                new_value = new_value()  # get value

            if isinstance(new_value, Node):
                raise TributaryException("Value should not itself be a node!")

            # set my value as new value if not tweaking
            if not node_tweaks:
                self._setValue(new_value)
            else:
                # set value in tweak dict
                node_tweaks[self] = new_value

                # iterate through upstream deps and unset tweak
                for deps in self._dependencies.values():
                    for arg in deps[0]:
                        arg._untweak()
                    for kwarg in deps[1].values():
                        kwarg._untweak()
        else:
            # if i don't have upstream dependencies, my value is fixed
            new_value = self.value()

        # mark calculation complete
        self._whited3g()

        # return my value
        return self.value()

    def _subtree_dirty(self, node_tweaks):
        for call, deps in self._dependencies.items():
            # callable node
            if hasattr(call, "_node_wrapper") and call._node_wrapper is not None:
                if call._node_wrapper.isDirty(node_tweaks):
                    # CRITICAL
                    # always set self to be dirty if subtree is dirty
                    self._dirty = True
                    return True

            # check args
            for arg in deps[0]:
                if arg.isDirty(node_tweaks):
                    # CRITICAL
                    # always set self to be dirty if subtree is dirty
                    self._dirty = True
                    return True

            # check kwargs
            for kwarg in deps[1].values():
                if kwarg.isDirty(node_tweaks):
                    # CRITICAL
                    # always set self to be dirty if subtree is dirty
                    self._dirty = True
                    return True
        return False

    def isDirty(self, node_tweaks=None):
        """Node needs to be re-evaluated, either because its value has changed
        or because its value *could* change

        Note that in evaluating if a node is dirty, you will have a side effect
        of updating that node's status to be dirty or not.
        """
        node_tweaks = node_tweaks or {}

        if self in node_tweaks:
            # return dirty but don't set
            return _compare(node_tweaks[self], self.value())

        self._dirty = self._dirty or self._subtree_dirty(node_tweaks) or self._dynamic
        return self._dirty

    def isDynamic(self):
        """Node isnt necessarily dirty, but needs to be reevaluated"""
        return self._dynamic

    def _recompute(self, node_tweaks):
        """returns result of computation"""
        # check if self or upstream dirty
        self.isDirty(node_tweaks)

        # if i'm dirty, recompute my value
        if self._dirty:
            # compute upstream and then apply to self
            new_value = self._compute_from_dependencies(node_tweaks)

            # if my new value is not equal to my old value,
            # make sure to indicate that i was really dirty
            if self._compare(new_value, self.value()):
                # mark my parents as dirty
                if self._parents:
                    for parent in self._parents:
                        # let your parents know you were dirty!
                        parent._dirty = True

                # set my value if not tweaking
                if not node_tweaks:
                    self._setValue(new_value)
        else:
            new_value = self.value()

        # mark myself as no longer dirty
        self._dirty = False

        # return result of computation
        return new_value

    def _gennode(self, name, func, func_args, **kwargs):
        if name not in self._node_op_cache:
            self._node_op_cache[name] = Node(
                name=name,
                derived=True,
                callable=func,
                callable_args=func_args,
                override_callable_dirty=True,
                **kwargs,
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
        """set the node's value, marking it as dirty as appropriate.
        this operation is permanent"""
        if self._compare(value, self.value()):
            # if callable, stash and force a fixed value
            if self._dependencies:
                # stash dependency tree for later
                self._dependencies_stashed = self._dependencies

                # reset to empty
                self._dependencies = {}

                # mark as not dynamic anymore
                self._dynamic = False

            # set the value
            self._setValue(value)  # leave for dagre

            # mark as dirty
            self._dirty = True

    def unlock(self):
        """if node has been set to a fixed value, reset to callable"""
        # no-op if not previously stashed
        if self._dependencies_stashed:
            # restore dependency tree
            self._dependencies = self._dependencies_stashed

            # clear out stashed
            self._dependencies_stashed = {}

            # mark as dynamic again
            self._dynamic = True

    def _setValue(self, value):
        """internal method to set value. this is a permanent operation"""
        # if value != self.value():
        self._values.append(value)

    def append(self, value):
        # TODO is this better or worse than
        # lst = []
        # n = Node(value=lst)
        # lst.append(x)
        # n._dirty = True
        iter(self.value())
        self.value().append(value)
        self._dirty = True

    def get(self, **kwargs):
        for k, v in kwargs.items():
            for deps in self._dependencies.values():
                # try to set args
                for i, arg in enumerate(deps[0]):
                    if arg._name_no_id() == k:
                        return arg

                # try to set kwargs
                for key, kwarg in deps[1].items():
                    if kwarg._name_no_id() == k:
                        return kwarg

    def set(self, **kwargs):
        """this method sets upstream dependencys' values to those given"""
        for k, v in kwargs.items():
            _set = False
            for deps in self._dependencies.values():
                # try to set args
                for i, arg in enumerate(deps[0]):
                    if arg._name_no_id() == k:
                        if isinstance(v, Node):
                            # overwrite node
                            deps[0][i] = v
                        else:
                            arg._dirty = arg.value() != v
                            arg.setValue(v)
                            _set = True
                            break

                if _set:
                    continue

                # try to set kwargs
                for key, kwarg in deps[1].items():
                    if kwarg._name_no_id() == k:
                        if isinstance(v, Node):
                            # overwrite node
                            deps[1][key] = v
                        else:
                            kwarg._dirty = kwarg.value() != v
                            kwarg._setValue(v)
                            # _set = True
                            break

    def getValue(self):
        """Get the value of the node"""
        return self.value()

    def value(self):
        """Get the value of the node"""
        # if tweaking, return my tweaked value
        if self._tweaks and self in self._tweaks:
            return self._tweaks[self]

        # otherwise return my latest value
        return self._values[-1] if self._values else None

    def __call__(self, node_tweaks=None, *positional_tweaks, **keyword_tweaks):
        """Lazily re-evaluate the node

        Args:
            node_tweaks (dict): A dict mapping node to tweaked value
            positional_tweaks (VAR_POSITIONAL): A tuple of positional tweaks to apply
            keyword_tweaks (VAR_KEYWORD): A dict of keyword tweaks to apply
                How it works: The "original caller" is the node being evaluted w/ tweaks.
                It will consume the positional_tweaks` and `keyword_tweaks`, which look like:
                    (1, 2,)  ,  {"a": 5, "b": 10}
                and join them with `node_tweaks` in a dict mapping node->tweaked value, e.g.
                    {Node1: 1, Node2: 2, NodeA: 5, NodeB: 10}
                and pass this dict up the call tree in `node_tweaks`.

                This dict is carried through all node operations through the entire call tree.
                If a node is being evaluated and is in `node_tweaks`, it ignores recalculation
                and returns the tweaked value.
        Returns:
            Any: the value, either via re-evaluation (if self or upstream dirty),
                 or the previously computed value
        """
        node_tweaks = node_tweaks or {}

        if not isinstance(node_tweaks, dict):
            # treat node_tweak argument as positional tweak
            positional_tweaks = list(positional_tweaks) + [node_tweaks]
            node_tweaks = {}

        # instantiate tweaks
        tweaks = {}

        # update with provided
        tweaks.update(node_tweaks)

        for i, positional_tweak in enumerate(positional_tweaks):
            tweaks[self._get_arg(i)] = positional_tweak

        for k, keyword_tweak in keyword_tweaks.items():
            tweaks[self._get_kwarg(k)] = keyword_tweak

        # tweak self
        if tweaks:
            self._tweak(tweaks)

        # calculate new value
        computed = self._recompute(tweaks)

        if tweaks:
            # untweak self
            self._untweak()

            # return the calculation result, not my current value
            return computed

        # otherwise return my permanent value, should equal computed
        # assert self.value() == computed
        return self.value()

    def evaluate(self, node_tweaks=None, *positional_tweaks, **keyword_tweaks):
        return self(node_tweaks, *positional_tweaks, **keyword_tweaks)

    def eval(self, node_tweaks=None, *positional_tweaks, **keyword_tweaks):
        return self(node_tweaks, *positional_tweaks, **keyword_tweaks)

    def __repr__(self):
        return self._name


@_either_type
def node(meth, dynamic=True, **default_attrs):
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

    signature = inspect.signature(meth)
    parameters = [
        p
        for p in signature.parameters.values()
        if p.kind
        not in (
            inspect._ParameterKind.VAR_POSITIONAL,
            inspect._ParameterKind.VAR_KEYWORD,
        )
    ]

    # don't handle varargs yet
    if len(parameters) != len(signature.parameters):
        raise TributaryException("varargs not supported yet!")

    if inspect.isgeneratorfunction(meth) and default_attrs:
        raise TributaryException("Not a supported pattern yet!")

    node_args = []
    node_kwargs = {}
    is_method = _ismethod(meth)

    # iterate through method's args and convert them to nodes
    for i, arg in enumerate(parameters):
        if arg.name == "self":
            continue

        node_kwargs[arg.name] = Node(name=arg.name, derived=True, value=arg.default)

    # add all attribute args to the argspec
    # see the docstring for more details

    # argspec.args.extend(list(default_attrs.keys()))
    node_kwargs.update(default_attrs)

    if (len(parameters) - 1 if is_method else len(parameters)) != (
        len(node_args) + len(node_kwargs)
    ):
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
                },
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
                },
            )
        return val

    new_node = Node(
        name=meth.__name__,
        derived=True,
        callable=meth_wrapper,
        callable_args=node_args,
        callable_kwargs=node_kwargs,
        dynamic=dynamic,
    )

    if is_method:
        ret = lambda self, *args, **kwargs: new_node._bind(  # noqa: E731
            self, *args, **kwargs
        )
    else:
        ret = lambda *args, **kwargs: new_node._bind(  # noqa: E731
            None, *args, **kwargs
        )

    ret._node_wrapper = new_node
    # ret = wraps(meth)(ret)
    return ret
