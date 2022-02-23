import inspect
import uuid
from collections import namedtuple
from frozendict import frozendict

from tomlkit import value

from ..base import TributaryException, StreamNone

# from boltons.funcutils import wraps
from ..utils import _compare, _ismethod, _either_type, _gen_to_func
from .dd3 import _DagreD3Mixin

ArgState = namedtuple("ArgState", ["args", "kwargs", "varargs", "varkwargs"])


class ParamType:
    POSITIONAL_ONLY = inspect._ParameterKind.POSITIONAL_ONLY
    KEYWORD_ONLY = inspect._ParameterKind.KEYWORD_ONLY
    POSITIONAL_OR_KEYWORD = inspect._ParameterKind.POSITIONAL_OR_KEYWORD
    VAR_POSITIONAL = inspect._ParameterKind.VAR_POSITIONAL
    VAR_KEYWORD = inspect._ParameterKind.VAR_KEYWORD


def extractParameters(callable):
    """Given a function, extract the arguments and defaults"""
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

        if kind == ParamType.VAR_POSITIONAL:
            # default is empty tuple
            self.default = tuple()

        elif kind == ParamType.VAR_KEYWORD:
            # default is empty dict
            self.default = {}

        elif default == inspect._empty:
            # default can be inspect._empty
            self.default = StreamNone()
        else:
            # default specified inline
            # TODO default-as-function
            self.default = default


class Node(_DagreD3Mixin):
    """Class to represent an operation that is lazy"""

    def __init__(
        self,
        value=None,
        name="",
        dynamic=False,
        use_dual=False,
        args=None,
        kwargs=None,
        graphvizshape="box",
    ):
        """Construct a new lazy node, wrapping a callable or a value"""

        # ID is unique identifier of the node
        self._id = str(uuid.uuid4())

        # Graphviz shape
        self._graphvizshape = graphvizshape
        # default is box instead of ellipse
        # because all lazy nodes are i/o nodes
        # by default

        # if using dagre-d3, this will be set
        self._dd3g = None

        # use dual number operators
        self._use_dual = use_dual

        # threshold for calculating difference
        self._compare = _compare

        # dynamic nodes are always reevaluated e.g. not lazy
        self._dynamic = dynamic

        # Downstream nodes so we can traverse graph, push
        # results to downstream nodes
        self._downstream = []

        # Upstream nodes so we can traverse graph, plot and optimize
        self._upstream = []

        # convert static value to callable
        if not callable(value):
            # static value, so not dirty
            # by default and don't need
            # to eval
            self._dirty = False
            self._last_value = value

            # construct a function wrapper
            self._value = lambda: value
            self._value.__name__ = str(self._last_value)
        else:
            # default function nodes to dirty
            self._dirty = True
            self._last_value = StreamNone()
            self._value = value

        if isinstance(self._value, Node):
            raise TributaryException("Cannot set value to be itself a node")

        # Name is a string for display
        self._name_no_id = (
            name or self._value.__name__ if hasattr(self._value, "__name__") else "?"
        )
        self._name = "{}#{}".format(
            self._name_no_id,
            self._id[:5],
        )

        # handle if callable is generator
        if inspect.isgeneratorfunction(self._value):
            # store the generator
            self._gen = self._value()

            # replace with a lambda for next
            self._value = lambda: _gen_to_func(self._gen)

            # set myself to dynamic so i'm always reevaluated
            self._dynamic = True

        # callable and args
        self._callable_is_method = _ismethod(self._value)

        # self reference for method calls
        self._self_reference = self

        # cache node operations that have already been done
        self._node_op_cache = {}

        # extract base params
        # Note that this might modify self._value, if self._value would be a generator
        self._parameters = extractParameters(self._value)

        # Attach node attribute to callable
        try:
            self._value._node_wrapper = self
        except AttributeError:
            # if we can't do this and we're a method,
            # detach our self. This it to handle errors that
            # come from methods like random.random()
            if (
                self._callable_is_method
                and self._parameters
                and self._parameters[0].name == "self"
            ):
                self._parameters = self._parameters[1:]

        # go through parameters and wrap in node, or used provided default
        self.args = []
        self.kwargs = {}

        # we may need to update parameters for dynamic args/kwargs, so we will modify this and set it at the end
        updated_parameters = self._parameters.copy()

        for param in self._parameters:
            if param.name == "self" and self._callable_is_method:
                # skip this
                continue

            # calculate the real position taking into account the "self"
            param.position = (
                param.position if not self._callable_is_method else param.position - 1
            )

            if param.kind == ParamType.VAR_POSITIONAL:
                # flush the remaining args into new parameters
                # first pop the args placeholder parameters
                updated_parameters.pop()

                # now flush into parameters
                for i, arg in enumerate(args):
                    new_param = Parameter(
                        "{}{}".format(param.name, param.position + i),
                        param.position + i,
                        None,
                        ParamType.POSITIONAL_ONLY,
                    )
                    updated_parameters.append(new_param)

                    if isinstance(arg, Node):
                        # if its a node, use the node
                        parameter_node = arg
                    else:
                        # else wrap it in a node with the provided value
                        parameter_node = Node(arg, new_param.name)

                    self._pushDep(new_param, parameter_node)

            elif param.kind == ParamType.VAR_KEYWORD:
                # flush the remaining kwargs into new parameters
                # TODO
                raise NotImplementedError()

            elif param.position < len(args or ()):
                # use input arg
                if isinstance(args[param.position], Node):
                    # if its a node, use the node
                    parameter_node = args[param.position]
                else:
                    # else wrap it in a node with the provided value
                    parameter_node = Node(args[param.position], param.name)

                # push node as dependency
                self._pushDep(param, parameter_node)

            elif param.name in (kwargs or {}):
                # use input kwarg
                if isinstance(kwargs[param.name], Node):
                    # if its a node, use the node
                    parameter_node = kwargs[param.name]
                else:
                    # else wrap it in a node with the provided value
                    parameter_node = Node(kwargs[param.name], param.name)

                # push node as dependency
                self._pushDep(param, parameter_node)

            else:
                # otherwise, wrap the function argument to a node using the function-defined default
                parameter_node = Node(param.default, param.name)

                # push node as dependency
                self._pushDep(param, parameter_node)

        self._parameters = updated_parameters

    def _pushDep(self, param, parameter_node):
        self.args.append(parameter_node)
        self.kwargs[param.name] = parameter_node
        self << parameter_node

    # ***********************
    # Public interface
    # ***********************
    def __repr__(self):
        return "{}".format(self._name)

    def upstream(self, node=None):
        """Access list of upstream nodes"""
        return self._upstream

    def downstream(self, node=None):
        """Access list of downstream nodes"""
        return self._downstream

    def __rshift__(self, other):
        """wire self to other"""
        self.downstream().append(other)
        other.upstream().append(self)

    def __lshift__(self, other):
        """wire other to self"""
        other.downstream().append(self)
        self.upstream().append(other)

    def _get_dirty(self):
        return self._is_dirty

    def _set_dirty(self, val):
        self._reddd3g() if val else self._whited3g()
        self._is_dirty = val

    _dirty = property(_get_dirty, _set_dirty)

    def setDirty(self, myself=True):
        if myself:
            # set myself as dirty
            self._dirty = True

        # notify my downstream
        for node in self.downstream():
            node.setDirty()

    def isDirty(self):
        """Node needs to be re-evaluated, either because its value has changed
        or because its value *could* change"""

        return self._dirty

    def isDynamic(self):
        """Node isnt necessarily dirty, but needs to be reevaluated"""
        return self._dynamic

    def setValue(self, value):
        if self._compare(value, self._last_value):
            # don't set self to dirty since overwriting previous value
            self.setDirty(myself=False)

            # explicitly un-dirty me
            self._dirty = False

            # explicitly un-dynamic me
            if not hasattr(self, "_was_dynamic"):
                self._was_dynamic = self._dynamic
            self._dynamic = False

        self._last_value = value

    def set(self, **kwargs):
        """this method sets upstream dependencys' values to those given"""
        for k, v in kwargs.items():
            self.kwargs[k].setValue(v)

    def unlock(self):
        # reset my dynamic state
        self._dynamic = self._was_dynamic

    def _computeArgState(self, *argsTweaks, **kwargTweaks):
        """recompute, potentially applying tweaks"""
        # argsTweak can be positional varargs, or dict mapping node to value
        if argsTweaks and isinstance(argsTweaks[0], dict):
            # set flag
            pass_arg_tweaks_by_node = True

            # pass it through to upper function calls
            passThroughArgsTweaks = (argsTweaks[0],)

        else:
            # set flag
            pass_arg_tweaks_by_node = False

            # don't pass through to upper function calls, local var reference
            passThroughArgsTweaks = ()

        # will build these into a named tuple later
        args = []
        kwargs = {}
        varargs = ()
        varkwargs = {}

        # go through each parameter and try to respect the type it was defined as
        for param in self._parameters:
            # first check if self and handle separately
            if param.name == "self" and self._callable_is_method:
                kwargs["self"] = self._self_reference

            # check if overridden in argsTweaks
            elif (
                pass_arg_tweaks_by_node
                and self.args[param.position] in passThroughArgsTweaks[0]
            ):
                # tweaked by object reference, add to positionals
                args.append(passThroughArgsTweaks[0][self.args[param.position]])

            elif not pass_arg_tweaks_by_node and param.position < len(argsTweaks):
                # tweaked positionally, use tweak value
                args.append(argsTweaks[param.position])

            elif param.name in kwargTweaks:
                # tweaked by keyword, use tweak value
                kwargs[param.name] = kwargTweaks[param.name]

            else:
                # try to repect original function definition
                if param.kind == ParamType.POSITIONAL_ONLY:
                    # NOTE: only pass kwarg tweaks, cannot tweak via indirect position
                    args.append(
                        self.args[param.position](*passThroughArgsTweaks, **kwargTweaks)
                    )

                elif param.kind == ParamType.KEYWORD_ONLY:
                    # NOTE: only pass kwarg tweaks, cannot tweak via indirect position
                    kwargs[param.name] = self.kwargs[param.name](
                        *passThroughArgsTweaks, **kwargTweaks
                    )

                elif param.kind == ParamType.POSITIONAL_OR_KEYWORD:
                    # use keyword
                    # NOTE: only pass kwarg tweaks, cannot tweak via indirect position
                    kwargs[param.name] = self.kwargs[param.name](
                        *passThroughArgsTweaks, **kwargTweaks
                    )

                elif param.kind == ParamType.VAR_POSITIONAL:
                    # pass in by name without packing/unpacking
                    # NOTE: only pass kwarg tweaks, cannot tweak via indirect position
                    varargs = (
                        self.kwargs[param.name](*passThroughArgsTweaks, **kwargTweaks),
                    )

                elif param.kind == ParamType.VAR_KEYWORD:
                    # pass in by name without packing/unpacking
                    # NOTE: only pass kwarg tweaks, cannot tweak via indirect position
                    varkwargs = dict(
                        **self.kwargs[param.name](*passThroughArgsTweaks, **kwargTweaks)
                    )

        # validate arg state
        for i, arg in enumerate(args):
            if arg == StreamNone():
                raise TypeError(
                    "Must provide argument for {}".format(self.args[i].name)
                )
        for name, kwarg in kwargs.items():
            if kwarg == StreamNone():
                raise TypeError("Must provide argument for {}".format(name))

        # Use tuple and fronzendict for hashing state
        return ArgState(
            tuple(args), frozendict(kwargs), tuple(varargs), frozendict(varkwargs)
        )

    def _execute(self, args_state):
        return self._value(
            *args_state.args,
            *args_state.varargs,
            **args_state.kwargs,
            **args_state.varkwargs,
        )

    def _bind(self, other_self=None, *args, **kwargs):
        """This function binds an alternative `self` to the node's callable"""
        if other_self is not None:
            self._self_reference = other_self
        return self(*args, **kwargs)

    def _call(self, *argTweaks, **kwargTweaks):
        args_state = self._computeArgState(*argTweaks, **kwargTweaks)

        # when tweaking, dont save the results
        tweaking = argTweaks or kwargTweaks

        if self._dirty or self._dynamic or tweaking:
            # reexecute
            new_value = self._execute(args_state)
        else:
            new_value = self.value()

        if self._compare(value, self._last_value) and not tweaking:
            # push dirtinesss to downstream nodes
            self.setDirty(myself=False)

            # explicitly un-dirty me
            self._dirty = False

        if not tweaking:
            # update last value
            self._last_value = new_value

            # return
            return self.value()

        # otherwise don't manipulate state and just return the calculated value
        return new_value

    def __call__(self, *argTweaks, **kwargTweaks):
        # NOTE: use separate `_call` function so that expire and interval work properly
        return self._call(*argTweaks, **kwargTweaks)

    def value(self):
        return self._last_value

    def _gennode(self, name, func, func_args=(), **kwargs):
        if name not in self._node_op_cache:
            self._node_op_cache[name] = Node(
                name=name,
                value=func,
                args=func_args,
                **kwargs,
            )
        return self._node_op_cache[name]

    def _tonode(self, other):
        if isinstance(other, Node):
            return other
        if str(other) not in self._node_op_cache:
            self._node_op_cache[str(other)] = Node(
                value=other, name="var(" + str(other) + ")"
            )
        return self._node_op_cache[str(other)]


@_either_type
def node(meth, **attribute_kwargs):
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

    # TODO attribute kwargs into nodes
    new_node = Node(value=meth)
    if new_node._callable_is_method:
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
