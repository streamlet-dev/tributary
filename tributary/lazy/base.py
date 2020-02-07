from .node import _Node, node  # noqa: F401


class BaseClass(object):
    def __init__(self, *args, **kwargs):
        pass

    def node(self, name, readonly=False, nullable=True, default_or_starting_value=None, trace=False):  # noqa: F811
        if not hasattr(self, '_BaseClass__nodes'):
            self.__nodes = {}

        if name not in self.__nodes:
            self.__nodes[name] = _Node(name=name,
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
            if isinstance(value, _Node) and node == value:
                return
            elif isinstance(value, _Node):
                raise Exception('Cannot set to node')
            else:
                node._dirty = (node._value != value) or (node._value is not None and abs(node._value - value) > 10**-5)
                node._value = value
        else:
            super(BaseClass, self).__setattr__(name, value)


def construct(dag):
    '''dag is representation of the underlying DAG'''
