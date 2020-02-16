from .node import BaseNode, node  # noqa: F401


class BaseGraph(object):
    '''Wrapper class around a collection of lazy nodes.'''
    def __init__(self, *args, **kwargs):
        pass

    def node(self, name, readonly=False, nullable=True, value=None, trace=False):  # noqa: F811
        '''method to create a lazy node attached to a graph.

        Args:
            name (str): name to represent the node
            readonly (bool): whether the node should be settable
            nullable (bool): whether node can have value None
            value (any): initial value for node
            trace (bool): trace the calls for a node
        Returns:
            BaseNode: the newly constructed lazy node
        '''
        if not hasattr(self, '_BaseGraph__nodes'):
            self.__nodes = {}

        if name not in self.__nodes:
            self.__nodes[name] = BaseNode(name=name,
                                       derived=False,
                                       readonly=readonly,
                                       nullable=nullable,
                                       value=value,
                                       trace=trace)
            setattr(self, name, self.__nodes[name])
        return self.__nodes[name]

    def __getattribute__(self, name):
        if name == '_BaseGraph__nodes' or name == '__nodes':
            return super(BaseGraph, self).__getattribute__(name)
        elif hasattr(self, '_BaseGraph__nodes') and name in super(BaseGraph, self).__getattribute__('_BaseGraph__nodes'):
            return super(BaseGraph, self).__getattribute__('_BaseGraph__nodes')[name]
        else:
            return super(BaseGraph, self).__getattribute__(name)

    def __setattr__(self, name, value):
        if hasattr(self, '_BaseGraph__nodes') and name in super(BaseGraph, self).__getattribute__('_BaseGraph__nodes'):
            node = super(BaseGraph, self).__getattribute__('_BaseGraph__nodes')[name]
            if isinstance(value, BaseNode) and node == value:
                return
            elif isinstance(value, BaseNode):
                raise Exception('Cannot set to node')
            else:
                node._dirty = (node._value != value) or (node._value is not None and abs(node._value - value) > 10**-5)
                node._value = value
        else:
            super(BaseGraph, self).__setattr__(name, value)


def construct(dag):
    '''dag is representation of the underlying DAG'''
