
class _Graph(object):
    '''internal representation of the entire graph state'''

    def __init__(self, node):
        self._starting_node = node

    def getNodes(self):
        return self._starting_node._deep_bfs()
