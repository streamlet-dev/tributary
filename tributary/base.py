
class StreamEnd:
    '''Indicates that a stream has nothing left in it'''
    pass


class StreamNone:
    '''indicates that a stream does not have a value'''
    def __init__(self, last):
        self.value = last
