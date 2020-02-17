
class StreamEnd:
    '''Indicates that a stream has nothing left in it'''
    pass


class StreamRepeat:
    '''Indicates that a stream has a gap, this object should be ignored
    and the previous action repeated'''
    pass


class StreamNone:
    '''indicates that a stream does not have a value'''

    def __init__(self, last=None):
        self.value = last

    def all_bin_ops(self, other):
        return self

    def all_un_ops(self):
        return self

    __add__ = all_bin_ops
    __radd__ = all_bin_ops
    __sub__ = all_bin_ops
    __rsub__ = all_bin_ops
    __mul__ = all_bin_ops
    __rmul__ = all_bin_ops
    __div__ = all_bin_ops
    __rdiv__ = all_bin_ops
    __truediv__ = all_bin_ops
    __rtruediv__ = all_bin_ops
    __pow__ = all_bin_ops
    __rpow__ = all_bin_ops
    __mod__ = all_bin_ops
    __rmod__ = all_bin_ops
    __and__ = all_bin_ops
    __or__ = all_bin_ops
    __invert__ = all_un_ops
    def __bool__(self): return False
    def int(self): return 0
    def float(self): return 0
    __lt__ = all_bin_ops
    __le__ = all_bin_ops
    __gt__ = all_bin_ops
    __ge__ = all_bin_ops
    __eq__ = all_bin_ops
    __ne__ = all_bin_ops
    __neg__ = all_un_ops
    __nonzero__ = all_un_ops
    __len__ = all_un_ops
