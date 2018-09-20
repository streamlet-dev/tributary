
def split(data, callback1, callback2):
    return (callback1(data), callback2(data))


def map(data, *callbacks):
    return (callback(data) for callback in callbacks)


def merge(data1, data2, callback):
    return callback((data1, data2))


def reduce(callback, *datas):
    return callback((data for data in datas))
