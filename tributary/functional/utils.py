
def split(data, callback1, callback2):
    return (callback1(data), callback2(data))


def merge(data1, data2, callback):
    return callback((data1, data2))
