
def split(data, callback1, callback2):
    '''Pass data to 2 callbacks

    Args:
        data (any): data to pass to both callbacks
        callback1 (callable): first function to call
        callback2 (callable): second function to call
    '''
    return (callback1(data), callback2(data))


def map(data, *callbacks):
    '''Pass data to multiple callbacks

    Args:
        data (any): data to pass to all callbacks
        callbacks (tuple): callbacks to pass data to
    '''
    return (callback(data) for callback in callbacks)


def merge(data1, data2, callback):
    '''merge two data sources into one callback

    Args:
        data1 (any): first data to pass to callback
        data2 (any): second data to pass to callback
        callback (callable): callback to pass data to
    '''
    return callback((data1, data2))


def reduce(callback, *datas):
    '''merge multiple data sources into one callback

    Args:
        callback (callable): callback to pass data to
        datas (tuple): data to pass to callback
    '''
    return callback((data for data in datas))
