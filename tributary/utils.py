import functools


def _either_type(f):
    '''Utility decorator to allow for either no-arg decorator or arg decorator

    Args:
        f (callable): Callable to decorate
    '''
    @functools.wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)
    return new_dec
