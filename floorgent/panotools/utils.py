from functools import wraps


def name_scoped(f=None, *, default_name=None):

    def deco(f):

        nonlocal default_name

        from tensorflow.compat import v1 as tf

        if default_name is None:
            default_name = f.__name__
        elif default_name is False:
            default_name = None

        @wraps(f)
        def inner(*args, name=None, **kwds):
            with tf.name_scope(name, default_name):
                return f(*args, **kwds)

        return inner

    return deco(f) if f else deco


# ------------- 8< -------------
import numpy as np


def print_arr(*args, axis=0):
    *text, arr = args
    arr = np.asarray(arr)
    if text:
        print(*text)
    if arr.size < 100:
        print(arr)
    else:
        print(f'{arr.dtype} {arr.shape}')
        print('mean, std:', arr.mean(axis=axis), arr.std(axis=axis))
        print('min, max:', arr.min(axis=axis), arr.max(axis=axis))
