def _multiplex_dispatch(list_attr, meth_name):
    def multiplexer(self, *a, **kw):
        others = getattr(self, list_attr)
        return [getattr(other, meth_name)(*a, **kw) for other in others]
    multiplexer.__name__ = meth_name
    multiplexer.__doc__ = f'Call {meth_name}(...) on objects in {list_attr}'
    return multiplexer


class Tee:
    "Write to many files with one file"

    def __init__(self, *file_objects):
        self.file_objects = file_objects

    write      = _multiplex_dispatch('file_objects', 'write')
    close      = _multiplex_dispatch('file_objects', 'close')
    flush      = _multiplex_dispatch('file_objects', 'flush')
    __enter__  = _multiplex_dispatch('file_objects', '__enter__')
    __exit__   = _multiplex_dispatch('file_objects', '__exit__')
    writelines = _multiplex_dispatch('file_objects', 'writelines')
