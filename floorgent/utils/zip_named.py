from types import SimpleNamespace


def zip_named(*, cls=SimpleNamespace, **named_iterables):
    "Zip-like but return SimpleNamespaces"
    named_iters = {k: iter(v) for k, v in named_iterables.items()}
    while True:
        kw = {}
        stop = True
        for k, it in named_iters.items():
            try:
                kw[k] = next(it)
            except StopIteration:
                kw[k] = None
            else:
                stop = False
        if stop:
            break
        yield cls(**kw)
