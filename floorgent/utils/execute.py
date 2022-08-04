import os
import warnings
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from concurrent.futures import as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def execute_seq(jobs, *, print_progress=True):
    if tqdm:
        jobs = tqdm(jobs)
    else:
        warnings.warn('print_progress=True but tqdm not installed!')
    return [f(*a) for f, *a in jobs]


def execute_many(jobs, *, pool_type='process', print_progress=True):

    n = len(jobs)

    pool_cls = {'process': ProcessPoolExecutor,
                'thread': ThreadPoolExecutor}[pool_type]

    with pool_cls(max_workers=min(20, os.cpu_count() - 2)) as executor:

        futs = {executor.submit(f, *args): i for (i, (f, *args)) in enumerate(jobs)}
        completed = as_completed(futs)
        res = [None]*n

        if print_progress:
            if tqdm:
                completed = tqdm(completed, total=n)
            else:
                warnings.warn('print_progress=True but tqdm not installed!')

        for fut in completed:
            i = futs[fut]
            try:
                res_i = fut.result()
            except Exception:
                print(f'{jobs[i]} failed')
                import traceback
                traceback.print_exc()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                res[i] = res_i

    return res


#__import__('warnings').warn('hotpatched execute_many')
#execute_many = execute_seq
