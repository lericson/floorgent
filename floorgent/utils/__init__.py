import numpy as np

from .tee import Tee
from .execute import execute_seq
from .execute import execute_many
from .split_pct import split_pct
from .zip_named import zip_named


def implies(p, q):
    return (not p) or q


def statstr(L):
    return f'µ={np.mean(L):.5g}, σ={np.std(L):.5g}, min={np.min(L):.5g}, max={np.max(L):.5g}'


__all__ = ['Tee', 'zip_named', 'split_pct', 'execute_seq', 'execute_many',
           'implies', 'statstr']
