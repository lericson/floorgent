def split_pct(L, pct):
    N         = len(L)
    split_idx = int(N*pct)
    assert 0 < split_idx < (N - 1)
    return L[:split_idx], L[split_idx:]
