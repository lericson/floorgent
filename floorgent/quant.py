import numpy as np
from .config import dequant_min, dequant_max, quant_min, quant_max


dequant_range = dequant_max - dequant_min
quant_range   = quant_max   - quant_min


def quantize(verts, check=True):
    "Map [dequant_min, dequant_max] to integers in [quant_min, quant_max]."
    if check:
        assert np.all(dequant_min <= verts), verts
        assert np.all(verts <= dequant_max), verts
    verts_quantize = (verts - dequant_min) / dequant_range * quant_max + quant_min
    verts_quantize += 0.5
    verts_quantize_int = verts_quantize.astype('int32')
    if check:
        assert np.all(quant_min <= verts_quantize_int)
        assert np.all(verts_quantize_int <= quant_max)
    return verts_quantize_int


def dequantize(verts):
    "Invert quantize(verts)."
    verts = verts.astype('float32')
    verts = (verts - quant_min) / quant_range * dequant_range + dequant_min
    return verts
