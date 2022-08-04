import numpy as np
import tensorflow._api.v1.compat.v1 as tf
import tensorflow_probability as tfp

from . import config
from .config import quant_min
from .config import quant_max
from .config import dequant_min
from .config import dequant_max
from .config import tuplet_size
from .config import shift_factor
from .config import yaw_stddev_deg
from .config import pitch_roll_stddev_deg
from .polylines import code_tokens
from .polylines import sequence_end_token
from .panotools.utils import name_scoped
from .panotools.panorotf import random_rotmat
from .panotools.panorotf import random_signed_perm
from .panotools.panorotf import reproject_panorama
from .panotools.bevrotf import reproject_birdseye


tfd = tfp.distributions
tfb = tfp.bijectors

assert quant_min == 0

dequant_range = dequant_max - dequant_min
quant_range   = quant_max   - quant_min

pitch_roll_stddev = np.radians(pitch_roll_stddev_deg)
yaw_stddev        = np.radians(yaw_stddev_deg)


def quantize(verts):
    "TF variant of quant.quantize"
    verts_quantize = (verts - dequant_min) / dequant_range * quant_max + quant_min
    return tf.cast(verts_quantize + 0.5, tf.int32)


def dequantize(verts):
    verts = tf.cast(verts, tf.float32)
    verts = (verts - quant_min) / quant_range * dequant_range + dequant_min
    return verts


def rotation_basis(x, y):
  "Compute a rotation that maps (x, y) -> (|x, y|, 0)."
  z = tf.linalg.norm((x, y))
  return tf.concat((tf.stack(( x, y)),
                    tf.stack((-y, x))))/z


@name_scoped
def random_shift(vertices):
    """Apply random shift to vertices."""

    assert config.conditioning == 'none', 'Cannot shift jitter with this conditioning'

    # Margin between maximum coordinate and largest quantizable value per dimension.
    max_shift_pos = tf.cast(quant_max - tf.reduce_max(vertices, axis=0), tf.float32)
    max_shift_pos = tf.maximum(max_shift_pos, 1e-9)

    # Margin between minimum coordinate and smallest quantizable value per dimension.
    max_shift_neg = tf.cast(tf.reduce_min(vertices, axis=0), tf.float32)
    max_shift_neg = tf.maximum(max_shift_neg, 1e-9)

    # Sample a 2D shift inside this margin. Guarantees that all vertices remain
    # inside quantizable region.
    shift = tfd.TruncatedNormal(tf.zeros([1, 2]), shift_factor*quant_max,
                                -max_shift_neg, max_shift_pos).sample()
    shift = tf.cast(shift, tf.int32)
    vertices += shift
    return vertices


@name_scoped
def jitter_triplets(ex):
    R_p = random_signed_perm()
    R_r = random_rotmat(stddev_rp=pitch_roll_stddev,
                        stddev_yaw=yaw_stddev)
    R = ex['rotmat'] = R_p @ R_r

    ex['triplets'] = rotate_triplets(R, ex['triplets'])

    if config.conditioning == 'none':
        #verts = random_shift(verts)
        pass
    elif config.conditioning == 'pano':
        ex['image'] = reproject_panorama(R, ex['image'])
    elif config.conditioning == 'bev':
        ex['image'] = reproject_birdseye(R, ex['image'])

    return ex


def rotate_triplets(R, triplets):
    opcodes = triplets[..., 0]
    verts   = triplets[..., 1:]

    # Apply our rotation matrix to the vertices. Since the vertices lie in the
    # XY plane at Z = 0 and are projected back onto the XY plane after
    # rotation, we can simply ignore the Z coordinates.
    verts_dq = dequantize(verts)
    verts_dq = verts_dq @ tf.transpose(R[0:2, 0:2])
    verts    = quantize(verts_dq)
    del verts_dq

    with tf.control_dependencies([tf.assert_less_equal(verts, quant_max)]):
        # Zero vertices of EOS ops
        cond  = tf.not_equal(opcodes, sequence_end_token)
        verts = tf.where(cond, verts, tf.zeros_like(verts))
        return tf.concat([opcodes[..., None], verts], axis=-1)


@name_scoped
def tokens_from_triplets(ex):
    opcodes   = ex['triplets'][..., 0]
    verts     = ex['triplets'][..., 1:]
    verts_adj = verts + len(code_tokens)
    cond      = tf.not_equal(opcodes, sequence_end_token)
    verts_adj = tf.where(cond, verts_adj, tf.zeros_like(verts_adj))
    with tf.control_dependencies([tf.assert_less(opcodes, len(code_tokens)),
                                  tf.assert_less_equal(verts, quant_max)]):
        if (tuplet_size % 3) == 0:
            triplets_adj = tf.concat([opcodes[..., None], verts_adj], axis=-1)
            ex['tokens'] = tf.reshape(triplets_adj, [-1])
        elif (tuplet_size % 2) == 0:
            ex['tokens'] = tf.reshape(verts_adj, [-1])
        else:
            raise ValueError(f'cannot make tokens from tuplet_size {tuplet_size!r}')
        ex['tokens_mask'] = tf.ones_like(ex['tokens'])
    return ex


def dataset_from_ex_list(ex_list):
    defs = {'triplets':  (tf.int32,   'triplets',  tf.TensorShape([None, 3])),
            'segs_norm': (tf.float64, 'segs_norm', tf.TensorShape([None, 2, 2])),
            'segs_full': (tf.float64, 'segs_full', tf.TensorShape([None, 2, 2])),
            'image':     (tf.float32, 'image',     tf.TensorShape(config.image_shape))}

    if config.conditioning == 'none':
        del defs['image']

    exs = [{k: getattr(ex, attr) for (k, (_, attr, _)) in defs.items()} for ex in ex_list]
    output_types  = {k: typ   for (k, (typ, _, _))   in defs.items()}
    output_shapes = {k: shape for (k, (_, _, shape)) in defs.items()}

    return tf.data.Dataset.from_generator(
        lambda: iter(exs),
        output_types=output_types,
        output_shapes=output_shapes)


@name_scoped
def get_batch(ds, *, batch_size, jitter=True, repeat=True, prefetch=4):
    if jitter:
        jittered_ds = ds.map(jitter_triplets)
    else:
        jittered_ds = ds

    tokens_ds = jittered_ds.map(tokens_from_triplets)

    if repeat:
        tokens_ds = tokens_ds.repeat()

    batched_ds = tokens_ds.padded_batch(batch_size, padded_shapes=tokens_ds.output_shapes)

    if prefetch:
        batched_ds = batched_ds.prefetch(prefetch)

    return batched_ds.make_one_shot_iterator().get_next()


@name_scoped
def batch_loss(model, batch, *, is_training=True):
    tokens      = batch['tokens']
    tokens_mask = tf.cast(batch['tokens_mask'], dtype=tf.bool)

    log_probs = model(batch, is_training=is_training).log_prob(tokens)
    log_probs = tf.boolean_mask(log_probs, tokens_mask)
    loss = -tf.reduce_mean(log_probs)

    return {**batch,
            'tokens_mask': tokens_mask,
            'log_probs': log_probs,
            'loss': loss}
