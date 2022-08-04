"Birds-Eye View rotation"

import numpy as np
from tensorflow.compat import v1 as tf

from .utils import name_scoped
from .bilintf import interp2_bilinear_wrap
from .panorotf import assert_finite, assert_between

tf.disable_eager_execution()

tf_float = tf.float32
np_float = np.float32


@name_scoped
def reproject_points(R, I, J, h, w):

    with tf.name_scope('project'):
        # X, Y \in [-1.0, +1.0]
        X =   2/w*J - 1
        Y = -(2/h*I - 1)
        del I, J

        P = tf.stack([X, Y], axis=-1)
        del X, Y

    with tf.name_scope('transform'):
        P_ = P @ tf.transpose(R[0:2, 0:2])
        X_, Y_ = tf.unstack(P_, axis=-1)
        del P, P_

    with tf.name_scope('invert'):
        I_ = tf.mod(h/2*(-Y_ + 1), h)
        J_ = tf.mod(w/2*( X_ + 1), w)
        del X_, Y_

    return I_, J_


@name_scoped
def reproject_birdseye(R, inp, interpolate=True):
    h, w, _ = inp.shape[-3:]
    I,  J  = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')
    I, J   = tf.cast(I, tf_float), tf.cast(J, tf_float)
    # Note that we transpose R since we're gathering from the resulting points,
    # thus we actually need the INVERSE rotation of R (i.e. R transpose.)
    I_, J_ = reproject_points(tf.transpose(R), I, J, tf.cast(h, tf_float), tf.cast(w, tf_float))
    query_points = tf.stack([I_, J_], axis=-1)
    if interpolate:
        return interp2_bilinear_wrap(inp, query_points)
    else:
        return tf.gather_nd(inp, tf.cast(query_points, tf.int32))
