from tensorflow.compat import v1 as tf
from .utils import name_scoped


@name_scoped
def interp2_bilinear_wrap(grid, query_points, name=None):
    "Bilinear interpolation that wraps around the edges."
    # Inspired by interpolate_bilinear from tensorflow-addons.

    assert len(grid.shape) > 2

    grid         = tf.convert_to_tensor(grid)
    query_points = tf.convert_to_tensor(query_points)

    q_floor_float = tf.math.floor(query_points)
    q_floor = tf.cast(q_floor_float, tf.dtypes.int32)
    q_floor = q_floor % grid.shape[:-1]
    q_ceil = q_floor + 1
    q_ceil = q_ceil % grid.shape[:-1]

    # Compute row and column-wise alphas, the interpolation factor, by how
    # far from the pixel's center the query point is.
    alphas = query_points - q_floor_float

    @name_scoped
    def gather(params, *coords):
        coords = tf.stack(coords, axis=-1)
        return tf.gather_nd(grid, coords)

    # Gather 2x2 pixel neighborhood
    top_left     = gather(grid, q_floor[..., 0], q_floor[..., 1], name='top_left')
    top_right    = gather(grid, q_floor[..., 0],  q_ceil[..., 1], name='top_right')
    bottom_left  = gather(grid,  q_ceil[..., 0], q_floor[..., 1], name='bottom_left')
    bottom_right = gather(grid,  q_ceil[..., 0],  q_ceil[..., 1], name='bottom_right')

    with tf.name_scope('interp_top'):
        interp_top    = alphas[..., 1, None] * (   top_right -    top_left) +    top_left

    with tf.name_scope('interp_bottom'):
        interp_bottom = alphas[..., 1, None] * (bottom_right - bottom_left) + bottom_left

    with tf.name_scope('interp'):
        interp = alphas[..., 0, None] * (interp_bottom - interp_top) + interp_top
        interp = tf.cast(interp, grid.dtype)

    return interp
