import time
import numpy as np
from tensorflow.compat import v1 as tf

from .utils import name_scoped
from .bilintf import interp2_bilinear_wrap


tf.disable_eager_execution()

tf_float = tf.float32
np_float = np.float32


@name_scoped
def rotmat_from_euler(roll, pitch, yaw):
    "Convention is x'-y''-z''', or roll, then pitch, then yaw."
    c1, c2, c3 = tf.unstack(tf.cos([roll, pitch, yaw]))
    s1, s2, s3 = tf.unstack(tf.sin([roll, pitch, yaw]))
    R = tf.stack([[           c2*c3,            -c2*s3,     s2],
                  [c1*s3 + c3*s1*s2,  c1*c3 - s1*s2*s3, -c2*s1],
                  [s1*s3 - c1*c3*s2,  c3*s1 + c1*s2*s3,  c1*c2]])
    return R


@name_scoped
def assert_finite(tensor):
    #assert tf.reduce_all(tf.is_finite(a))
    return tf.Assert(tf.reduce_all(tf.is_finite(tensor)), [tensor])


@name_scoped
def assert_between(tensor, lo, hi, strict=False):
    if strict:
        cond = tf.reduce_all(tensor < hi) & tf.reduce_all(lo < tensor)
    else:
        cond = tf.reduce_all(tensor <= hi) & tf.reduce_all(lo <= tensor)
    return tf.Assert(cond, [tensor])


@name_scoped
def reproject_points(R, I, J, h, w):

    with tf.name_scope('project'):

        # Long[j] \in [-π, +π)
        Long = (2*J/w - 1)*np.pi
        Lat  = (2*I/h - 1)*np.pi/2
        del I, J

        X = tf.cos(Long) * tf.cos(Lat)
        Y = tf.sin(Long) * tf.cos(Lat)
        Z = -tf.sin(Lat)
        del Long, Lat

        P = tf.stack([X, Y, Z], axis=-1)
        del X, Y, Z
        #Pnorm = tf.norm(P, axis=-1)
        #Phat = P / Pnorm
        #del P, Pnorm

    with tf.name_scope('transform'):
        #Phat_ = tf.matmul(Phat, tf.transpose(R))
        #del Pnorm, Phat, X, Y, Z
        P_ = P @ tf.transpose(R)
        Pnorm_ = tf.norm(P_, keepdims=True, axis=-1)
        Phat_ = P_ / Pnorm_
        Xhat_, Yhat_, Zhat_ = tf.unstack(Phat_, axis=-1)
        del P, P_, Pnorm_, Phat_

    with tf.name_scope('invert'):
        assertions = [assert_finite(Xhat_, name='finite_Xhat_'),
                      assert_finite(Yhat_, name='finite_Yhat_'),
                      assert_finite(Zhat_, name='finite_Zhat_'),
                      assert_between(Xhat_, -1.0, +1.0, name='range_Xhat_'),
                      assert_between(Yhat_, -1.0, +1.0, name='range_Yhat_'),
                      assert_between(Zhat_, -1.0, +1.0, name='range_Zhat_')]
        with tf.control_dependencies(assertions):
            Long_ = tf.atan2(Yhat_, Xhat_)
            Lat_  = tf.asin(-Zhat_)
            del Xhat_, Yhat_, Zhat_
        with tf.control_dependencies([assert_finite(Long_, name='finite_Long_'),
                                      assert_finite(Lat_, name='finite_Lat_')]):
            J_ = tf.mod((w/2*(Long_/np.pi    + 1)), w)
            I_ = tf.mod((h/2*(Lat_/(np.pi/2) + 1)), h)
            del Long_, Lat_

    return I_, J_


@name_scoped
def reproject_panorama(R, inp, interpolate=True):
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


@name_scoped
def random_rotmat(*, stddev_rp=0.0, stddev_yaw=0.0):
    if stddev_rp == 0.0 and stddev_yaw == 0.0:
        return tf.eye(3)

    roll = pitch = yaw = 0.0

    if stddev_rp != 0.0:
        roll  = tf.random.truncated_normal(shape=(), stddev=stddev_rp)
        pitch = tf.random.truncated_normal(shape=(), stddev=stddev_rp)

    if stddev_yaw != 0.0:
        yaw   = tf.random.truncated_normal(shape=(), stddev=stddev_yaw)

    return rotmat_from_euler(roll, pitch, yaw)


@name_scoped
def binom(shape=(), *, p=0.5, dtype=None):
    z = tf.random.uniform(shape=shape) < p
    if dtype is not None:
        z = tf.cast(z, dtype)
    return z


@name_scoped
def random_signed_perm():
    perm_xy = binom(dtype=tf.int32)
    R_perm = tf.stack([[1 - perm_xy,     perm_xy, 0],
                       [    perm_xy, 1 - perm_xy, 0],
                       [          0,           0, 1]])

    reflect_x = 2*binom(dtype=tf.int32) - 1
    reflect_y = 2*binom(dtype=tf.int32) - 1
    R_sign = tf.stack([[reflect_x,         0, 0],
                       [        0, reflect_y, 0],
                       [        0,         0, 1]])

    return tf.cast(R_perm @ R_sign, tf_float)


def main():
    from matplotlib import pyplot as plt
    from utils import print_arr

    inp_np = plt.imread('pano_mp.png')
    #inp_np = inp_np[::4, ::4, :]

    inp_shape = h, w, c = inp_np.shape

    assert 2*h == w

    inp = tf.placeholder(name='inp', shape=(h, w, c), dtype=tf_float)

    # We set the stddev at 3 sigmas, so obtaining less than x degrees is a 3-sigma
    # event (i.e. about 98%). The normal distribution is truncated at 2-sigma;
    # samples drawn outside that region are redrawn. Do not yaw since we use signed
    # permutations for that.
    R = random_rotmat(stddev_rp=np.radians(2.5)/3, stddev_yaw=0.0)
    R = random_signed_perm() @ R

    out = reproject_panorama(R, inp)

    #plt.figure()
    #plt.imshow(inp_np)

    import json
    with open('pano_mp.json', 'r') as fp:
        json_doc = json.load(fp)

    zc = json_doc['cameraHeight']
    zh = json_doc['layoutHeight']
    points_xyz = np.array([p['xyz'] for p in json_doc['layoutPoints']['points']])
    points_xyz = points_xyz @ [[ 0, 1,  0],
                               [ 0, 0, -1],
                               [-1, 0,  0]]
    points_xyz = points_xyz - (0, 0, zc)
    #print_arr(points_xyz)
    #print_arr(np.array( [[-p['xyz'][2], p['xyz'][0], (-0*p['xyz'][1] - zc)] for p in json_doc['layoutPoints']['points']] ))
    #exit()
    xyz_lines = []
    for wall in json_doc['layoutWalls']['walls']:
        xyz_line = [points_xyz[ind] for ind in wall['pointsIdx']]
        xyz_lines.append(xyz_line)

    with tf.Session() as sess:
        for i in range(10):
            print()
            print(f'frame {i}')
            out_np, R_np = sess.run((out, R), {inp: inp_np})
            print_arr('R_np', R_np)

            print('drawing walls and projecting to panorama')
            for xyz_line in xyz_lines:
              for a, b in zip(xyz_line[:-1], xyz_line[1:]):
                print(np.array([a, b]))
                points = np.linspace(a, b, 1000)
                norm = np.linalg.norm(points, axis=-1, keepdims=True)
                #print_arr('points', points)
                #print_arr('norm', norm)
                points /= norm
                points = points @ R_np
                X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
                Long = np.arctan2(Y, X)
                Lat  = np.arcsin(-Z)
                J = np.mod((w/2*(Long/np.pi    + 1)), w)
                I = np.mod((h/2*(Lat/(np.pi/2) + 1)), h)
                out_np[np.floor(I).astype(np.int32),
                       np.floor(J).astype(np.int32), :] = [0, 1.0, 0]

            out_fn = f'frame{i:02d}.png'
            print(f'writing to {out_fn}')
            plt.imsave(out_fn, out_np)
            #plt.figure()
            #plt.imshow(out_np)

    #plt.show()
    raise SystemExit


    t0 = time.time()
    print('begin')
    I,  J  = np.mgrid[0:h, 0:w]
    I_, J_ = project(R_np, I, J)
    out_np = inp_np[I_, J_]
    t1 = time.time()
    print('end', t1-t0)

    plt.figure()
    plt.imshow(inp_np)
    plt.figure()
    plt.imshow(out_np)
    plt.show()


if __name__ == "__main__":
    main()
