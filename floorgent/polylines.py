import warnings
import collections
from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Optional
from functools import wraps
from itertools import islice

import numpy as np
import numba as nb
from matplotlib import pyplot as plt
from matplotlib import patheffects as mpe
from matplotlib import patches as mpatches
from matplotlib.path import Path as MPath
from scipy.spatial import distance_matrix

from . import stats
from . import quant
from . import floorplans


def compose(f):
  def composer(g):
    @wraps(g)
    def inner(*a, **k):
      return f(g(*a, **k))
    return inner
  return composer


def njit(*a, **kw):
    kw.setdefault('cache', True)
    kw.setdefault('nogil', True)
    kw.setdefault('nopython', True)
    kw.setdefault('parallel', False)
    return nb.jit(*a, **kw)


class PathBuilder:
    def __init__(self, vertices=None, codes=None):
        self.vertices = list(vertices) if vertices else []
        self.codes    = list(codes)    if codes    else []

    def move_to(self, vertex):
        self.vertices.append(vertex)
        self.codes.append(MPath.MOVETO)

    def line_to(self, vertex):
        self.vertices.append(vertex)
        self.codes.append(MPath.LINETO)

    @property
    def last_vertex(self):
        return self.vertices[-1]

    def to_path(self):
        return MPath(self.vertices, self.codes)

    @classmethod
    def start_at(cls, vert):
        return cls([vert], [MPath.MOVETO])


def find_center_scale(vertices):
  vert_min = vertices.min(axis=0)
  vert_max = vertices.max(axis=0)
  center = 0.5 * (vert_min + vert_max)
  aabb = vert_max - vert_min
  scale = np.max(np.abs(aabb))
  return center, scale


def normalize_vertices(vertices):
    center, scale = find_center_scale(vertices)
    return (vertices - center) / scale


def floorplan_to_path(fplan: floorplans.FloorPlan, *, recenter=True) -> MPath:
    lines = [line for s in fplan.spaces for line in s.lines
             if line.kind.casefold() != 'portal']

    pb = PathBuilder.start_at(lines[0].start)

    for line in lines:
        if not np.allclose(line.start, pb.last_vertex):
            pb.move_to(line.start)
        pb.line_to(line.end)

    path = pb.to_path()

    if recenter:
        center, _ = find_center_scale(path.vertices)
        path.vertices -= center

    return path


code_names = {MPath.MOVETO: 'M',
              MPath.LINETO: 'L'}
def _label_for_path(path, *, label):
    counts = collections.Counter(path.codes)
    parts = [f'{counts.pop(code, 0)}{name}' for code, name in code_names.items()]
    if counts:
        parts.append(f'?i{sum(counts.values())}')
    lines = [label,
             f'{len(path.vertices)} vertices',
             ' '.join(parts)]
    return '\n'.join(filter(None, lines))


def plot_path(path, *, autoscale=True, ax=None, fig=None, text=True, vert_size=6,
              label=None, text_x=0.02, text_y=0.98, text_va='top', shift=0.0,
              color='C0', zorder=0, **kw):

    fig = plt.gcf() if fig is None else fig
    ax  = fig.gca() if ax is None else ax

    kw.setdefault('facecolor', 'None')

    path = MPath(path.vertices + shift, path.codes)
    ax.add_patch(mpatches.PathPatch(path, edgecolor=color, **kw))

    plot_verts = np.unique(path.vertices, axis=0)
    ax.scatter(plot_verts[:, 0], plot_verts[:, 1], s=vert_size,
               color=color, label=label, **kw)

    if autoscale:
        ax.autoscale()

    if text:
        s = _label_for_path(path, label=label)
        t = ax.text(text_x, text_y, s, transform=ax.transAxes,
                    va=text_va, color=color, alpha=kw.get('alpha'),
                    zorder=kw.get('zorder', 0)+1e3)
        shadow = mpe.withSimplePatchShadow(offset=(1, -1),
                                           alpha=0.8*kw.get('alpha', 1.0))
        t.set_path_effects([shadow])


def plot_paths(paths, fig_size=(4.0, 4/3), n_cols=3, **kw):
    paths = list(paths)
    n_plot = len(paths)
    n_cols = np.minimum(n_plot, n_cols)
    n_rows = int(np.ceil(n_plot / n_cols))
    paths += [None] * (n_cols * n_rows - n_plot)
    fig, axs = plt.subplots(n_rows, n_cols, sharex='all', sharey='all', squeeze=False,
                            subplot_kw=dict(adjustable='box', aspect='equal'),
                            figsize=fig_size*np.r_[n_cols, n_rows])
    for i, path in enumerate(paths):
        row, col = np.divmod(i, n_cols)
        ax = axs[row, col]
        if path is not None:
            plot_path(path, ax=ax, fig=fig, **kw)
        else:
            axs[row, col].text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                               va='center', ha='center')
    fig.subplots_adjust(left=0., right=1., bottom=0., top=1., wspace=0.025, hspace=0.025)


# Choose 0 as EOS so zero-padding just adds EOS tokens.
sequence_end_token = 0
line_start_token   = 1
next_vertex_token  = 2
code_tokens        = {sequence_end_token, line_start_token, next_vertex_token}


T = TypeVar('T', int, float)
Pos = Tuple[T, T]
PolyLineSet   = List[List[Pos]]
TokenTriplet  = Tuple[int, int, int]
TokenTriplets = List[TokenTriplet]


def quantize_path(path: MPath, *, return_stats=False):
    verts = path.vertices
    center, scale = find_center_scale(verts)
    verts_norm  = (verts - center) / scale
    verts_quant = quant.quantize(verts_norm)
    qpath = MPath(verts_quant, path.codes)
    if not return_stats:
        return qpath
    else:
        return qpath, center, scale


def dequantize_path(path: MPath):
    verts = quant.dequantize(np.array(path.vertices))
    return MPath(verts, path.codes)


def path_to_polylines(path: MPath) -> PolyLineSet:
    polylines = []
    for code, vert in zip(path.codes, path.vertices):
        if code == MPath.MOVETO:
            polylines.append([])
        elif code == MPath.LINETO:
            prev_vert = polylines[-1][-1]
            # Some segments may have quantized to zero length.
            if np.allclose(vert, prev_vert):
                continue
        else:
            0/0

        polylines[-1].append(vert)

    return [np.array(line) for line in polylines]


def polylines_to_path(polylines: PolyLineSet) -> MPath:
    "Polylines into a Path"
    codes = []
    verts = []
    for line in polylines:
        codes.append(MPath.MOVETO)
        codes.extend([MPath.LINETO] * (len(line) - 1))
        verts.extend(line)
    return MPath(np.array(verts), codes)


@compose(np.asarray)
def split_segments(polylines: PolyLineSet) -> PolyLineSet:
    "Split set of polylines into set of constituent line segments"
    return [[a, b] for line in polylines for a, b in zip(line[:-1], line[1:])]


def stitch_segments(polylines: PolyLineSet) -> PolyLineSet:
    "Stitch together polylines that end where the previous started"
    out = []
    for line in polylines:
        if not (out and np.allclose(line[0], out[-1][-1])):
            out.append([])
        out[-1].extend(line)
    return [np.array(line) for line in out]


@compose(list)
def remove_segments(polylines: PolyLineSet, *, min_area=1e-4) -> PolyLineSet:
    "Remove line segments that form small triangles"
    for line in stitch_segments(polylines):
        while line.shape[0] > 2:
            triplets = np.hstack((line[:-2,  None],
                                  line[1:-1, None],
                                  line[2:,   None]))
            A = np.dstack((triplets, np.ones((triplets.shape[0], 3, 1))))
            area = np.abs(0.5*np.linalg.det(A))
            smallest = np.argmin(area)
            if area[smallest] > min_area:
                break
            line = np.vstack((line[0:smallest+1], line[smallest+2:]))
        yield line


@compose(lambda it: np.array(list(it), dtype=int).reshape((-1, 3)))
def flatten_segments(polylines: PolyLineSet) -> TokenTriplets:
    "Flatten list of segments into a tensor of (triplets, 3)."
    for line in polylines:
        for i, vertex in enumerate(line):
            yield line_start_token if i == 0 else next_vertex_token
            yield from vertex
    yield from [sequence_end_token]*3


def unflatten_tokens(tokens: TokenTriplets) -> PolyLineSet:
    "Turn flattened tensor (triplets, 3) into list of segments."
    polylines = []
    for code, *qvert in tokens:
        if code == line_start_token:
            polylines.append([])
        elif code == next_vertex_token:
            pass
        elif code == sequence_end_token:
            assert qvert == [sequence_end_token, sequence_end_token]
            break
        else:
            raise ValueError(f'invalid opcode: {code}')
        polylines[-1].append(qvert)

    return [np.array(line) for line in polylines]


@stats.measure_func
@compose(list)
def subdivide_segments(polylines: PolyLineSet, *, max_length: float,
                       eps=np.finfo(float).eps) -> PolyLineSet:
    for a, b in split_segments(polylines):
        d = np.linalg.norm(b - a)
        # When n = 2, linspace returns [a,          b].
        # When n = 3, linspace returns [a, (a+b)/2, b].
        n = max(2, int(d / max_length - eps) + 2)
        yield np.linspace(a, b, n, dtype=a.dtype)

segs = np.array([[[0.0, 0.0], [0.0, 9.0]]])
assert subdivide_segments(segs, max_length=20.0)[0].tolist() == segs[0].tolist()
assert subdivide_segments(segs, max_length=10.0)[0].tolist() == segs[0].tolist()
assert subdivide_segments(segs, max_length=9.0)[0].tolist() == segs[0].tolist()
assert subdivide_segments(segs, max_length=5.0)[0][:, 1].tolist() == [0.0, 4.5, 9.0]
assert subdivide_segments(segs, max_length=8.0)[0][:, 1].tolist() == [0.0, 4.5, 9.0]
del segs


def intersect(path: MPath, polygon: MPath, subdivision) -> MPath:
    segs = subdivide_segments(path_to_polylines(path), max_length=subdivision)
    segs = split_segments(segs)
    segs_inside = [line for line in segs if np.any(polygon.contains_points(line))]
    return polylines_to_path(remove_segments(segs_inside))


def _valid_positions_gen(floorplan: floorplans.FloorPlan, n=100) -> Pos:
    path = floorplan.to_path()
    extents = path.get_extents()
    paths = [space.to_path() for space in floorplan.spaces]
    paths = [path for path in paths if path is not None]
    while True:
        P = np.random.uniform(extents.p0, extents.p1, size=(n, 2))
        contained = np.zeros(n, dtype=np.bool)
        for path in paths:
            contained |= path.contains_points(P)
        yield from P[contained]


@stats.measure_func
def sample_valid_positions(floorplan, n) -> List[Pos]:
    gen = _valid_positions_gen(floorplan, n=n)
    poses = islice(gen, n)
    return np.array(list(poses))


@stats.measure_func
def maximum_distance_subset(points, *, n=None, min_dist=np.inf):
    n = n if n else len(points)
    dmat = distance_matrix(points, points)
    selected = np.zeros(len(points), dtype=bool)
    selected[0] = True
    while np.count_nonzero(selected) < n:
        dmin_subset = dmat[selected, :].min(axis=0)
        ind = np.argmax(dmin_subset)
        if dmin_subset[ind] < min_dist:
            break
        assert not selected[ind]
        selected[ind] = True
    return selected


@njit('types.Tuple([float64[:, :], float64[:]])(float64[:], float64[:], float64[:, :], b1)')
def project_points_line(x0, x1, P, clip=True):
    "Project points P onto line x0-x1"
    v = x1 - x0
    x1norm = np.sqrt(v[0]**2 + v[1]**2)
    assert x1norm > 0.0
    x1unit = v / x1norm
    prods = np.dot(P - x0, x1unit)
    P_ = np.expand_dims(prods, -1) * np.expand_dims(x1unit, 0)
    T_ = prods/x1norm
    if clip:
        T_[T_ < 0.0] = 0.0
        T_[1.0 < T_] = 1.0
        #P_ = T_[:, None] * v[None, :]
        P_ = np.expand_dims(T_, -1) * np.expand_dims(v, 0)
    return P_ + x0, T_


from numba.extending import overload, register_jitable  # noqa: E402

@register_jitable
def _isclose_impl(a, b):
    return np.abs(a - b) < 1e-8

@overload(np.isclose, inline='always')
def np_isclose(a, b):
    return _isclose_impl

@overload(np.allclose, inline='always')
def np_allclose(a, b):
    return lambda a, b: np.all(_isclose_impl(a, b))


@njit('uint32(float64[:, :, ::1])')
def _expand_overlapping(segs):
    """Deduplicate line segments by expanding them to their
    largest overlapping segment endpoints. For example, if three
    segments are situated like this (but on top of each other)
        --
         ---
           ----
    we generate three segments
        ----
        -------
         ------
    If this routine is applied again, the three segments will be
    identical and collapsed into one segment.
    """
    num_updated = 0
    segs_ = np.empty_like(segs)

    # Shorthands. Segments span a-b, where a = (x1, y1) and b = (x2, y2).
    A_, B_  = segs_[:, 0, :], segs_[:, 1, :]
    X1_, Y1_ = A_[:, 0], A_[:, 1]
    X2_, Y2_ = B_[:, 0], B_[:, 1]

    for i in range(segs.shape[0]):

        # Translate so current segment starts at a = (0, 0) and ends at b = (x, y).
        t_i      = segs[i, 0, :]
        x_i, y_i = segs[i, 1] - t_i

        norm = np.sqrt(x_i**2 + y_i**2)
        #if norm < 1e-9:
        #    continue
        assert norm > 1e-12

        # Rotate all segments so current segment ends at b' = (x', 0) where x' = |x, y|.
        R_i = np.array([[x_i, -y_i],
                        [y_i,  x_i]]) / norm

        for j in range(segs_.shape[0]):
            segs_[j] = (segs[j] - t_i) @ R_i
            #np.dot(segs_t[j], R_i, out=segs_[j])
            if X2_[j] < X1_[j]:
                A_j_tmp = A_[j].copy()
                A_[j] = B_[j]
                B_[j] = A_j_tmp

        assert np.allclose(A_[i], 0.0)
        assert np.isclose(X2_[i], norm)
        assert np.isclose(Y2_[i], 0.0)

        # Flip segments (a,b) to (b,a) s.t. x1 <= x2.
        #flip = X2_ < X1_
        #A_[flip], B_[flip] = B_[flip].copy(), A_[flip].copy()
        #assert np.all(X1_ <= X2_)

        # Segment is collinear if it's on the X axis
        collinear = np.isclose(Y1_, 0.0) & np.isclose(Y2_, 0.0)

        # Check if other segment's X coordinates are inside this line
        overlap_x1 = (0.0 <= X1_) & (X1_ <= X2_[i])
        overlap_x2 = (0.0 <= X2_) & (X2_ <= X2_[i])

        selected = collinear & (overlap_x1 | overlap_x2)

        x1_min = np.min(X1_[selected])
        x2_max = np.max(X2_[selected])

        # Invert and put back into original array. Overwrite all overlapping
        # line segments with the same maximum line segment, which is exactly
        # collinear with segs[i].
        update = selected & ~(np.isclose(X1_, x1_min) & np.isclose(X2_, x2_max))
        new_seg = np.array([[x1_min, 0.0],
                            [x2_max, 0.0]])
        segs[update] = (new_seg @ R_i.T) + t_i
        num_updated += np.count_nonzero(update)

    return num_updated


@njit('void(float64[:, :, ::1])')
def _expand_overlapping_completely(segs):
    n = len(segs)
    # Routine needs to run N^2 times at worst.
    for i in range(n**2):
        num_updated = _expand_overlapping(segs)
        if num_updated == 0:
            return
    raise RuntimeError('logically impossible')


def filter_null_segments(polylines):
    segs = split_segments(polylines)
    return segs[~np.all(np.isclose(segs[:, 0], segs[:, 1]), axis=1)]


@stats.measure_func
def dedupe_polylines(polylines, *, dtype=None):
    "Deduplicate segments by expanding and removing duplicate segments."
    dtype = type(polylines[0][0][0]) if dtype is None else dtype
    segs = split_segments(polylines).astype(np.float64)
    segs = segs[~np.all(np.isclose(segs[:, 0], segs[:, 1]), axis=1)]
    _expand_overlapping_completely(segs)
    segs = np.asarray(segs, dtype=dtype)
    segs = np.unique(segs, axis=0)
    return segs


@stats.measure_func
def split_intersections(segs, *, dtype=None):
    "Split each segment whereever another segment crosses"
    dtype = segs.dtype if dtype is None else dtype
    segs = split_segments(segs).astype(np.float64)
    out = _split_intersection_impl(segs)
    out = np.array(out, dtype=dtype)
    # Remove zero-length segments, happens because of integer truncation.
    out = out[~np.all(np.isclose(out[:, 0], out[:, 1]), axis=1)]
    return out


@njit('float64[:, :, :](float64[:, :, :])')
def _split_intersection_impl(segs):

    out = []
    segs_ = np.empty_like(segs)

    # Iterate over each line, split at its intersections.
    for i in range(segs.shape[0]):

        # Translate all segments s.t. current segment a = (0, 0) and ends at b = (x, y).
        a_i      = segs[i, 0, :]
        x_i, y_i = segs[i, 1, :] - a_i

        norm = np.sqrt(x_i**2 + y_i**2)
        if norm < 1e-9:
            continue

        # Rotate all segments s.t. current segment ends at b' = (x', 0) where x' = |x, y|.
        R = np.array([[x_i, -y_i],
                      [y_i,  x_i]])/norm

        for j in range(segs_.shape[0]):
            segs_[j] = (segs[j] - a_i) @ R

        # Shorthands. Segments span a-b, where a = (x1, y1) and b = (x2, y2).
        A_, B_  = segs_[:, 0, :], segs_[:, 1, :]
        X1_, Y1_ = A_[:, 0], A_[:, 1]
        X2_, Y2_ = B_[:, 0], B_[:, 1]

        assert np.allclose(A_[i], 0.0)
        assert np.isclose(Y2_[i], 0.0)

        # Flip segments (a,b) to (b,a) s.t. x1 <= x2.
        flip = X2_ < X1_
        A_[flip], B_[flip] = B_[flip].copy(), A_[flip].copy()
        assert np.all(X1_ <= X2_)

        # Compute X intercept for all segments, i.e., where y = 0. We define the
        # segments parametrically by (x, y) = (1 - t)a + tb where 0 <= t <= 1.
        # Now we can deduce the X intercept:
        #
        #  0.      y = 0                 premise
        #  1.      x = (1-t)x1 + tx2     definition
        #  2.      y = (1-t)y1 + ty2     definition
        #  3.      y = y1 - ty1 + ty2    expand parentheses
        #  4.      y = y1 + ty2 - ty1    switch last two terms
        #  5.      y = y1 + t(y2 - y1)   factor t
        #  6.      0 = y1 + t(y2 - y1)   subst. 0 into 5
        #  7.    -y1 = t(y2 - y1)        subtract y1
        #  8.      t = -y1/(y2 - y1)     divide, switch RHS/LHS
        #  9.      t = y1/(y1 - y2)      factor negation into denom
        # 10.      x = x1 + t(x2 - x1)   analogous to 2-5
        # 11.      x = x1 + (y1/(y1 - y2))(x2 - x1)
        #                                subst. 9 into 10
        # If y1 = y2, the X intercept is undefined.
        #with np.errstate(divide='ignore', invalid='ignore'):
        T_ = Y1_/(Y1_ - Y2_)
        X_int = X1_ + T_*(X2_ - X1_)

        # We want to find segmenst whose X intercept lies in the
        # i'th segment's _interior_. If it is on the boundary, no
        # splitting is necesasry. Note that this filters out NaNs.
        overlap = (0.0 < X_int) & (X_int < X2_[i])

        # The X intercept is must lie inside the segment, i.e., 0 <= t <= 1.
        inside  = (0.0 <= T_) & (T_ <= 1.0)

        # X_out is a sequence of X coordinates where we want to cut.
        X_out = np.concatenate((X1_[i:i+1], X_int[overlap & inside], X2_[i:i+1]))

        # Remove multiple cuts at the same X coordinate.
        X_out = np.unique(X_out)

        # Create segments from pairs of X coordinates, let their y = 0.
        segs_out = np.empty((len(X_out) - 1, 2, 2))
        segs_out[:, :, 1] = 0.0
        for j in range(segs_out.shape[0]):
            segs_out[j, 0, 0] = X_out[j]
            segs_out[j, 1, 0] = X_out[j+1]

        # X coordinates must be strictly increasing.
        assert np.all(segs_out[:, 0, 0] < segs_out[:, 1, 0])

        # Invert rotation and translation.
        for j in range(segs_out.shape[0]):
            out.append(segs_out[j] @ R.T + a_i)

    out_arr = np.empty((len(out), 2, 2))
    for j in range(out_arr.shape[0]):
        out_arr[j] = out[j]
    return out_arr


@njit('float64[:, :, :](float64[:, :, :], float64[:, :])')
def project_lines_points(lines, points):
    n_points, n_dims = points.shape
    n_lines = lines.shape[0]
    assert lines.shape == (n_lines, 2, n_dims)
    out = np.empty((n_lines, n_points, n_dims))
    for i in range(n_lines):
        a, b = lines[i]
        out[i], _ = project_points_line(a, b, points, clip=True)
    return out


@stats.measure_func
def sorted_segments(segs: PolyLineSet, p: Pos):
    "Sort segments in order of increasing distance from the given point"
    segsf  = np.array(segs, dtype=np.float64)
    point  = np.array(p, dtype=np.float64)
    points = project_lines_points(segsf, point[None])[:, 0, :]
    dists  = np.linalg.norm(points - p, axis=1)
    inds   = np.argsort(dists)
    return segs[inds], points[inds], dists[inds]


@stats.measure_func
def sample_floorplan(floorplan, min_point_dist, max_num_points, trunc_dist,
                     max_seg_len, min_num_segs, max_num_segs, max_seg_dist,
                     min_seg_dist):
    "Sample maximally distant points in *floorplan* and sort segments from each point."

    path = floorplan_to_path(floorplan, recenter=False)
    segs = polylines_to_cleaned_segments(path_to_polylines(path),
                                         trunc_dist=trunc_dist,
                                         max_seg_len=max_seg_len)

    points = sample_valid_positions(floorplan, max_num_points)
    selected = maximum_distance_subset(points, min_dist=min_point_dist)
    selected_inds, = np.nonzero(selected)

    if np.all(selected):
        warnings.warn('all sampled points selected, either max_num_points too '
                      'low or min_point_dist too short')

    samples = []
    for i, pt_i in zip(selected_inds, points[selected]):
        segs_i, _, dists_i = sorted_segments(segs, pt_i)

        if dists_i[0] < min_seg_dist:
            continue

        cut_i    =  segs_i[:max_num_segs]
        dists_i  = dists_i[:max_num_segs]
        cut_i    =   cut_i[dists_i < max_seg_dist]

        if len(cut_i) < min_num_segs:
            continue

        samples.append((pt_i, cut_i, segs_i))

    if not samples:
        warnings.warn('no sampled points selected, either max_num_points too '
                      'low or min_point_dist too short')

    return path, samples


@stats.measure_func
def polylines_to_cleaned_segments(polylines, *, trunc_dist, max_seg_len):
    segs = split_segments(polylines)
    segs = np.round(segs/trunc_dist)*trunc_dist
    segs = dedupe_polylines(segs)
    segs = split_intersections(segs)
    segs = subdivide_segments(segs, max_length=max_seg_len)
    segs = split_segments(segs)
    return segs


class ConversionError(Exception):
    pass

def triplets_to_path(triplets, *, dequantize=True, fix_major_axis=False) -> MPath:
    if len(triplets) == 0:
        raise ConversionError('empty triplet array')

    opcodes  = triplets[:, 0]
    vertices = triplets[:, 1:]

    if not np.all(np.isin(opcodes, list(code_tokens))):
        raise ConversionError('some opcodes are invalid')

    vertices[opcodes == sequence_end_token, :] = 0

    try:
        segs = unflatten_tokens(triplets)
        qpath = polylines_to_path(segs)
        path = dequantize_path(qpath) if dequantize else qpath
        if fix_major_axis:
            bbox = path.get_extents()
            if bbox.width < bbox.height:
                # Both a 90 degree rotation and a reflection, this preserves
                # positivity in quantized representations.
                path.vertices = path.vertices @ [[0, 1], [1, 0]]
    except Exception as e:
        raise ConversionError(e)
    return path


@njit(['void(float64[:, :, :], int32, int32, int32, int32, float64)',
       'void(float64[:, :, :], int32, int32, int32, int32, float64)'])
def _draw_line(image, r0, c0, r1, c1, color):
    h, w, _ = image.shape
    dx =  abs(c1 - c0)
    sx = +1 if c0 < c1 else -1
    dy = -abs(r1 - r0)
    sy = +1 if r0 < r1 else -1
    err = dx + dy  # error value e_xy
    while True:
        if (0 <= r0 < h) and (0 <= c0 < w):
            image[r0, c0] = color
        if c0 == c1 and r0 == r1:
            break
        e2 = 2*err
        if e2 >= dy:  # e_xy+e_x > 0
            err += dy
            c0 += sx
        if e2 <= dx:   # e_xy+e_y < 0
            err += dx
            r0 += sy


@njit(['float64[:, :, :](float64[:, :, :], Tuple((int32, int32, int32)))',
       'float64[:, :, :](float32[:, :, :], Tuple((int32, int32, int32)))'])
def rasterize_segs(segs, shape):
    image = np.ones(shape, dtype=np.float64)
    h, w, c = shape
    xmax = w-1
    ymax = h-1
    for (x0, y0), (x1, y1) in segs:
        c0, r0 = int(xmax*(x0 + 0.5) + 0.5), int(ymax*(1 - (y0 + 0.5)) + 0.5)
        c1, r1 = int(xmax*(x1 + 0.5) + 0.5), int(ymax*(1 - (y1 + 0.5)) + 0.5)
        _draw_line(image, r0, c0, r1, c1, 0.0)
    return image


def figure_to_ndarray(fig=None):
    "Turn a matplotlib figure into a 4-channel ndarray"
    if fig is None:
        fig = plt.gcf()
    fig.canvas.draw()
    buf, (w, h) = fig.canvas.print_to_buffer()
    im = np.frombuffer(buf, dtype=np.uint8)
    im = im.reshape((h, w, 4))
    return im


def rasterize_path_fig(path: Optional[MPath]) -> np.ndarray:
    "Draw a path and return an RGBA rasterization"
    fig = plt.gcf()
    fig.clear()
    ax = fig.gca()
    if path is not None:
        plot_path(path, fig=fig, ax=ax)
    else:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, va='center', ha='center')
    ax.axis('equal')
    return figure_to_ndarray(fig)
