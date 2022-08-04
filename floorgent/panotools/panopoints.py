import numpy as np


def pano_points(h, w):
    """Compute points with near-equal density in a panorama.

    A panorama is an equirectangular projection of a sphere onto a
    two-dimensional image grid, by first projecting the sphere onto a cylinder
    with equal radius, then cutting axially along the cylinder and "unrolling"
    it to be flat. The first step means that the points on the resulting grid
    have a non-uniform density. For example, the two poles of the sphere are
    projected to a circle at the top and bottom of the cylinder, while points
    at the equator are projected to single points in the image grid.

    This function computes a set of pixels for each meridian, so that the arc
    length between pixels is approximately constant. This is a "fairer"
    distribution.
    """

    # Compute latitude of each row of pixels
    Lat = np.linspace(-np.pi/2, np.pi/2, h)

    # Compute number of pixel desired for each latitude. The ith meridian's
    # radius is cos(lat[i]), and we want w pixels at the equator.
    n_pix_lat = np.ceil(w * np.cos(Lat)).astype(np.uint32)

    # Desired columns j for row i
    def cols_for_row(i):
        return (np.linspace(0, w, n_pix_lat[i], endpoint=False) + w/2) % w

    points = [(i, j) for i in range(h) for j in cols_for_row(i)]

    return np.array(points)


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from utils import print_arr

    pano = plt.imread('pano2.png')
    #pano = pano[::8, ::8, :]
    h, w, c = pano.shape
    print_arr('pano', pano, axis=(0, 1))

    points = pano_points(h, w)
    print_arr('points', points)
    assert np.all(0 <= points)


    def interp_bilinear(arr, points):
        points_floor_float = np.floor(points)
        points_floor = points_floor_float.astype(np.int32)
        points_ceil = points_floor + 1
        points_floor %= arr.shape[:points.ndim]
        points_ceil  %= arr.shape[:points.ndim]

        blend = points - points_floor_float

        top_left  = arr[points_floor[..., 0], points_floor[..., 1]]
        top_right = arr[points_floor[..., 0],  points_ceil[..., 1]]
        bot_left  = arr[ points_ceil[..., 0], points_floor[..., 1]]
        bot_right = arr[ points_ceil[..., 0],  points_ceil[..., 1]]

        interp_top = blend[..., 1][..., None]*(top_right - top_left) + top_left
        interp_bot = blend[..., 1][..., None]*(bot_right - bot_left) + bot_left
        interp     = blend[..., 0][..., None]*(interp_bot - interp_top) + interp_top

        return interp

    rows, cols = points[:, 0], points[:, 1]
    rows_int, cols_int = np.round((rows, cols)).astype(np.uint32)

    print()
    # Column-wise subpixel interpolation. Only useful for making some kind of
    # density heatmap.
    pano2 = np.zeros((h, w))
    left_col  = np.floor(cols).astype(np.uint32)
    right_col = (left_col + 1) % w
    subpixel = cols - left_col
    left_val = 1.0 - subpixel
    right_val =      subpixel
    #pano2[rows_int, cols_int] = (right_val[:, None]*pano[rows_int, right_col] + left_val[:, None]*pano[rows_int,  left_col])
    pano2[rows_int, right_col]  = right_val
    pano2[rows_int,  left_col] +=  left_val

    pano3 = np.zeros((h, w, c))
    pano3[rows_int, cols_int] = np.ones_like(pano)[rows_int, cols_int]
    pano4 = np.zeros((h, w, c))
    pano4[rows_int, cols_int] = pano[rows_int, cols_int]

    num_input = h*w
    num_output = points.shape[0]
    print(h*w, 'input pixels')
    print(np.count_nonzero(np.max(pano3, axis=2)), 'output pixels')

    #plt.figure()
    #plt.hist2d(points[:, 1], points[:, 0], bins=(w, h), range=[[0, w], [0, h]])
    ##plt.colorbar()
    #plt.axis('equal')

    plt.figure()
    plt.imshow(pano)
    plt.tight_layout()

    plt.figure()
    plt.imshow(pano2)
    plt.tight_layout()

    plt.figure()
    plt.imshow(pano3)
    plt.tight_layout()

    plt.figure()
    plt.imshow(pano4)
    plt.tight_layout()

    plt.show()
