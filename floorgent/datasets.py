"""Data loading by specification

For example

    dataset_spec = 'kth:/ds/kth_floorplans/**/*.xml'
    dataset_spec = 'zind:/ds/zind/data'

NOTE: This module deals with both filesystem paths and curves as paths, so we
call them path specifications (pathspec) and drawings instead.
"""

import matplotlib.patches as patches

import json
import logging
from glob import glob
from typing import List, Optional
from os.path import expanduser
from pathlib import Path as FSPath
from dataclasses import field
from dataclasses import dataclass

import numpy as np
import matplotlib as mpl
from skimage import io as im_io
from matplotlib.path import Path as MPath

from . import quant
from . import config
from . import floorplans_kth
from .utils import execute_seq
from .utils import execute_many
from .polylines import PathBuilder
from .polylines import rasterize_segs
from .polylines import sorted_segments
from .polylines import stitch_segments
from .polylines import flatten_segments
from .polylines import sample_floorplan
from .polylines import path_to_polylines
from .polylines import polylines_to_path
from .polylines import project_points_line
from .polylines import polylines_to_cleaned_segments
from .floorplans import FloorPlan


log = logging.getLogger(__name__)


def _qsegs_to_trips(qsegs):
    if config.line_soup:
        polylines = qsegs
    else:
        polylines = stitch_segments(qsegs)
    return flatten_segments(polylines)


@dataclass
class Example:
    position:  np.ndarray
    drawing:   MPath                = field(repr=False)
    triplets:  np.ndarray           = field()
    image:     Optional[np.ndarray] = field(default=None)
    floorplan: Optional[FloorPlan]  = field(default=None)


def _sample(floorplan):
    sampling_opts = {'min_point_dist': config.min_point_dist,
                     'max_num_points': config.max_num_points,
                     'trunc_dist':     config.truncation_dist,
                     'max_seg_len':    config.max_seg_len,
                     'min_num_segs':   config.min_num_segs,
                     'max_num_segs':   config.max_num_segs,
                     'max_seg_dist':   config.max_seg_dist,
                     'min_seg_dist':   config.min_seg_dist}
    return sample_floorplan(floorplan, **sampling_opts)


def _generate_examples_bev(drawing, point, segs, segs_full, center, scale):
    drawing = MPath((drawing.vertices - point - center) / scale, drawing.codes)
    segs_norm = (segs - center) / scale
    segs_quant = quant.quantize(segs_norm, check=False)
    triplets = _qsegs_to_trips(segs_quant)

    if config.conditioning == 'none':
        image = None
    elif config.conditioning == 'bev':
        image = rasterize_segs(segs_norm[:config.max_bev_segs],
                               shape=config.image_shape)
    else:
        raise ValueError('conditioning not possible for this dataset')

    ex   = Example(position=point, drawing=drawing,
                   image=image, triplets=triplets)
    ex.segs_norm  = segs_norm
    ex.segs_quant = segs_quant
    ex.segs_full = (segs_full - center) / scale

    return ex


class KTH:
    def __init__(self, pathspec: FSPath = '/ds/kth_floorplans/**/*.xml'):
        self.xmls = glob(expanduser(pathspec), recursive=True)

    def generate_jobs(self):

        floorplans = floorplans_kth.load_all(self.xmls)

        print('Subsampling floorplans')
        per_fp_samples = execute_many([(_sample, fp) for fp in floorplans])

        drawings   = []
        points     = []
        segs_list  = []
        full_list  = []
        for drawing_ij, samples_i in per_fp_samples:
            for point_ij, segs_ij, full_ij in samples_i:
                drawings.append(drawing_ij)
                points.append(point_ij)
                segs_list.append(segs_ij - point_ij)
                full_list.append(full_ij - point_ij)

        #from panotools.utils import print_arr
        #print_arr('verts', np.concatenate(segs_list, axis=0), axis=(0,1))

        print('Computing appropriate global scale factor')
        submaxes = [np.amax(np.abs(segs)) for segs in segs_list]
        scale    = 2*(1 + 1/config.quant_range)*np.amax(submaxes)
        center   = np.r_[0.0, 0.0]

        self.center     = center
        self.scale      = scale
        self.drawings   = drawings
        self.floorplans = floorplans

        jobs = [(_generate_examples_bev, *args, center, scale)
                for args in zip(drawings, points, segs_list, full_list)]

        return jobs

    def generate_examples(self):
        jobs = self.generate_jobs()
        print('Generating training examples')
        # Don't use execute_many(). Submitting 1000's of jobs takes longer than
        # just executing them sequentially.
        return execute_seq(jobs)

    @property
    def size(self):
        return f'{len(self.xmls)} floorplan XML files'


def _generate_image_example(drawing, image_file, position=np.r_[0.0, 0.0],
                            scale=1.0, center=np.r_[0.0, 0.0]):
    from .polylines import split_segments
    segs_full = split_segments(path_to_polylines(drawing))
    segs = polylines_to_cleaned_segments(segs_full,
                                         trunc_dist=config.truncation_dist,
                                         max_seg_len=config.max_seg_len)
    segs_norm = (segs - center) / scale
    segs_quant = quant.quantize(segs_norm, check=False)
    triplets = _qsegs_to_trips(segs_quant)
    point = quant.quantize((np.array(position) - center) / scale)
    qsegs_near, _, qdists = sorted_segments(segs_quant, point)

    image = im_io.imread(image_file)
    if image.dtype == np.uint8:
        image = (image / 255.0).astype(np.float32)
    assert image.dtype == np.float32
    step = config.image_stride
    image = image[::step, ::step, :]
    assert image.shape == config.image_shape

    ex = Example(position=point, drawing=drawing, image=image, triplets=triplets)
    ex.segs_norm  = segs_norm
    ex.segs_quant = segs_quant
    ex.segs_full = (segs_full - center) / scale
    return ex



class Matterport:
    """
    The path parameter should contain a folder structure like this:
    /ds/matterport3d$ tree -d -L 3 --filelimit 5
        .
        ├── layout_annotations
        │   └── floorplans
        ├── panoramas
        │   ├── all_skybox_images/<floorname>/matterport_skybox_images/
    """

    # Matterport panoramas are always at (0, 0)
    position = np.r_[0.0, 0.0]

    def __init__(self, label_dir):
        self.label_dir = FSPath(label_dir)
        self.json_files = list(self.label_dir.glob('*.json'))

    @property
    def size(self):
        return f'{len(self.json_files)} JSON files'

    dataset_dir  = property(lambda self: self.label_dir.parent)
    skyboxes_dir = property(lambda self: self.dataset_dir/'panoramas'/'all_skybox_images')

    def skybox_image_path_aligned(self, floor, section):
        return self.skyboxes_dir/floor/'matterport_skybox_images'/f'{section}_skybox_pano_align.png'

    def _iter_panos(self):
        for json_file in self.json_files:
            # zsNo4HB9uLZ_2e349b06dd94494ea4b887458e4ab4a3_label.json
            floor, section, _ = json_file.stem.split('_', 2)
            image_file = self.skybox_image_path_aligned(floor, section)

            if not image_file.exists():
                log.warn('could not load json document %r, '
                         'corresponding image not found: %s',
                         json_file, image_file)
                continue

            with open(json_file, 'r') as fp:
                json_doc = json.load(fp)

            drawing = self._json_doc_to_path(json_doc)

            yield floor, section, drawing, image_file

    @classmethod
    def _json_doc_to_path(cls, json_doc):
        #zc = json_doc['cameraHeight']
        #zh = json_doc['layoutHeight']
        points_xyz = np.array([p['xyz'] for p in json_doc['layoutPoints']['points']])
        points_xyz = points_xyz @ [[ 0, 1,  0],
                                   [ 0, 0, -1],
                                   [-1, 0,  0]]
        #points_xyz = points_xyz - (0, 0, zc)
        xyz_lines = []
        for wall in json_doc['layoutWalls']['walls']:
            xyz_line = points_xyz[wall['pointsIdx']]
            xyz_lines.append(xyz_line)
        return polylines_to_path([xyz_line[:, 0:2] for xyz_line in xyz_lines])

    def generate_examples(self):
        self.panoramas = list(self._iter_panos())
        vertices = np.concatenate([drawing.vertices for _, _, drawing, _ in self.panoramas])
        self.center = np.r_[0.0, 0.0]
        self.scale = 2*np.max(np.abs(vertices))
        jobs = [(_generate_image_example, drawing, image_file,
                 self.position, self.scale, self.center)
                for _, _, drawing, image_file in self.panoramas]
        return execute_many(jobs)


class Zillow:
    #  The walls of a building in Zillow are just a sequence of vertices
    # [[-2.3847386714338312, 3.4969748126146567],
    #  ...
    #  [-2.3847386714338312, 3.4969748126146567]]
    #
    # The doors are specified by start and end coordinate:
   #  [[-2.3826519882331745, 4.3444224795339785],
    #   [-2.3842327448669822, 3.702442651994133]]
    #
    def __init__(self, path=FSPath("/ds/zind/data")):
        self.path = path

    def generate_examples(self):
        all_floors = []
        for building_path in [e for e in self.path.iterdir() if e.is_dir()]:
            with open((building_path / "zind_data.json"), 'r') as fp:
                data = json.load(fp)

            floors = self.building_floorplans(data, building_path)
            all_floors.extend(floors)
            # panoramas = []
            # panos = building_path / "panos"
            # for floor_polyline, floorname in floors:
            #     for pano in panos.iterdir():
            #         if floorname in str(pano.stem):
            #             panoramas.append(pano)
            #     all_floors.append((floor_polyline, panoramas))

        # should use execute_many/execute_seq, but need to figure out how things fit together first.
        return all_floors

    @property
    def size(self):
        bs = sum([1 for e in self.path.iterdir() if e.is_dir()])
        return f'{bs} buildings in Zillow'

    def building_floorplans(self, data, building_path, coord_sys='image', ax=None,
                            floors=None, rooms=None, tol=5e-8) -> List[Example]:
        floorplan_examples = []

        for floor in data["redraw"]:
            if floors and floor not in floors:
                continue

            rotation_deg = data["floorplan_to_redraw_transformation"][floor]["rotation"]
            translation = data["floorplan_to_redraw_transformation"][floor]["translation"]
            if coord_sys == 'image':
                scale = 1/data["floorplan_to_redraw_transformation"][floor]["scale"]
            elif coord_sys == 'meters':
                scale = data["scale_meters_per_coordinate"][floor]
            else:
                raise ValueError('??')

            tra = mpl.transforms.Affine2D()
            tra = tra.scale(scale)
            tra = tra.rotate_deg(rotation_deg)
            tra = tra.translate(*tra.transform_point(-np.array(translation)))

            rooms_floor = rooms if rooms is not None else data["redraw"][floor].keys()
            #doors = np.array([door for room in rooms_floor for door in data["redraw"][floor][room]["doors"]])

            for room in rooms_floor:

                vertices = np.array(data["redraw"][floor][room]["vertices"])
                doors = np.array(data["redraw"][floor][room]["doors"])

                pins = data["redraw"][floor][room]["pins"]
                if len(pins) > 0:
                    position = data["redraw"][floor][room]["pins"]["position"]
                else:
                    position = np.array([0.0, 0.0])

                builder = PathBuilder.start_at(vertices[0])

                for next_vert in vertices[1:]:

                    doors_on_line = []

                    # Project each door onto the line segment between
                    # last_vertex and next_vert, so we can later sort them by
                    # order of appearance on the line segment.
                    for door in doors:
                        (pa, pb), (ta, tb) = project_points_line(
                            builder.last_vertex, next_vert, door, clip=False)

                        # Assert both door endpoint are inside the line segment.
                        if not (0.0 < ta < 1.0 and 0.0 < tb < 1.0):
                            continue

                        # Assert both door endpoints are sufficiently near the line.
                        da = np.linalg.norm(pa - door[0])
                        db = np.linalg.norm(pb - door[1])
                        if not (da*scale < tol and db*scale < tol):
                            continue
                        del da, db

                        # Assert that the door endpoints are in "line order".
                        if tb < ta:
                            pa, pb = pb, pa
                            ta, tb = tb, ta

                        doors_on_line.append((ta, tb, pa, pb))

                    doors_on_line.sort(key=lambda p: p[0])

                    for i, (ta, _, pa, pb) in enumerate(doors_on_line):

                        # If door starts before the previous door ended, skip.
                        if i > 0:
                            (_, tb_prev, _, _) = doors_on_line[i-1]
                            if ta < tb_prev:
                                continue

                        # Line is intersected by pa, pb as last_vertex to pa to
                        # pb to next_vert. Make a door by drawing to pa and
                        # jumping to pb.
                        builder.line_to(pa)
                        builder.move_to(pb)

                    builder.line_to(next_vert)

                path = builder.to_path()
                path = path.transformed(tra)

                #patch = patches.PathPatch(path, lw=1, facecolor='none')

                segs = polylines_to_cleaned_segments(path_to_polylines(path),
                                                    trunc_dist=config.truncation_dist,
                                                    max_seg_len=config.max_seg_len)
                qsegs = quant.quantize(segs / scale)
                point = quant.quantize(np.array(position) / scale)

                ex = Example(drawing=path, triples=triplets, position=point)
                floorplan_examples.append(example)

        return floorplan_examples

                #path = MPath([v for door in doors for v in door],
                #             [MPath.MOVETO, MPath.LINETO]*len(doors))
                #path.vertices += 0.001*(np.arange(len(path.vertices))//2)[:, None]
                #path = path.transformed(tra)
                #patch = patches.PathPatch(path, lw=1, edgecolor='C3', facecolor='none')
                #ax.add_patch(patch)

                #pts=np.array(vertices)*scale
                #ax.scatter(pts[:, 0], pts[:, 1])

            # plt.axis("equal")
            #plt.savefig(f"{'imgs/' + building_path.stem + floor}.png")
            #plt.close("all")


class Shrek:

    def __init__(self, pathspec=None):
        if pathspec is None:
            pathspec = FSPath.cwd() / 'shrek.jpg'
        assert pathspec.exists(), f'{pathspec} exists'
        self.image_file = pathspec
        self.drawing = polylines_to_path([[[-0.1, +0.2],
                                           [-0.1, -0.2],
                                           [+0.1, -0.2],
                                           [+0.1, -0.1],
                                           [ 0.0, -0.1],
                                           [ 0.0, +0.2],
                                           [-0.1, +0.2]]])

    size = 'the one and only Shrek'

    def generate_examples(self):
        self.center = np.r_[0.0, 0.0]
        self.scale = 1.0
        return 100*[_generate_image_example(self.drawing, self.image_file)]


dataset_loaders = {'kth': KTH,
                   'zind': Zillow,
                   'shrek': Shrek,
                   'matterport': Matterport}


def load(uri):
    kind, _, pathspec = uri.partition(':')
    if kind not in dataset_loaders:
        raise ValueError(f'unknown dataset kind {kind!r}')
    loader = dataset_loaders[kind]
    return loader(pathspec) if pathspec else loader()

if __name__ == '__main__':
    z = Zillow()
    z.generate_examples()
