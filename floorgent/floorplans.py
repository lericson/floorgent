"""Ⓕ ⓛ ⓞ ⓞ ⓡ ⓟ ⓛ ⓐ ⓝ ⓢ

Note that all units are stored in PIXEL units, with the origin at the top left
corner. This library rescales to SI units by default, because that seems more
logical.
"""


import logging
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from pathlib import Path
from dataclasses import field
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path as MPath


log = logging.getLogger(__name__)


Point = np.ndarray
BoundingBox = Tuple[Point, Point]


@dataclass
class Line():
    "A line segment of a space in a floor plan"
    start: Point
    end: Point
    kind: str
    portal: str


@dataclass
class Space():
    "A space within a floorplan"
    name: str
    kind: str
    centroid: Point
    lines: List[Line] = field(default_factory=list, repr=False)
    portals: List[str] = field(default_factory=list)

    def to_path(self) -> Optional[MPath]:
        # Polygons with holes in them are represented by a second polyline but
        # in the other direction. So if you have a circle in clockwise fashion,
        # you can carve out an inner circle by a polyline going counter
        # clockwise. However, this isn't always the case. We therefore must
        # detect the orientation of each polyline.

        # Begin by generating all polyline sets. *polys* is a list of vertex
        # lists of the polygons generated so far, including the current
        # possibly unfinished polygon, polys[-1], with its current position,
        # polys[-1][-1].
        polys = []
        for line in self.lines:
            if not (polys and np.allclose(polys[-1][-1], line.start)):
                polys.append([line.start])
            polys[-1].append(line.end)

        if not polys:
            return None

        # Find orientations by computing the signed area of each polygon. A
        # negative area is the result of a polygon wound counter clockwise.
        # Once the orientation is recovered, make sure that it's CW if it's the
        # first (outside) polygon, and CCW if anything else. This assumes that
        # there are no polygons inside polygons inside polygons.
        for i, poly in enumerate(polys):
            r = np.hstack((poly[:-1], poly[1:]))
            x1, y1, x2, y2 = r.T
            signed_area_2x = np.sum((x2 - x1)*(y2 - y1))
            orientation = np.sign(signed_area_2x)
            if orientation != (-1)**(i == 0):
                # Flip orientation.
                poly[:] = poly[::-1]

        # Now generate the path
        verts = []
        codes = []
        for poly in polys:
            verts.extend(poly)
            codes_poly = [MPath.LINETO]*len(poly)
            codes_poly[0] = MPath.MOVETO
            codes.extend(codes_poly)

        return MPath(verts, codes)

    def plot(self, *, ax, color='C0'):
        from matplotlib.patches import PathPatch

        path = self.to_path()
        if path is not None:
            ax.add_patch(PathPatch(path, facecolor=color, alpha=0.7))

        # Make lines for all portals
        verts = []
        codes = []
        for line in self.lines:
            if line.kind.casefold() != 'portal':
                continue
            verts.extend((line.start, line.end))
            codes.extend((MPath.MOVETO, MPath.LINETO))
        if verts:
            ax.add_patch(PathPatch(MPath(verts, codes), color=color, lw=3, alpha=0.9))

        #lc = mc.LineCollection([(line.start, line.end) for line in self.lines], colors=color)
        #ax.add_collection(lc)


@dataclass
class FloorPlan():
    "A floor plan"

    domain: str
    floor_name: str
    building_name: Optional[str]

    scale_pixels: float           = field(repr=False)
    scale_meters: float           = field(repr=False)
    scale:        float           = field(repr=False, init=False)

    spaces: List[Space]           = field(default_factory=list, repr=False)
    space_map: Dict[str, Space]   = field(repr=False, init=False)

    filepath: Optional[Path]      = None

    def __post_init__(self):
        self.scale = self.scale_pixels / self.scale_meters
        self.space_map = {space.name: space for space in self.spaces}

    @property
    def extents(self) -> BoundingBox:
        0/0

    @property
    def centroid(self) -> Point:
        0/0

    def plot(self, *, ax):
        #from matplotlib import pyplot as plt
        for i, space in enumerate(self.spaces):
            #if not np.allclose(space.lines[-1].end, space.lines[0].start):
            #    log.warning('space contour is not closed, ignored')
            space.plot(ax=ax, color=f'C{i}')

    def to_path(self) -> Optional[MPath]:
        paths = [s.to_path() for s in self.spaces]
        paths = [p for p in paths if p is not None]
        return MPath.make_compound_path(*paths)


def show_all(floorplans: List[FloorPlan]):
    pending = 0
    for i, fp in enumerate(floorplans):
        from matplotlib import pyplot as plt
        print('plotting', fp)
        fig=plt.figure(num=i+1)
        ax=fig.gca()
        ax.set_title(fp.filepath)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        fp.plot(ax=ax)
        ax.autoscale_view()
        fig.tight_layout()
        fig.show()
        pending += 1
        if pending >= 10:
            plt.show()
            pending = 0
            input(f'Showed {i+1}/{len(floorplans)} plans. Continue? [^D to stop] ')
    if pending:
        plt.show()
