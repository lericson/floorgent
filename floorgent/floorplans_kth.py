import sys
import numpy as np

try:
    import lxml.etree.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import logging
from typing import List
from pathlib import Path

from .floorplans import Line
from .floorplans import Space
from .floorplans import show_all
from .floorplans import FloorPlan


log = logging.getLogger(__name__)


def load_xml(filepath, *, rescale_si=True) -> FloorPlan:
    tree = ET.parse(filepath)
    root = tree.getroot()

    domain = root.find('Domain[@name]').get('name')
    scale_attrs = root.find('Scale').attrib

    fp = FloorPlan(domain=domain,
                   scale_pixels=float(scale_attrs['PixelDistance']),
                   scale_meters=float(scale_attrs['RealDistance']),
                   floor_name=root.get('FloorName'),
                   building_name=root.get('BuildingName'),
                   filepath=filepath)

    scale = fp.scale if rescale_si else 1.0

    for space_el in root.findall('space'):

        attrs = space_el.find('contour/centroid').attrib
        centroid = np.r_[float(attrs['x']), float(attrs['y'])] / scale

        space = Space(name=space_el.get('name'),
                      kind=space_el.get('type') or None,
                      centroid=centroid)

        for ls_el in space_el.findall('contour/linesegment'):
            attrs = ls_el.attrib
            start = np.r_[float(attrs['x1']), float(attrs['y1'])] / scale
            end   = np.r_[float(attrs['x2']), float(attrs['y2'])] / scale
            line  = Line(start, end,
                         kind=attrs['type'] or None,
                         portal=attrs['target'] or None)
            space.lines.append(line)

        for portal_el in space_el.findall('portal[@target]'):
            space.portals.append(portal_el.get('target'))

        fp.spaces.append(space)
        fp.space_map[space.name] = space

    return fp


def check_for_nul_portals(fp: FloorPlan):
    # Some floorplans have spaces without a contour. Need to remove them, but
    # for some reason, they can have portals, so we make sure that portals lead
    # to non-contoured spaces too.
    for space in fp.spaces:
        if space.lines:
            continue
        for name in space.portals:
            if fp.space_map[name].lines:
                log.error(f'empty space {space.name} has portal to non-empty space {name}')

    if not fp.spaces or not any(space.lines for space in fp.spaces):
        raise ValueError('empty floor plan')


def load_all(filepaths: List[Path], *, reject=True) -> List[FloorPlan]:
    results = []
    for filepath in filepaths:
        log.debug(f'loading {filepath}')
        try:
            results.append(load_xml(filepath))
        except Exception:
            if not reject:
                raise
            log.exception(f'did not load {filepath}')
    return results


def main(args=sys.argv[1:]):
    import logcolor
    logcolor.basic_config(level=logging.INFO)

    filepaths = [Path(p) for p in sys.argv[1:]]

    if not filepaths:
        log.error('please input some XML filepaths')
        return

    if len(filepaths) == 1 and filepaths[0].is_dir():
        log.info('argument is a directory, searching for all XML files within')
        filepaths = list(filepaths[0].glob('**/*.xml'))
        log.info(f'found {len(filepaths)} xml files')

    floorplans = load_all(filepaths)

    show_all(floorplans)


if __name__ == "__main__":
    main()
