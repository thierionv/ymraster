#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import os.path
sys.path.append(os.path.join(os.path.expanduser('~'), 'ProjetsDev', 'ymraster'))
from ymraster import Raster

IMG = '../tests/data/l8_20130714.tif'


def main():
    raster = Raster(IMG)
    raster.ndvi('/tmp/l8_20130714_ndvi.tif', 4, 5)
    raster.ndmi('/tmp/l8_20130714_ndwi.tif', 5, 6)
    raster.ndsi('/tmp/l8_20130714_mndwi.tif', 3, 6)


if __name__ == '__main__':
    main()
