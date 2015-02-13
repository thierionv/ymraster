#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import os.path
sys.path.append(os.path.join(os.path.expanduser('~'), 'ProjetsDev', 'ymraster'))
from ymraster import Raster

IMG = '../tests/data/l8_20130714.tif'


def main():
    raster = Raster(IMG)
    raster.ndvi(4, 5, '/tmp/l8_20130714_ndvi.tif')
    raster.ndmi(5, 6, '/tmp/l8_20130714_ndwi.tif')
    raster.ndsi(3, 6, '/tmp/l8_20130714_mndwi.tif',)


if __name__ == '__main__':
    main()
