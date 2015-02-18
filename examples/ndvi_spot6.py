#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import sys
import os.path
sys.path.append(os.path.join(os.path.expanduser('~'), 'ProjetsDev', 'ymraster'))
from ymraster import Raster

IMG = '../tests/data/Spot6_MS_31072013.tif'


def main():
    raster = Raster(IMG)
    raster.ndvi(3, 4, '/tmp/spot6_ndvi.tif')


if __name__ == '__main__':
    main()
