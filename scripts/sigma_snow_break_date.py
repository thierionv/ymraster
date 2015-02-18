#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-

from ymraster import Raster
import numpy as np

import os.path
from datetime import datetime
from time import mktime
import argparse

_DATE_THRESHOLD_MAP = {
    (datetime(2013, 01, 01), datetime(2013, 02, 01)): 0.35,
    (datetime(2013, 02, 01), datetime(2013, 05, 01)): 0.4,
    (datetime(2013, 05, 01), datetime(2013, 12, 31)): 0.35,
}


def _under_threshold_date(stack_array, dates, out_array):
    ysize, xsize = out_array.shape
    for i in range(ysize):
        tranche = stack_array[i, ...]
        for j in range(xsize):
            line = tranche[j, ...]

            snow_brake_date = np.nan
            k = len(dates) - 1
            done = False
            while k >= 0 and not done:
                date = dates[k]
                for bound_dates, threshold in _DATE_THRESHOLD_MAP.iteritems():
                    date_min, date_max = bound_dates
                    if date_min <= date and date < date_max \
                            and line[k] > threshold:
                        out_array[i, j] = mktime(snow_brake_date.timetuple()) \
                            if isinstance(snow_brake_date, datetime) \
                            else snow_brake_date
                        print("{}, {} <- {} ({})".format(i, j, snow_brake_date, line[k]))
                        done = True
                snow_brake_date = date
                k -= 1


def snow_brake_date():
    # Command-line parameters
    parser = argparse.ArgumentParser(
        description="Compute a raster whose value for each pixel is the "
        "date/time of snow break, given a list of temporally distinct, but "
        "spatially identical, images.\n\n"
        "That is, the value of each pixel will be the date/time of the first "
        "image where the snow disappear on the pixel and does not appear again "
        "in the following images.\n\n"
        "Presence or absence of snow is evaluated from the Normalized "
        "Difference Snow Index (NDSI) which is computed from green and "
        "mid-infrared band."
        "Dates will be in numeric format (eg. Apr 25, 2013 (midnight) "
        "-> 1366840800.0)")
    parser.add_argument("rasters", nargs='+',
                        help="List of input images")
    parser.add_argument("-g", "--idx_green", type=int, default=3,
                        help="Index of a green band, common to all images "
                        "(default: 3)")
    parser.add_argument("-mir", "--idx_mir", type=int, default=6,
                        help="Index of a mid-infrared band, common to all "
                        "images (default: 6)")
    parser.add_argument("-o", "--out_file", default='./snow_break.tif',
                        help="Path to the output file (default: "
                        "'snow_break.tif' in the current folder)")
    args = parser.parse_args()

    # Perform the actual work

    # Compute all NDSIs, read them as NumPy arrays, and stack them into one
    # array
    rasters = [Raster(filename) for filename in args.rasters]
    rasters.sort(key=lambda raster: raster.meta['datetime'])
    for raster in rasters:
        assert raster.meta['datetime'] is not None, "Raster has no "
        "TIFFTAG_DATETIME metatadata: {}".format(raster.filename)
    ndsis = [raster.mndwi('{}.mndwi.tif'.format(
        os.path.basename(os.path.splitext(raster.filename)[0])),
        idx_green=args.idx_green,
        idx_mir=args.idx_mir)
        for raster in rasters]
    arrays = [ndsi.array() for ndsi in ndsis]
    stack_array = np.dstack(arrays)

    # List of dates
    dates = [ndsi.meta['datetime'] for ndsi in ndsis]

    # Create an empty array of correct size
    ndsi0 = ndsis[0]
    width = ndsi0.meta['width']
    height = ndsi0.meta['height']
    out_array = np.empty((height, width), dtype=np.float64)

    # Find the snow-brake date
    _under_threshold_date(stack_array, dates, out_array)


if __name__ == '__main__':
    snow_brake_date()
