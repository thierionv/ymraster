# -*- coding: utf-8 -*-

import argparse
from ymraster import Raster, temporal_stats

import os.path
from datetime import datetime


def valid_date(s):
    try:
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return datetime.strptime(s, '%Y-%m-%d')
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def compute_temporal_stats():
    # Command-line parameters
    parser = argparse.ArgumentParser(
        description="Compute pixel-wise statistics on radiometric indices from "
        "a given list of multitemporal, but spatially identical, images.\n\n"
        "Output is a multi-band image where each band contains a result (eg. "
        "maximum, mean). For results taken from original values (eg. maximum), "
        "there is an additional band which gives the date/time at which the "
        "result has been found in numeric format (eg. maximum has occured on "
        "Apr 25, 2013 (midnight) -> 1366840800.0)")
    parser.add_argument("rasters", nargs='+',
                        help="List of input images")
    parser.add_argument("-b", "--idx_blue", type=int, default=2,
                        help="Index of a blue band, common to all images "
                        "(default: 2)")
    parser.add_argument("-g", "--idx_green", type=int, default=3,
                        help="Index of a green band, common to all images "
                        "(default: 3)")
    parser.add_argument("-r", "--idx_red", type=int, default=4,
                        help="Index of a red band, common to all images "
                        "(default: 4)")
    parser.add_argument("-nir", "--idx_nir", type=int, default=5,
                        help="Index of a near-infrared band, common to all "
                        "images (default: 5)")
    parser.add_argument("-mir", "--idx_mir", type=int, default=6,
                        help="Index of a mid-infrared band, common to all "
                        "images (default: 6)")
    parser.add_argument("-d", "--date-from", type=valid_date,
                        help="Date from which to compute statis "
                        "the input images and whose values are dates in "
                        "numeric format. If specified, statistical dates are "
                        "computed relative to the dates in this raster")
    parser.add_argument("-t", "--time_raster",
                        help="Path to a mono-band raster with same extent than "
                        "the input images and whose values are dates in "
                        "numeric format. If specified, statistical dates are "
                        "computed relative to the dates in this raster")
    parser.add_argument("-o", "--out_file", default='./stats.tif',
                        help="Path to the output file (default: 'stats.tif' in "
                        "the current folder)")
    args = parser.parse_args()

    # Do the actual work
    rasters = [Raster(filename) for filename in args.rasters]
    ndvis = [raster.ndvi(
        idx_red=args.idx_red,
        idx_nir=args.idx_nir,
        '{}.ndvi.tif'.format(
            os.path.basename(os.path.splitext(raster.filename)[0])))
        for raster in rasters]
    ndwis = [raster.ndwi(
        idx_nir=args.idx_nir,
        idx_mir=args.idx_mir,
        '{}.ndwi.tif'.format(
            os.path.basename(os.path.splitext(raster.filename)[0])))
        for raster in rasters]
    mndwis = [raster.mndwi(
        idx_green=args.idx_green,
        idx_mir=args.idx_mir,
        '{}.mndwi.tif'.format(
            os.path.basename(os.path.splitext(raster.filename)[0])))
        for raster in rasters]
    temporal_stats(ndvis, 'stats.ndvi.tif', 'GTiff',
                   stats=['min', 'max', 'range'])
    temporal_stats(ndwis, 'stats.ndwi.tif', 'GTiff',
                   stats=['min', 'max', 'range'])
    temporal_stats(mndwis, 'stats.mndwi.tif', 'GTiff',
                   stats=['min', 'max', 'range'])


if __name__ == "__main__":
    compute_temporal_stats()
