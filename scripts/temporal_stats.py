# -*- coding: utf-8 -*-

import argparse
from ymraster import Raster
import numpy as np
from osgeo import gdal

import os.path


def temporal_stats():
    # Command-line parameters
    parser = argparse.ArgumentParser(
        description="Compute statistics from a given list of temporally "
        "distinct (but spatially identical) images. Results are saved to the "
        "specified output file.")
    parser.add_argument("rasters", nargs='+',
                        help="List of input images")
    parser.add_argument("-b", "--idx_blue", type=int, default=1,
                        help="Index of a blue band common to all images "
                        "(default: 1)")
    parser.add_argument("-g", "--idx_green", type=int, default=2,
                        help="Index of a green band common to all images "
                        "(default: 2)")
    parser.add_argument("-r", "--idx_red", type=int, default=3,
                        help="Index of a red band common to all images "
                        "(default: 3)")
    parser.add_argument("-nir", "--idx_nir", type=int, default=4,
                        help="Index of a near-infrared band common to all "
                        "images (default: 4)")
    parser.add_argument("-mir", "--idx_mir", type=int, default=5,
                        help="Index of a mid-infrared band common to all "
                        "images (default: 5)")
    parser.add_argument("-sl", "--stat_list", nargs='*',
                        help="List of statistics to compute")
    parser.add_argument("-o", "--out_file", default='./stats.tif',
                        help="Path to the output file (default: 'stats.tif')")
    args = parser.parse_args()

    # The actual work: first compute all indices from all images
    ndvis = [Raster(filename).ndvi(
        '{}_ndvi.tif'.format(os.path.basename(os.path.splitext(filename)[0])),
        idx_red=args.idx_red,
        idx_nir=args.idx_nir)
        for filename in args.rasters]
    ndwis = [Raster(filename).ndwi(
        '{}_ndwi.tif'.format(os.path.basename(os.path.splitext(filename)[0])),
        idx_nir=args.idx_nir,
        idx_mir=args.idx_mir)
        for filename in args.rasters]
    mndwis = [Raster(filename).mndwi(
        '{}_mndwi.tif'.format(os.path.basename(os.path.splitext(filename)[0])),
        idx_green=args.idx_green,
        idx_mir=args.idx_mir)
        for filename in args.rasters]

    # Then, block after block, compute the max, min
    blocks = Raster(args.rasters[0]).blocks()
    for block in blocks:
        hsize, vsize = block[2], block[3]
        array_blocks = []
        # Read all NDVIs in the block and concatenate them into a stack
        for ndvi in ndvis:
            ds = gdal.Open(ndvi.filename)
            band_block = ds.GetRasterBand(1)
            array_block = band_block.ReadAsArray(*block)
            array_blocks.append(array_block)
            ds = None
        ndvi_stack = np.dstack(array_blocks)
        # Compute the max NDVI for each pixel in the block and put the result
        # into an empty array
        ndvi_max = np.empty((vsize, hsize))
        np.amax(ndvi_stack, axis=2, out=ndvi_max)
        # Compute the index of the max for each pixel in the block
        ndvi_max_date = np.argmax(ndvi_stack, axis=2)
        # Compute the max NDVI for each pixel in the block and put the result
        # into an empty array
        ndvi_min = np.empty((vsize, hsize))
        np.amin(ndvi_stack, axis=2, out=ndvi_min)
        # Compute the index of the max for each pixel in the block
        ndvi_min_date = np.argmin(ndvi_stack, axis=2)
        stats = np.dstack((ndvi_max, ndvi_max_date, ndvi_min, ndvi_min_date))
        print(stats.shape)
        print(stats[:, :, 0])
        print(stats[:, :, 1])
        print(stats[:, :, 2])
        print(stats[:, :, 3])


if __name__ == "__main__":
    temporal_stats()
