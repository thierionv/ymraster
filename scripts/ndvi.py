# -*- coding: utf-8 -*-

import argparse
from ymraster import Raster

import os.path
import string


def compute_ndvi():
    # Command-line parameters
    parser = argparse.ArgumentParser(
        description="Compute Normalized Difference Vegetation Index (NDVI) "
        "of all the given images (red and near-infrared bands must be at same "
        "position) and save it into the specified output file.")
    parser.add_argument("in_list", nargs='+',
                        help="Path to the input image")
    parser.add_argument("-r", "--idx_red", type=int, required=True,
                        help="Index of red band in all input images")
    parser.add_argument("-nir", "--idx_nir", type=int, required=True,
                        help="Index of near-infrared band in all input images")
    parser.add_argument("-o", "--out_file",
                        help="Path to the output file. Default is "
                        "'${basename}.ndvi.tif' in the current folder. "
                        "${basename} is the input filename without extension. "
                        "If you specify more than one images, you really "
                        "should use ${basename}.")
    args = parser.parse_args()

    # Do the actual work
    for filename in args.in_list:
        raster = Raster(filename)
        if args.out_file is None:
            out_filename = string.Template(
                "${basename}.ndvi.tif").substitute(
                    {'basename':
                     os.path.basename(os.path.splitext(filename)[0])})
            raster.ndvi(out_filename, args.idx_red, args.idx_nir)


if __name__ == "__main__":
    compute_ndvi()
