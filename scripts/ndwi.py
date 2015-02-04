# -*- coding: utf-8 -*-

import argparse
from ymraster import Raster

import os.path
import string


def compute_ndwi():
    # Command-line parameters
    parser = argparse.ArgumentParser(
        description="Compute Normalized Difference Water Index (NDWI) "
        "of all the given images (near-infrared and mid-infrared bands must be "
        "at same position) and save it into the specified output file.")
    parser.add_argument("in_list", nargs='+',
                        help="Path to the input image")
    parser.add_argument("-nir", "--idx_nir", type=int, required=True,
                        help="Index of near-infrared band in all input images")
    parser.add_argument("-mir", "--idx_mir", type=int, required=True,
                        help="Index of mid-infrared band in all input images")
    parser.add_argument("-o", "--out_file",
                        help="Path to the output file. Default is "
                        "'${basename}.ndwi.tif' in the current folder. "
                        "${basename} is the input filename without extension. "
                        "If you specify more than one images, you really "
                        "should use ${basename}.")
    args = parser.parse_args()

    # Do the actual work
    for filename in args.in_list:
        raster = Raster(filename)
        if args.out_file is None:
            out_filename = string.Template(
                "${basename}.ndwi.tif").substitute(
                    {'basename':
                     os.path.basename(os.path.splitext(filename)[0])})
            raster.ndwi(out_filename, args.idx_red, args.idx_nir)


if __name__ == "__main__":
    compute_ndwi()
