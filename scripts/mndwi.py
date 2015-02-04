# -*- coding: utf-8 -*-

import argparse
from ymraster import Raster

import os.path
import string


def compute_mndwi():
    # Command-line parameters
    parser = argparse.ArgumentParser(
        description="Compute "
        "Modified Normalized Difference Water Index (MNDwI) "
        "also called Normalized Difference Snow Index (NDSI) "
        "of all the given images (green and mid-infrared band must be at same "
        "position) and save it into the specified output file.")
    parser.add_argument("in_list", nargs='+',
                        help="Path to the input image")
    parser.add_argument("-g", "--idx_green", type=int, required=True,
                        help="Index of green band in all input images")
    parser.add_argument("-mir", "--idx_mir", type=int, required=True,
                        help="Index of mid-infrared band in all input images")
    parser.add_argument("-o", "--out_file",
                        help="Path to the output file. Default is "
                        "'${basename}.mndwi.tif' in the current folder. "
                        "${basename} is the input filename without extension. "
                        "If you specify more than one images, you really "
                        "should use ${basename}.")
    args = parser.parse_args()

    # Do the actual work
    for filename in args.in_list:
        raster = Raster(filename)
        if args.out_file is None:
            out_filename = string.Template(
                "${basename}.mndwi.tif").substitute(
                    {'basename':
                     os.path.basename(os.path.splitext(filename)[0])})
            raster.mndwi(out_filename, args.idx_red, args.idx_nir)


if __name__ == "__main__":
    compute_mndwi()
