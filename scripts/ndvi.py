#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ymraster import Raster

import argparse


def command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Compute Normalized Difference Vegetation Index (NDVI) "
        "of all the given images (red and near-infrared bands must be at same "
        "position in each image) and save it into the specified output file.")
    parser.add_argument("in_list", nargs='+',
                        help="Path to the input image")
    parser.add_argument("-r", "--idx_red", type=int, required=True,
                        help="Index of red band in all input images")
    parser.add_argument("-nir", "--idx_nir", type=int, required=True,
                        help="Index of near-infrared band in all input images")
    parser.add_argument("-o", "--out_file",
                        help="Path to the output file. A default file name is "
                        "chosen if omitted")
    return parser.parse_args()


def ndvi(args):
    for filename in args.in_list:
        raster = Raster(filename)
        raster.ndvi(args.idx_red, args.idx_nir, out_filename=args.out_file)


def main():
    args = command_line_arguments()
    ndvi(args)


if __name__ == "__main__":
    main()
