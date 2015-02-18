#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from ymraster import Raster

import argparse


def command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Compute "
        "Modified Normalized Difference Water Index (MNDwI), "
        "also called Normalized Difference Snow Index (NDSI), "
        "of all the given images (green and mid-infrared bands must be at same "
        "position in each image) and save it into the specified output file.")
    parser.add_argument("in_list", nargs='+',
                        help="Path to the input image")
    parser.add_argument("-g", "--idx_green", type=int, required=True,
                        help="Index of green band in all input images")
    parser.add_argument("-mir", "--idx_mir", type=int, required=True,
                        help="Index of mid-infrared band in all input images")
    parser.add_argument("-o", "--out_file",
                        help="Path to the output file. A default file name is "
                        "choosen if omitted")
    return parser.parse_args()


def mndwi(args):
    for filename in args.in_list:
        raster = Raster(filename)
        raster.mndwi(args.idx_green, args.idx_mir, out_filename=args.out_file)


def main():
    args = command_line_arguments()
    mndwi(args)


if __name__ == "__main__":
    main()
