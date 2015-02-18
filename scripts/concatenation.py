#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from ymraster import Raster, concatenate_rasters

import argparse


def command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Write an image which is the concatenation of the given "
        "rasters in order.")
    parser.add_argument("raster", nargs='+',
                        help="Space separated list of rasters to concatenate")
    parser.add_argument("-o", "--out_file",
                        help="Path to the output file. The first image is "
                        "overwritten if omitted")
    return parser.parse_args()


def concatenate(args):
    rasters = [Raster(raster) for raster in args.raster]
    concatenate_rasters(*rasters, out_filename=args.out_file)


def main():
    args = command_line_arguments()
    concatenate(args)


if __name__ == "__main__":
    main()
