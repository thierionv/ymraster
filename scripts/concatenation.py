# -*- coding: utf-8 -*-

from ymraster import Raster, concatenate_rasters

import argparse


def command_line_arguments():
    parser = argparse.ArgumentParser(description="Write an image which is "
                                     "the concatenation of the given rasters "
                                     "in order. All input rasters must have "
                                     "same size.")
    parser.add_argument("raster", nargs='+',  help="Space separated list of "
                        "images to concatenate")
    parser.add_argument("-o", "--out_file", help="Path to the output file."
                        "By default, the first image is overwritten")
    return parser.parse_args()


def concatenate(args):
    rasters = [Raster(raster) for raster in args.raster]
    concatenate_rasters(*rasters, out_filename=args.out_file)


def main():
    args = command_line_arguments()
    concatenate(args)


if __name__ == "__main__":
    main()
