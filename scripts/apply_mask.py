#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from ymraster import Raster

import argparse


def command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Apply a mask to each given raster, that is set a common "
        "value to all pixels that are under a given mask.")
    parser.add_argument("raster", nargs="+",
                        help="Space separated list of raster on which to apply "
                        "the mask")
    parser.add_argument("-m", "--mask",  required=True,
                        help="Path to the mask to use")
    parser.add_argument("-v", "--mask-value", type=int, default=1,
                        help='"Masked" value in the mask raster (default: 1)')
    parser.add_argument("-s", "--set_value",  type=int,
                        help='Value to set to the "masked" pixels in the '
                        "output file. Default is the max of the data type")
    parser.add_argument("-o", "--out_file",
                        help="Path to the output file. The original raster "
                        "file is overwritten if omitted")
    return parser.parse_args()


def apply_mask(args):
    mask_raster = Raster(args.mask)
    for filename in args.raster:
        raster = Raster(filename)
        raster.apply_mask(mask_raster=mask_raster,
                          mask_value=args.mask_value,
                          set_value=args.set_value,
                          out_filename=args.out_file)


def main():
    args = command_line_arguments()
    apply_mask(args)


if __name__ == "__main__":
    main()
