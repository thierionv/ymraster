# -*- coding: utf-8 -*-

from ymraster import Raster

import argparse


def command_line_arguments():
    parser = argparse.ArgumentParser(description="Apply a mask to a raster "
                                     "that is set a common value to all pixels "
                                     "that are under a given mask.")
    parser.add_argument("raster",  help="Path to the raster on which to apply "
                        "the mask")
    parser.add_argument("-m", "--mask",  required=True, help="Path to the mask")
    parser.add_argument("-v", "--mask-value", type=int, default=1,
                        help='"Masked" value in the mask raster')
    parser.add_argument("-s", "--set_value",  type=int, help='Value to set '
                        'to the "masked" pixels in the output file. Default is '
                        "the max of the data type")
    parser.add_argument("-o", "--out_file", required=True, help="Path to the "
                        "output file. By default it is the max of the data "
                        "type")
    args = parser.parse_args()
    return args


def apply_mask(args):
    raster = Raster(args.raster)
    mask_raster = Raster(args.mask)
    raster.apply_mask(mask_raster=mask_raster,
                      mask_value=args.mask_value,
                      set_value=args.set_value,
                      out_filename=args.out_file)


def main():
    args = command_line_arguments()
    apply_mask(args)


if __name__ == "__main__":
    main()
