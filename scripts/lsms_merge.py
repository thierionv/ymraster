#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from ymraster import Raster

import argparse


def command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Merge small objects of a labelled raster into bigger "
        "adjacent objects.\n\n"
        "This is the optional third step of a Large-Scale-Mean-Shift (LSMS) "
        "object segmentation. It makes use of the smoothed image produce by "
        "the first step (smoothing) to determine in which adjacent bigger "
        "object the small one will be merged.")
    parser.add_argument("raster",
                        help="Path to the labeled raster to merge.")
    parser.add_argument("-s", "--smoothed_file", required=True,
                        help="Path to the smoothed image that was used to "
                        "produce the labeled raster.")
    parser.add_argument("-m", "--minsize", type=int, required=True,
                        help="Minimum size of an object/label")
    parser.add_argument("-o", "--out_file",
                        help="Path to the output file. "
                        "The original labeled file is overwritten if omitted")
    return parser.parse_args()


def lsms_merge(args):
    raster = Raster(args.raster)
    smoothed_raster = Raster(args.smoothed_file)
    raster._lsms_merging(args.minsize, smoothed_raster,
                         out_filename=args.out_file)


def main():
    args = command_line_arguments()
    lsms_merge(args)


if __name__ == "__main__":
    main()
