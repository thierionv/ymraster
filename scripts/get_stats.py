#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from ymraster import Raster

import argparse


def command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Compute per object/label statistics on a raster, given a "
        "label file")
    parser.add_argument("raster",
                        help="Path to the raster to compute statistics from")
    parser.add_argument("-s", "--stats", nargs='+',
                        default=["min", "max", "mean", "std", "per:20",
                                 "per:40", "median", "per:60", "per:80"],
                        help="Space separated fist of statistics to compute")
    parser.add_argument("-l", "--label_file", required=True,
                        help="Path to the label image")
    parser.add_argument("-o", "--out_file",
                        help="Path to the output file. "
                        "A default name is chosen if omitted")
    return parser.parse_args()


def label_stats(args):
    raster = Raster(args.raster)
    label_raster = Raster(args.label_file)
    raster.label_stats(args.stats, label_raster=label_raster,
                       out_filename=args.out_file)


def main():
    args = command_line_arguments()
    label_stats(args)


if __name__ == "__main__":
    main()
