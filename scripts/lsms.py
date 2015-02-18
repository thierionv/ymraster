#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from ymraster import Raster

import argparse


def command_line_arguments():
    parser = argparse.ArgumentParser(
        description="Performs a Large-Scale-Mean-Shift (LSMS) object "
        "segmentation on a raster.\n\n"
        "Produces a labeled image and (optionally) a labeled vector file")
    parser.add_argument("raster",
                        help="Path to the raster on which to perform "
                        "segmentation.")
    parser.add_argument("-r", "--spatialr", type=int, required=True,
                        help="Spatial radius of the window (in pixel)")
    parser.add_argument("-s", "--spectralr", type=float, required=True,
                        help="Spectral radius (expressed in radiometry unit)")
    parser.add_argument("-t", "--thres", type=float, default=0.1,
                        help="Mean shift vector threshold (default: 0.1)")
    parser.add_argument("-ra", "--rangeramp", type=float, default=0,
                        help="Range radius coefficient (default: 0)")
    parser.add_argument("-m", "--maxiter", type=int, default=10,
                        help="Maximum number of iterations of the algorithm in "
                        "case of non-convergence (default: 10)")
    parser.add_argument("-ms", "--minsize", type=int,
                        help="minimum size of each object")
    parser.add_argument("-vf", "--vector_file",
                        help="Path to the vector file."
                        "No vector file is produced if omitted")
    parser.add_argument("-o", "--out_file",
                        help="Path to the output file. "
                        "A default file name is chosen if omitted")
    return parser.parse_args()


def lsms_segmentation(args):
    raster = Raster(args.raster)
    raster.lsms_segmentation(args.spatialr, args.spectralr, args.thres,
                             args.rangeramp, args.maxiter,
                             object_minsize=args.minsize,
                             out_vector_filename=args.vector_file,
                             out_filename=args.out_file)


def main():
    args = command_line_arguments()
    lsms_segmentation(args)

if __name__ == "__main__":
    main()
