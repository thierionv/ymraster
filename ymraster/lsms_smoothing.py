# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:40:31 2015

@author: sig
"""

if __name__ == "__main__":
    """
    """
    import argparse
    import os
    from ymraster import *

    #Set of the parse arguments
    parser = argparse.ArgumentParser(description= "First step of LSMS : " +
                                    "perform a mean shift fitlering, using " +
                                    "the MeanShiftSmoothing otb application." +
                                    " It writes two images : the filtered one"+
                                    "and the spatial one")
    parser.add_argument("--xs_file", "-xs", help="Path of the multi-spectral" +
                        "image.",required = True)
    parser.add_argument("--spatialr", "-spr", help="Spatial radius of the " +
                        "neighborhooh",required = True, type = int)
    parser.add_argument("--ranger", "-rg", help="Range radius defining the " +
                        "radius (expressed in radiometry unit) in the multi" +
                        "-spectral space.",required = True, type = float)
    parser.add_argument("--maxiter", "-max", help="Maximum number of " +
                        "iterations of the algorithm used in "+
                        "MeanSiftSmoothing application (default value is 10)",
                        type = int, default = 10)
    parser.add_argument("--thres", "-th", help="Mean shift vector threshold" +
                        "(default value is 0.1).", type = float, default = 0.1)
    parser.add_argument("--rangeramp", "-rga", help="Range radius coefficient"+
                        ": This coefficient makes dependent the ranger of the"+
                        " colorimetry of the filtered pixel : y = rangeramp" +
                        " * x + ranger(default value is 0).", type = float,
                         default = 0)
    parser.add_argument("--modesearch", "-mos", help="Mean shift vector threshold ",
                        type = int, default = 0)
    parser.add_argument("-out", "--out_file", help ="Name of the output files"+
                        ". Two arguments expected, corresponding respectively"+
                        " to the filetered image and the spatial image" ,
                        required = True, type = str, nargs = 2)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the output will be written.")
    args = parser.parse_args()
    print args

    #set of the instance and output names
    xs = Raster(args.xs_file)
    out_smoothed_filename = os.path.join(args.dir, args.out_file[0])
    out_spatial_filename = os.path.join(args.dir, args.out_file[1])

    #Execution of the method
    smooth_img,pos_img = xs.lsms_smoothing(out_smoothed_filename,
                                           args.spatialr, args.ranger,
                                           out_spatial_filename, thres =
                                           args.thres, rangeramp =
                                           args.rangeramp, maxiter =
                                           args.maxiter, modesearch =
                                           args.modesearch)
    print "smoothing step has been realized succesfully\n"
