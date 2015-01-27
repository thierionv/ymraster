# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:40:31 2015

@author: sig
"""

if __name__ == "__main__":
    """
    """
    import argparse
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
                        "MeanSiftSmoothing application",required = True, 
                        type = int)
    parser.add_argument("--thres", "-th", help="Mean shift vector threshold ",
                        required = True, type = float)                
    parser.add_argument("--rangeramp", "-rga", help="Range radius coefficient"+
                        ": This coefficient makes dependent the ranger of the"+
                        " colorimetry of the filtered pixel : y = rangeramp" +
                        " * x + ranger.",required = True, type = float)
    parser.add_argument("-pref", "--prefixe", help ="Prefixe to add to the " +
                        "file to be written", default = "", type = str)
    parser.add_argument("-dir","--dir_file", default = "", help = "Path of "+
                        "the folder where the outputs will be written" )
    args = parser.parse_args()
    
    
    #Symbol to add in function of the optional parse arguments, to have a 
    #proper path
    if args.dir_file:
        args.dir_file += '/'
    if args.prefixe:
        args.prefixe += '_'
    print args
    
    #smoothing step
    xs = Raster(args.xs_file)
    output_filtered_image = args.dir_file + args.prefixe + "_spr_" + \
                            str(args.spatialr) + "_rg_" + str(args.ranger) + \
                            "_max_" + str(args.maxiter) + "_rga_" + \
                            str(args.rangeramp) + "_th_" + str(args.thres)\
                            + "filtered.tif"
    output_spatial_image = args.dir_file + args.prefixe + 'spatial.tif'
    smooth_img,pos_img = xs.lsms_smoothing(output_filtered_image, 
                                                   args.spatialr, args.ranger,
                                                   args.maxiter, args.thres, 
                                                   args.rangeramp, 
                                                   output_spatial_image )    
    print "smoothing step has been realized succesfully"