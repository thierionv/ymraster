# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 17:52:54 2015

@author: sig
"""

if __name__ == "__main__":
    """
    """
    import argparse
    from ymraster import *
     
    #Set of the parse arguments
    parser = argparse.ArgumentParser(description= "Second step of LSMS : " +
                                    "produce a labeled image with different " +
                                    "clusters, according to the range and " +
                                    "spatial proximity of the pixels, using " +
                                    "the LSMSSegmentation otb application. It"+
                                    " returns the segmented image.") 
    parser.add_argument("--filtered_file", "-fil", help="Path of the filtered"
                        + "image.",required = True)
    parser.add_argument("--pos_file", "-pos", help="Path of the spatial"
                        + "image.",required = True)
    parser.add_argument("--spatialr", "-spr", help="Spatial radius of the " +
                        "neighborhooh. It should be the same that specified" +
                        "in the smoothing step",required = True, type = int)
    parser.add_argument("--ranger", "-rg", help="Range radius defining the " +
                        "radius (expressed in radiometry unit) in the multi" +
                        "-spectral space. It should be the same that specified"
                        + " in the smoothing step",required = True,
                        type = float)
    parser.add_argument("-pref", "--prefixe", help ="Prefixe to add to the " +
                        "file to be written", default = "", type = str)
    parser.add_argument("-dir","--dir_file", default = "", help = "Path of "+
                        "the folder where the output will be written" )
    args = parser.parse_args()
    

    #Symbol to add in function of the optional parse arguments, to have a 
    #proper path
    if args.dir_file:
        args.dir_file += '/'
    if args.prefixe:
        args.prefixe += '_'
    print args
    
    #segmentation step
    output_seg = args.dir_file + args.prefixe + 'lsms_seg.tif'
    smooth_img = Raster(args.filtered_file)
    pos_file = Raster(args.pos_file)
    seg_img = smooth_img.lsms_seg (pos_file, output_seg, args.spatialr, 
                                   args.ranger)
                                   
    print "segmentation step has been realized succesfully"
    