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
    parser.add_argument("--tilesizex", "-tx",help="Size of tiles along the "+
                        "X-axis (default value is 256)", type = int, default =
                        256)
    parser.add_argument("--tilesizey", "-ty",help="Size of tiles along the "+
                        "Y-axis (default value is 256)", type = int, default =
                        256)
    parser.add_argument("-out", "--out_file", help ="Name of the output file",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the output will be written. The \"/\"" +
                        " or \"\\\" have to be add at the end.")
    args = parser.parse_args()
    print args
    
    #set of the instances and the output name
    output_seg = args.dir + args.out_file 
    smooth_img = Raster(args.filtered_file)
    pos_img = Raster(args.pos_file)
    
    #Execution of the method
    seg_img = smooth_img.lsms_seg (pos_img, output_seg, args.spatialr, 
                                   args.ranger, tilesizex = args.tilesizex,
                                   tilesizey = args.tilesizey)
                                   
    print "Segmentation has been realized succesfully"
    