# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 18:15:27 2015

@author: sig
"""

if __name__ == "__main__":
    """
    """
    import argparse
    from ymraster import *
     
    #Set of the parse arguments
    parser = argparse.ArgumentParser(description= "Third step LSMS :  merge " +
                                    "regions, whose size in pixels is lower " +
                                    "than minsize parameter, with" +
                                    " the adjacent region with the " +
                                    "closest radiometry and acceptable size, "+
                                    "using the LSMSSmallRegionsMerging otb " +
                                    "application. It returns  the merged " +
                                    "image.")
    parser.add_argument("--seg_file", "-seg", help="Path of the segmented "+
                        "image.",required = True)
    parser.add_argument("--filtered_file", "-fil", help="Path of the filtered"
                        + " image.",required = True)
    parser.add_argument("--minsize", "-ms",help="minimum size of a label",
                        type = int, required = True)
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
    
    #merging small regions 
    seg_img = Raster(args.seg_file)
    smooth_img = Raster(args.filtered_file)
    output_merged = args.dir_file + args.prefixe + 'lsms_merged.tif'
    merged_img = seg_img.lsms_merging(smooth_img, output_merged, 
                                          args.minsize)
    print "merging step has been realized succesfully"
    
