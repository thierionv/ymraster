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
                        " or \"\\\" have to be added at the end.")   
    args = parser.parse_args()
    print args
    
    #set of the instances and the output name
    seg_img = Raster(args.seg_file)
    smooth_img = Raster(args.filtered_file)
    output_merged = args.dir + args.out_file
    
    #Execution of the method
    merged_img = seg_img.lsms_merging(smooth_img, output_merged, 
                                          args.minsize, tilesizex = \
                                          args.tilesizex,tilesizey = \
                                          args.tilesizey)
    print "merge has been realized succesfully"
    
