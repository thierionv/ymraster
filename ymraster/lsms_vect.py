# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 09:41:11 2015

@author: sig
"""

if __name__ == "__main__":
    """
    """
    import argparse
    import os
    from ymraster import *
     
    #Set of the parse arguments
    parser = argparse.ArgumentParser(description= "Final step of LSMS : " +
                                    "convert a label image to a GIS vector " +
                                    "file containing one polygon per segment,"+
                                    " using the LSMSVectorization otb " +
                                    "application.") 
    parser.add_argument("--xs_file", "-xs", help="Path of the multi-spectral" +
                        " image.",required = True)
    parser.add_argument("--seg_file", "-seg", help="Path of the segmented" +
                        " image.",required = True)
    parser.add_argument("--tilesizex", "-tx",help="Size of tiles along the "+
                        "X-axis (default value is 256)", type = int, default =
                        256)
    parser.add_argument("--tilesizey", "-ty",help="Size of tiles along the "+
                        "Y-axis (default value is 256)", type = int, default =
                        256)
    parser.add_argument("-out", "--out_file", help ="Name of the output file",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the output will be written.")
    args = parser.parse_args()
    print args
    
    #set of the instances and the output name
    output_vector = os.path.join(args.dir, args.out_file)
    merged_img = Raster(args.seg_file)
    xs = Raster(args.xs_file)
    
    #Execution of the method
    merged_img.lsms_vectorization(xs, output_vector, tilesizex = \
                                    args.tilesizex,tilesizey = args.tilesizey)
    print "Vectorization has been realized succesfully"