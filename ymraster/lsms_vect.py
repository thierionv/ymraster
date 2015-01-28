# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 09:41:11 2015

@author: sig
"""

if __name__ == "__main__":
    """
    """
    import argparse
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
    
    #fourth step (optional) : vectorization 
    output_vector = args.dir_file + args.prefixe + 'lsms_vect.shp'
    merged_img = Raster(args.seg_file)
    xs = Raster(args.xs_file)
    merged_img.lsms_vectorisation(xs, output_vector, tilesizex = \
                                    args.tilesizex,tilesizey = args.tilesizey)
    print "vectorisation step has been realized succesfully"