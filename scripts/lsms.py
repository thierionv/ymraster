# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 14:26:20 2015

@author: sig
"""
if __name__ == "__main__":
    """
    """
    import argparse
    import os
    from ymraster import *

    #Set of the parse arguments
    parser = argparse.ArgumentParser(description= "Perform a Large-Scale Mean"+
                                    "-Shift segmentation in four step:  a " +
                                    "mean shift fitlering, a segmentation, " +
                                    " a merge of small regions (optional)" +
                                    ", a vectorisation (optional). This " +
                                    "application works using LSMS otb " +
                                    "application.")
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
    parser.add_argument("--modesearch", "-mos", help="Mean shift vector thres"+
                        "hold (default value is 0)",type = int,default = 0)
    parser.add_argument("--tilesizex", "-tx",help="Size of tiles along the "+
                        "X-axis (default value is 256)", type = int, default =
                        256)
    parser.add_argument("--tilesizey", "-ty",help="Size of tiles along the "+
                        "Y-axis (default value is 256)", type = int, default =
                        256)
    parser.add_argument("--mstep", "-m",help="Do the merge step if notified",
                        action = "store_true")
    parser.add_argument("--minsize", "-ms",help="minimum size of a label",
                        type = int)
    parser.add_argument("--vstep", "-v",help="Do the vectorisation step if "+
                        "notified", action = "store_true")
    parser.add_argument("--delete", "-del",help="Delete the transitional steps"+
                        " specified; e.g. if 1 is specified the filtered image"+
                        " will be deleted at the end of the treatment. By " +
                        "default, all the transitional files are kept.",
                        default = [], choices = [1,2,3], type = int, 
                        nargs = '+')                
    parser.add_argument("-out", "--out_file", help ="Name of the output file",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the outputs will be written.")
    args = parser.parse_args()
    
    print "\n"
    #control the coherency of the arguments
    if not(args.mstep) and args.minsize:
        print "Warning : --msize shouldn't be specified without --mstep\n"
    if not(args.vstep) and args.mstep and 3 in args.delete:
        print "Error : The final file can not be deleted. Check the --delete"+\
                " option.\n"
        exit()
    if not(args.vstep) and not(args.mstep) and 2 in args.delete:
        print "Error : The final file can not be deleted. Check the --delete"+\
                " option.\n"
        exit()   
    print args, "\n"

    #Extraction of the input file name
    head, ext = os.path.splitext(args.xs_file)
    tail = os.path.basename(head)

    #first step : smoothing
    xs = Raster(args.xs_file)
    out_smoothed_filename = os.path.join(args.dir, tail + "_spr_" + \
                            str(args.spatialr) + "_rg_" + str(args.ranger) + \
                            "_max_" + str(args.maxiter) + "_rga_" + \
                            str(args.rangeramp) + "_th_" + str(args.thres)\
                            + "filtered.tif")
    out_spatial_filename = os.path.join(args.dir, tail + '_spatial.tif')
    smooth_img,pos_img = xs.lsms_smoothing(out_smoothed_filename,
                                           args.spatialr, args.ranger,
                                           out_spatial_filename, thres =
                                           args.thres, rangeramp =
                                           args.rangeramp, maxiter =
                                           args.maxiter, modesearch =
                                           args.modesearch)
    print "smoothing step has been realized succesfully\n"

    #second step : segmentation
    if not(args.mstep or args.vstep):#If this is the final outup or not
        output_seg = os.path.join(args.dir, args.out_file)
    else:
        output_seg = os.path.join(args.dir, tail + '_lsms_seg.tif')
    seg_img = smooth_img.lsms_segmentation (pos_img, output_seg, args.spatialr,
                                   args.ranger, tilesizex = args.tilesizex,
                                   tilesizey = args.tilesizey)
    print "segmentation step has been realized succesfully\n"
    

    #third step (optional) : merging small regions
    if args.mstep:
        if not args.vstep: #If this is the final output or not
            output_merged = os.path.join(args.dir, args.out_file)
        else:
            output_merged = os.path.join(args.dir, tail + '_lsms_merged.tif')
        merged_img = seg_img.lsms_merging(smooth_img, output_merged,
                                          args.minsize, tilesizex = \
                                          args.tilesizex,tilesizey = \
                                          args.tilesizey)
        print "merging step has been realized succesfully\n"
    else:
        merged_img = seg_img

    
        
    #fourth step (optional) : vectorization
    if args.vstep:
        output_vector = os.path.join(args.dir, args.out_file)
        merged_img.lsms_vectorization(xs, output_vector, tilesizex = \
                                    args.tilesizex,tilesizey = args.tilesizey)
        print "vectorization step has been realized succesfully\n"

    #delete the transitional files if specified in --delete argument
    if 1 in args.delete:
        os.remove(out_smoothed_filename)
        os.remove(out_spatial_filename)
        print "Step 1 files have been deleted.\n"
    if 2 in args.delete:
        os.remove(output_seg)
        print "Step 2 file has been deleted.\n"
    if 3 in args.delete and args.mstep:
        os.remove(output_merged)
        print "Step 3 file has been deleted.\n"