# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 10:11:29 2015

@author: sig
"""

if __name__ == "__main__":
    """
    """
    import argparse
    import os
    from ymraster import *
     
    #Set of the parse arguments
    desc = "Perform a chain of treatments from a given multi-spectral image" +\
            ". The treatments are : a fusion between the two images,a calcul"+\
            "ation of the ndvi band, an optional extraction of a chosen band"+\
            " a concatenation between the ndvi and the ms image, and a LSMS "+\
            " from this last image."
    parser = argparse.ArgumentParser(description= desc) 
    parser.add_argument("--xs_file", "-xs", help="Path of the multi-spectral" +
                        "image.",required = True)
    parser.add_argument("--pan_file","-pan", help="Path of the panchromatic " +
                        "image",required = True)
    parser.add_argument("--idx_red", "-red", help="Chanel number of the red " +
                        "band", type = int, required = True)
    parser.add_argument("--idx_nir", "-nir", help="Chanel number of the nir " +
                        "band", type = int, required = True)
    parser.add_argument("--estep", "-e",help="Do an extraction if notified",
                        action = "store_true")
    parser.add_argument("--idx", "-idx", help="Chanel number of the band to " +
                        "be removed. Indexation starts at 1.",required = True, 
                        type = int)
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
    parser.add_argument("-out", "--out_file", help ="Name of the output file",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the outputs will be written.")
    args = parser.parse_args()
    
    #control the coherency of the arguments
    if not(args.mstep) and args.minsize:
        print "Warning : --msize shouldn't be specified without --mstep"
    if not(args.estep) and args.idx:
        print "Warning : --idx shouldn't be specified without --estep"
    
    print args
    
    #Extraction of the input file name
    head, ext = os.path.splitext(args.xs_file)
    tail = os.path.basename(head)
    
    #--------------------------    
    # -------fusion -----------
    #--------------------------
    
    #set of the instances and the parameters   
    spot_xs = Raster(args.xs_file)
    spot_pan = Raster(args.pan_file)
    output_fusion = os.path.join(args.dir, tail + '_fusion.tif')
    
    #Execution of the method
    fus_img = spot_xs.fusion(spot_pan,output_fusion)
    print "Fusion step has been realized succesfully\n"    
    
    #--------------------------
    #------------ndvi----------
    #--------------------------
    
    #set of the parameters    
    output_ndvi = os.path.join(args.dir, tail + '_ndvi.tif')
    
    #Execution of the method
    ndvi_img = fus_img.ndvi(output_ndvi, args.idx_red, args.idx_nir)
    print "Writting the ndvi image has been realized succesfully\n"    

    #---------------------------    
    #---extraction (optional)---
    #---------------------------
    
    if args.estep:
        #set of the parameter
        output_rmv = os.path.join(args.dir, tail + '_extracted.tif')        
        
        #Execution of the method
        rmv_img = fus_img.remove_band(args.idx, output_rmv)
        print "Extraction step has been realized succesfully\n"    
    else:
        rmv_img = fus_img
    
    #--------------------------------------------
    #--Concatenate the rmv_img and the ndvi_img--
    #--------------------------------------------
    
    #set of the parameters
    list_im = [ndvi_img]
    output_concat = os.path.join(args.dir, tail + '_concatenated.tif')
    
    #execution of the method
    concat_img = rmv_img.concatenate( list_im, output_concat)
    print "Concatenation step has been realized succesfully\n"
    
    #--------------------------
    #-----------LSMS-----------    
    #--------------------------
    
    #first step : smoothing
    output_filtered_image = os.path.join(args.dir, tail + "_spr_" + \
                            str(args.spatialr) + "_rg_" + str(args.ranger) + \
                            "_max_" + str(args.maxiter) + "_rga_" + \
                            str(args.rangeramp) + "_th_" + str(args.thres)\
                            + "_filtered.tif")
    output_spatial_image = os.path.join(args.dir, tail + '_spatial.tif')
    smooth_img,pos_img = concat_img.lsms_smoothing(output_filtered_image, 
                                           args.spatialr, args.ranger,
                                           output_spatial_image, thres = 
                                           args.thres, rangeramp = 
                                           args.rangeramp, maxiter = 
                                           args.maxiter, modesearch = 
                                           args.modesearch)    
    print "smoothing step has been realized succesfully\n"
    
    #second step : segmentation
    if not(args.mstep or args.vstep): #If this is the final outup or not
        output_seg = os.path.join(args.dir, args.out_file)
    else:
        output_seg = os.path.join(args.dir, tail + '_lsms_seg.tif')
    seg_img = smooth_img.lsms_seg (pos_img, output_seg, args.spatialr, 
                                   args.ranger, tilesizex = args.tilesizex,
                                   tilesizey = args.tilesizey)
    print "segmentation step has been realized succesfully"
    
    #third step (optional) : merging small regions
    if args.mstep:
        if not args.vstep : #If this is the final outup or not
            output_merged = os.path.join(args.dir, args.out_file)
        else:
            output_merged = os.path.join(args.dir, tail + '_lsms_merged.tif')
        merged_img = seg_img.lsms_merging(smooth_img, output_merged, 
                                          args.minsize, tilesizex = \
                                          args.tilesizex,tilesizey = \
                                          args.tilesizey)
        print "merging step has been realized succesfully"
    else:
        merged_img = seg_img
        
    #fourth step (optional) : vectorization
    if args.vstep: 
        output_vector = os.path.join(args.dir, args.out_file)
        merged_img.lsms_vectorisation(concat_img, output_vector, tilesizex = \
                                    args.tilesizex,tilesizey = args.tilesizey)
        print "vectorisation step has been realized succesfully"