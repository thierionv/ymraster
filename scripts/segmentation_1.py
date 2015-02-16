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
            " a concatenation between the ndvi and the ms image, an optional"+\
            "mask application and a LSMS from this last image."
    parser = argparse.ArgumentParser(description= desc)
    parser.add_argument("--xs_file", "-xs", help="Path of the multi-spectral" +
                        "image.",required = True)
    parser.add_argument("--pan_file","-pan", help="Path of the panchromatic " +
                        "image",required = True)
    parser.add_argument("--idx_red", "-red", help="Chanel number of the red " +
                        "band", type = int, required = True)
    parser.add_argument("--idx_nir", "-nir", help="Chanel number of the nir " +
                        "band", type = int, required = True)
    parser.add_argument("--estep", "-e",help="(optional) Do an extraction if "+
                        "notified", action = "store_true")
    parser.add_argument("--idx", "-idx", help="(Required only if --estep" +
                        " is specified). List of index of the band(s) to be " +
                        "removed. Indexation starts at 1.",default = [],
                        type = int, nargs ='+')
    parser.add_argument("--mask", "-mk",  help="(optional)Path of the mask to"+
                        " apply. The mask must contend two values, one that " +
                        "represents the pixels to hide, and an other to those"+
                        " that are not to hide", default = "")
    parser.add_argument("--in_mask_value", "-inv",  help="(optional, relevant"+
                        " only if --mask is specified). The value of the " +
                        "pixels masked in mask raster. The default value is "+
                        "-9999", type = int, default = -9999)
    parser.add_argument("--out_mask_value", "-outv",  help="(optional, " +
                        "relevant only if --mask is specified). The value to "+
                        "set to the pixels masked in the output file. The " +
                        "default value is 65636", type = int, default = 65636)
    parser.add_argument("--spatialr", "-spr", help="Spatial radius of the " +
                        "neighborhooh",required = True, type = int)
    parser.add_argument("--ranger", "-rg", help="Range radius defining the " +
                        "radius (expressed in radiometry unit) in the multi" +
                        "-spectral space.",required = True, type = float)
    parser.add_argument("--maxiter", "-max", help="(optional). Maximum number "+
                        "of iterations of the algorithm used in "+
                        "MeanSiftSmoothing application (default value is 10)",
                        type = int, default = 10)
    parser.add_argument("--thres", "-th", help="(optional). Mean shift vector "+
                        "threshold (default value is 0.1).", type = float, 
                        default = 0.1)
    parser.add_argument("--rangeramp", "-rga", help="(optional). Range radius "+ 
                        " coefficient : This coefficient makes dependent the "+
                        "ranger of the colorimetry of the filtered pixel : " +
                        "y = rangeramp * x + ranger(default value is 0).",
                         type = float, default = 0)
    parser.add_argument("--modesearch", "-mos", help="(optional). Mean shift "+
                        " vector thres hold (default value is 0)",type = int,
                        default = 0)
    parser.add_argument("--tilesizex", "-tx",help="(optional). Size of tiles "+
                        "along the X-axis (default value is 256)", type = int,
                         default = 256)
    parser.add_argument("--tilesizey", "-ty",help="(optional). Size of tiles "+
                        "along the Y-axis (default value is 256)", type = int,
                         default = 256)
    parser.add_argument("--mstep", "-m",help="(optional). Do the merge step "+
                        "if specified", action = "store_true")
    parser.add_argument("--minsize", "-ms",help="(required only if --mstep is "+
                        "specified). Minimum size of a label", type = int)
    parser.add_argument("--vstep", "-v",help="(optional). Do the vectorisation"+
                        " step if specified", action = "store_true")
    parser.add_argument("-out", "--out_file", help ="Name of the output file. "+
                        "The extension of the output file depends on what is "+
                        "the last operation performed, eg : if --vstep is "+
                        "specified, it must be something like \"my_output.shp"+
                        "\", otherwise something like \"my_output.tif\".",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the outputs will be written.")
    args = parser.parse_args()
    print "\n"
    #control the coherency of the arguments
    spot_xs = Raster(args.xs_file)
    d = spot_xs.meta['count']
    if not(args.mstep) and args.minsize:
        print "Warning : --msize shouldn't be specified without --mstep\n"
    if args.estep: # if the argument extraction step is specified 
        if not args.idx : # if the --idx argument is not specified
            print "Warning : none index specified in --idx argument.\n"
        else:
            if not all ([(boo in range(1,d+1)) for boo in args.idx ]):
                print "Error : one of the index specified is out of range.\n"
                exit()
            if sorted(args.idx) == range(1,d+1):
                print "Error : you can not remove all the bands.\n"
                exit()
    else:
        if args.idx:
            print "Warning : --idx shoud not be specified without --estep.\n"
    print args,"\n"
    
    #Extraction of the input file name
    head, ext = os.path.splitext(args.xs_file)
    tail = os.path.basename(head)

    #--------------------------
    # -------fusion -----------
    #--------------------------

    #set of the instances and the parameters
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

    #--------------------------------------------
    #-----------Apply a mask (optional)----------
    #--------------------------------------------

    if args.mask:
        #Set of the instances and the output parameter
        mask_img = Raster(args.mask)
        output_masked = os.path.join(args.dir, tail + '_masked.tif')
        
        #Execution of the method
        masked_img = concat_img.apply_mask( mask_img, args.in_mask_value,
                                           output_masked,
                                           out_mask_value = args.out_mask_value)
        print "The mask has been applied succesfully\n"
    else:
        masked_img = concat_img

    #--------------------------
    #-----------LSMS-----------
    #--------------------------

    #first step : smoothing
    out_smoothed_filename = os.path.join(args.dir, tail + "_spr_" + \
                            str(args.spatialr) + "_rg_" + str(args.ranger) + \
                            "_max_" + str(args.maxiter) + "_rga_" + \
                            str(args.rangeramp) + "_th_" + str(args.thres)\
                            + "_filtered.tif")
    out_spatial_filename = os.path.join(args.dir, tail + '_spatial.tif')
    smooth_img,pos_img = masked_img.lsms_smoothing(out_smoothed_filename,
                                           args.spatialr, args.ranger,
                                           out_spatial_filename, thres =
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
    seg_img = smooth_img.lsms_segmentation(pos_img, output_seg, args.spatialr,
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
        merged_img.lsms_vectorization(concat_img, output_vector, tilesizex = \
                                    args.tilesizex,tilesizey = args.tilesizey)
        print "vectorization step has been realized succesfully"
