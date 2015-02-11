# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 11:21:18 2015

@author: sig
"""

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
                        "removed. Indexation starts at 1.", default = [],
                        type = int, nargs = '+')
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
    if not args.idx :
        print "Warning : none index specified in --idx argument.\n"
    if not all ([(boo in range(1,d+1)) for boo in args.idx ]):
        print "Error : one of the index specified is out of range.\n"
        exit()
    if sorted(args.idx) == range(1,d+1):
        print "Error : you can not remove all the bands.\n"
        exit()    
    print args, "\n"

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
    if args.mask:
        output_concat = os.path.join(args.dir, tail + '_concatenated.tif')
    else:
        output_concat = os.path.join(args.dir, args.out_file)

    #execution of the method
    concat_img = rmv_img.concatenate( list_im, output_concat)
    print "Concatenation step has been realized succesfully\n"

    #--------------------------------------------
    #-----------Apply a mask (optional)----------
    #--------------------------------------------

    if args.mask:
        #Set of the instances and the output parameter
        mask_img = Raster(args.mask)
        output_masked = os.path.join(args.dir, args.out_file)
        
        #Execution of the method
        masked_img = concat_img.apply_mask( mask_img, args.in_mask_value,
                                           output_masked,
                                           out_mask_value = args.out_mask_value)
        print "The mask has been applied succesfully\n"
    else:
        masked_img = concat_img

    