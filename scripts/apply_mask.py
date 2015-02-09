# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 10:05:46 2015

@author: sig
"""

if __name__ == "__main__":
    """Executable of the extraction part
    """
    
    import argparse
    import os
    from ymraster import *
    
    desc = "Apply a mask to an image. It can be a multi-band image."
    #Set of the parse arguments
    parser = argparse.ArgumentParser(description= desc) 
    parser.add_argument("--im_file", "-im",  help="The raster path on which "+
                        "to apply the mask", required = True)
    parser.add_argument("--mask", "-mk",  help="Path of the mask to apply." +
                        "The mask must contend two values, one that represents"+
                        " the pixels to hide, and an other to those that are" 
                        " not to hide", required = True)
    parser.add_argument("--in_mask_value", "-inv",  help="The value of the " +
                        "pixels masked in mask raster. The default value is "+
                        "-9999", type = int)
    parser.add_argument("--out_mask_value", "-outv",  help="The value to set "+
                        "to the pixels masked in the output file. The default"+
                        " value is 65636", type = int)
    parser.add_argument("-out", "--out_file", help ="Name of the output file",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the output will be written.")
    args = parser.parse_args()
    print args
    
    
    #Set of the instances and the output parameter
    img_to_mask = Raster(args.im_file)
    mask_img = Raster(args.mask)
    out_filename = os.path.join(args.dir, args.out_file)
    
    #Execution of the method
    img_to_mask.apply_mask( mask_img, args.in_mask_value , out_filename,
                   out_mask_value = args.out_mask_value)
    
    print "The mask has been applied succesfully"
    