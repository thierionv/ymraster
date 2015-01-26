# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 13:24:55 2015

@author: sig
"""

if __name__ == "__main__":
    """Executable of the extraction part
    """
    
    import argparse
    from ymraster import *
    
    #Set of the parse arguments
    parser = argparse.ArgumentParser(description= "Write an image which is " +
    "the concatenation of the given rasters in order.\nAll bands in all input"+
    " rasters must have same size.\n\nMoreover, if data types are different, "+
    "then everything will be converted to the default data type in OTB " +
    "(_float_ currently).") 
    parser.add_argument("im_files",nargs = '+',  help="list of the images " +
    "paths.")
    
    parser.add_argument("-pref", "--prefixe", help ="Prefixe to add to the " +
    "file to be written", default = "", type = str)
    parser.add_argument("-dir","--dir_file", default = "", help = "Path of "+
    "the folder where the output will be written" )
    args = parser.parse_args()
    
    #Symbol to add in function of the optional parse arguments, to have a 
    #proper path
    if args.dir_file:
        args.dir_file += '/'
    if args.prefixe:
        args.prefixe += '_'
    output_concat = args.dir_file + args.prefixe + 'concatenated.tif'
    print args
    
    #set of the instances    
    rasters = [Raster(im) for im in args.im_files]    
    
    #Execution of the method
    concatenate_images(rasters, output_concat)
    
    