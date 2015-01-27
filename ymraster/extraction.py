# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 11:19:31 2015

@author: sig
"""

if __name__ == "__main__":
    """Executable of the extraction part
    """
    
    import argparse
    from ymraster import *
    
    #Set of the parse arguments
    parser = argparse.ArgumentParser(description= "Write a new image with the"+
                                    " band at the given index removed\n ")
    parser.add_argument("--xs_file", help="Path of the multi-spectral image.",
                        required = True)
    parser.add_argument("--idx", help="Chanel number of the band to be removed"
                        + ". Indexation starts at 1.",required = True, 
                        type = int)    
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
    output_rmv = args.dir_file + args.prefixe + 'extracted.tif'    
    print args    

    #set of the instance    
    spot_xs = Raster(args.xs_file)
    
    #Execution of the method
    rmv_img = spot_xs.remove_band(args.idx, output_rmv)
    
