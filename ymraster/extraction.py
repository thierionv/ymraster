# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 11:19:31 2015

@author: sig
"""

if __name__ == "__main__":
    """Executable of the extraction part
    """
    
    import argparse
    import os
    from ymraster import *
    
    #Set of the parse arguments
    parser = argparse.ArgumentParser(description= "Write a new image with the"+
                                    " band at the given index removed\n ")
    parser.add_argument("--xs_file","-xs", help="Path of the multi-spectral " +
                        "image", required = True)
    parser.add_argument("--idx", "-idx", help="Chanel number of the band to " +
                        "be removed. Indexation starts at 1.",required = True, 
                        type = int)  
    parser.add_argument("-out", "--out_file", help ="Name of the output file",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the output will be written.")
    args = parser.parse_args()
    print args
    
    
    #set of the instance and the output file name   
    spot_xs = Raster(args.xs_file)
    output_rmv = os.path.join(args.dir, args.out_file)
    
    #Execution of the method
    rmv_img = spot_xs.remove_band(args.idx, output_rmv)
    print "Extraction has been realized succesfully\n"
