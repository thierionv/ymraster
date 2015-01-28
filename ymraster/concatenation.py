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
                                    "the concatenation of the given rasters " +
                                    "in order.All bands in all input rasters" +
                                    " must have same size.Moreover, if data " +
                                    "types are different, then everything " +
                                    "will be converted to the default data " +
                                    "type in OTB (_float_ currently).") 
    parser.add_argument("--im_files", "-im", nargs = '+',  help="list of the" +
                        "images paths.", required = True)
    parser.add_argument("-out", "--out_file", help ="Name of the output file",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the output will be written. The \"/\"" +
                        " or \"\\\" have to be added at the end.")
    args = parser.parse_args()
    print args
    
    output_concat = args.dir + args.out_file 
    #set of the instances    
    rasters = [Raster(im) for im in args.im_files]    
    
    #Execution of the method
    concatenate_images(rasters, output_concat)
    print "Concatenation has been realized succesfully\n"
    