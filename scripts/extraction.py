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
    desc = "Writes a new raster (in the specified output file) which is the "+\
            "same than the current raster, except that the band(s) at the "+\
            "given index has been remove."
    parser = argparse.ArgumentParser(description= desc)
    parser.add_argument("--xs_file","-xs", help="Path of the multi-spectral " +
                        "image", required = True)
    parser.add_argument("--idx", "-idx", help=" List of the index of"+
                        " the band to be removed. Indexation starts at 1.",
                        required = True, type = int, nargs = '+') 
    parser.add_argument("-out", "--out_file", help ="Name of the output file",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the output will be written.")
    args = parser.parse_args()
    print "\n"
    print args,"\n"
    
    
    #set of the instance and the output file name   
    xs_img = Raster(args.xs_file)
    output_rmv = os.path.join(args.dir, args.out_file)
    
    #control the coherency of the arguments    
    d = xs_img.meta['count']
    if not args.idx :
        print "Warning : none index specified in --idx argument.\n"
    if not all ([(boo in range(1,d+1)) for boo in args.idx ]):
        print "Error : one of the index specified is out of range.\n"
        exit()
    if sorted(args.idx) == range(1,d+1):
        print "Error : you can not remove all the bands.\n"
        exit()    
    
    #Execution of the method
    rmv_img = xs_img.remove_band(args.idx, output_rmv)
    print "Extraction has been realized succesfully\n"
