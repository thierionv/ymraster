# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 16:56:29 2015

@author: sig
"""

if __name__ == "__main__":
    """
    """
    import argparse
    import os
    from ymraster import *

    #Set of the parse arguments
    desc ="Calcul statistics of the labels from a label image and raster." +\
            " The statistics calculated by default are : mean, standard " +\
            "deviation, min, max and the 20, 40, 50, 60, 80th percentiles. " +\
            "The output is an image at the given format that contains n_band"+\
            " * n_stat_features bands. This method uses the GDAL et NUMPY " +\
            "library."
    parser = argparse.ArgumentParser(description= desc)
    parser.add_argument("--label_file", "-lab", help="Path of the label image"
                        ,required = True)
    parser.add_argument("--ms_file","-ms", help="Path of the multi spectral" +
                        " image on wich the statistics are calculated"
                        ,required = True)        
    parser.add_argument("-out", "--out_file", help ="Name of the output file",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the output will be written.")
    parser.add_argument("--stats","-s", default = ["mean","std","min","max",
                        "per"], choices = ["mean","std","min","max","per"],
                        help = "List of the statistics to be calculated. By " +
                        "default, all the features are calculated, i.e. mean,"+
                        "std, min, max and per.", nargs = '+')
    parser.add_argument("--percentile","-per", default = [20,40,50,60,80],
                        type = int, help = "List of the percentile to be " +
                        "calculated. By default,the percentiles are 20, 40, " +
                        "50, 60, 80.", nargs = '+')
    parser.add_argument("--format","-f", help = " Format in wich the output "+
                        "image is written. Any formats supported by GDAL"
                        , default = "Gtiff")
    args = parser.parse_args()    
    print "\n"
    print args
    print "\n"
    
    #set of the instances and the output parameter
    label_img = Raster(args.label_file)
    ms_img = Raster(args.ms_file)
    out_filename = os.path.join(args.dir, args.out_file)
    
    label_img.get_stat(ms_img,out_filename, stats = args.stats, percentile = \
                        args.percentile, ext = args.format)
    
    print "Obtention of statistics has been realized succesfully\n"
    
    