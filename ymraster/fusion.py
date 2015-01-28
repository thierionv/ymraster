# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:40:55 2015

@author: sig
"""

if __name__ == "__main__":
    """Executable of the fusion part
    """
    
    import argparse
    from ymraster import *
    
    #Set of the parse arguments
    parser = argparse.ArgumentParser(description= "Write the merge result " +  
    "between the two images of a bundle, using the BundleToPerfectSensor" + 
    "OTB application" + "\n" +
    "Example : python fusion.py ../../Donnees/Donnes_supp/" +
    "Spot6_Pan_31072013.tif ../../Donnees/Donnes_supp/Spot6_MS_31072013.tif" +
    " -pref A -dir data_example_seg")
    parser.add_argument("--pan_file", "-pan", help="Path of the panchromatic "+
                        "image", required = True)
    parser.add_argument("--xs_file","-xs", help="Path of the multi-spectral " +
                        "image", required = True)
    parser.add_argument("-out", "--out_file", help ="Name of the output file",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the output will be written. The \"/\"" +
                        " or \"\\\" have to be added at the end.")
    args = parser.parse_args()
    print args

    
    
    #set of the instances and the output name    
    spot_xs = Raster(args.xs_file)
    spot_pan = Raster(args.pan_file)
    output_fusion = args.dir + args.out_file 
    
    #Execution of the method
    fus_img = spot_xs.fusion(spot_pan,output_fusion)
    print "Fusion has been realized succesfully\n" 
    


