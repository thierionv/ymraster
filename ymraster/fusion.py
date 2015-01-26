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
    parser.add_argument("pan_file", help="Path of the panchromatic image")
    parser.add_argument("xs_file", help="Path of the multi-spectral image")
    parser.add_argument("-pref", "--prefixe", help ="Prefixe to add to the " +
    "file to be written", default = "", type = str)
    parser.add_argument("-dir","--dir_file", default = "", help = "" + 
    "Path of the folder where the output will be written" )
    args = parser.parse_args()
    
    #Symbol to add in function of the optional parse arguments, to have a 
    #proper path
    if args.dir_file:
        args.dir_file += '/'
    if args.prefixe:
        args.prefixe += '_'
    output_fusion = args.dir_file + args.prefixe + 'fusion.tif'
    
    #set of the instances    
    spot_xs = Raster(args.xs_file)
    spot_pan = Raster(args.pan_file)
    
    #Execution of the method
    fus_img = spot_xs.fusion(spot_pan,output_fusion)
    
    


