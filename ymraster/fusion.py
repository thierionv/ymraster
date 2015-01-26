# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:40:55 2015

@author: sig
"""

if __name__ == "__main__":

    import argparse
    from ymraster import *
    
    parser = argparse.ArgumentParser(description= "Write the merge result" +  
    "between the two images of a bundle, using the BundleToPerfectSensor" + 
    "OTB application")
    
    parser.add_argument("pan_file", help="Path of the panchromatic image")
    parser.add_argument("xs_file", help="Path of the multi-spectral image")
    parser.add_argument("-dir","--dir_file", default = "", help = "" + 
    "Path of the folder where the output will be written" )
    args = parser.parse_args()
    
    if args.dir_file:
        output_fusion = args.dir_file + '/fusionABC.tif'
    else:
        output_fusion = 'fusionABC.tif'
    
    spot_xs = Raster(args.xs_file)
    spot_pan = Raster(args.pan_file)
    
    fus_img = spot_xs.fusion(spot_pan,output_fusion)
    



