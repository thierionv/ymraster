# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 17:17:41 2015

@author: sig
"""


from ymraster import *

num = 'A'

#--------------------Fusion of the mutlispectral and panchromatic image

pan_file = '../../Donnees/Donnes_supp/Spot6_Pan_31072013.tif'
xs_file = '../../Donnees/Donnes_supp/Spot6_MS_31072013.tif'
output_fusion = 'data_example_seg/fusion_' + num +'.tif'

spot_xs = Raster(xs_file)
spot_pan = Raster(pan_file)

fus_img = spot_xs.fusion(spot_pan,output_fusion)

#--------------------Writting of the ndvi image

output_ndvi = 'data_example_seg/ndvi_' + num +'.tif'

ndvi_img = fus_img.ndvi(output_ndvi,idx_red = 2, idx_nir = 3)

#--------------------Extracting the blue band

idx = 0 
output_rmv = 'data_example_seg/rmv_' + num +'.tif'

rmv_img = fus_img.remove_band(idx, output_rmv)


#--------------------Concatenate the rmv_img and the ndvi_img

list_im = [ndvi_img]
output_concat = 'data_example_seg/concat_' + num +'.tif'

concat_img = rmv_img.concatenate( list_im, output_concat)

#--------------------Perform a LSMS


spatialr = 5
ranger = 500
maxiter = 10
thres = 0.1
rangeramp = 0

minsize = 50


output_filtered_image = "data_example_seg/lsms_filtered_sr_" + str(spatialr)\
+ "_rg_" + str(ranger) + "_max_" + str(maxiter) + "_rga_" + str(rangeramp) \
+ "_ms_" + str(minsize) + "_" + num + ".tif"
output_spatial_image = "data_example_seg/lsms_spatial_" + num + ".tif"
output_seg_image = "data_example_seg/lsms_seg_" + num + ".tif"
output_merged = "data_example_seg/lsms_merged_" + num + ".tif"
output_vector = "data_example_seg/lsms_vector_" + num + ".shp"

concat_img.lsms(spatialr, ranger, maxiter, thres, rangeramp, 
                output_filtered_image, output_spatial_image, 
                output_seg_image, output_merged, minsize, output_vector)
