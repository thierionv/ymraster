# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 13:57:57 2015

@author: sig
"""

import scipy as sp
from osgeo import gdal
from ymraster import *
from ymraster.dtype import RasterDataType
from ymraster import stat_into_roi

in_label = "data/concat_sample_lsms_merged.tif"
raster = Raster(in_label)
label = gdal.Open(in_label,gdal.GA_ReadOnly)
col = label.RasterXSize
line = label.RasterYSize
print col, line
sample = sp.zeros((line,col))
#print sample
t = [sp.empty(10),sp.empty(10)]
t[0] = [6,6,7,7,7,7,8,8,8,8]
t[1] = [2,3,0,1,2,3,0,1,2,3]

sample[t] = 77
t[0] = [1,1,1,2,2,2,2,3,3,3,4,4,4]
t[1] = [8,9,10,7,8,9,10,8,9,10,8,9,10]
sample[t] = 55
#print sample

proj = raster.meta['srs']
geotransform = label.GetGeoTransform()
driver = "Gtiff"
out_filename = "data/sample_sample.tif"

write_file(out_filename, overwrite=True, drivername= driver, dtype= RasterDataType(numpy_dtype = np.uint32) ,
                array=sample, width=None, height=None, depth=None, dt=None,
                srs= proj, transform=geotransform, xoffset=0, yoffset=0)
in_stat = "data/test_arg.tif"                
X,Y = get_samples_from_roi(in_label,out_filename,in_stat)
print X
print Y
