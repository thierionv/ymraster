# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:54:42 2015

@author: 
"""
import scipy as sp
from osgeo import gdal


def get_samples_from_roi(in_raster_label,in_roi,in_stat_raster ):
    '''
    The function get the set of pixel given the thematic map. Both map should be of same size.
    Input:
        raster_name: the name of the raster file, could be any file that GDAL can open
        roi_name: the name of the thematic image: each pixel whose values is greater than 0 is returned
    Output:
        X: the sample matrix. A nXd matrix, where n is the number of referenced pixels and d is the number of variables. Each 
            line of the matrix is a pixel.
        Y: the label of the pixel
    ''' 
    
    ## Open data
    stat = gdal.Open(in_stat_raster,gdal.GA_ReadOnly)
    if stat is None:
        print 'Impossible to open '+ in_stat_raster
        exit()
        
    roi = gdal.Open(in_roi,gdal.GA_ReadOnly)
    if roi is None:
        print 'Impossible to open '+ in_roi
        exit()
        
    label = gdal.Open(in_raster_label, gdal.GA_ReadOnly)
    if label is None:
        print 'Impossible to open '+ in_raster_label
        exit()
    
    if not((stat.RasterXSize == roi.RasterXSize) and (stat.RasterYSize ==
    roi.RasterYSize) and (stat.RasterXSize == label.RasterXSize) and 
    (stat.RasterYSize == label.RasterYSize) ):
        print 'Images should be of the same size'
        exit()
            
    ## Get the number of variables
    d  = stat.RasterCount
    
    ## load the ROI array
    ROI = roi.GetRasterBand(1).ReadAsArray()
    t = (ROI == 0).nonzero()
    print "ROI :", ROI
    ##load the label array and set nan where the objects don't correspond to
    #samples
    LABEL = label.GetRasterBand(1).ReadAsArray()
    LABEL[t] = -99
    print "LABEL :", LABEL    
    ##get the indices of one pixel per sample
    l, l_ind  = sp.unique(LABEL,return_index = True)
    l_ind = l_ind[1:len(l_ind)] 
    print 'l :', l
    print 'l_ind :', l_ind
    nb_samp = len(l_ind)
    print "nb_samp: ", nb_samp
    col = LABEL.shape[1]
    print "col: ", col
    indices = [sp.empty(nb_samp),sp.empty(nb_samp)]    
    indices[0] = [l_ind // col]
    indices[1] = [l_ind % col]
    print "indices:",indices
    ##set the Y array
    Y = ROI[indices].reshape((nb_samp,1))
    print "Y : ",Y
    del ROI
    roi = None    
    
    ##set the X array
    try:
        X = sp.empty((nb_samp,d))
        
    except MemoryError:
        print 'Impossible to allocate memory: roi too big'
        exit()
    
    for i in range(d):
        temp = stat.GetRasterBand(i+1).ReadAsArray()
        X[:,i] = temp[indices]   
    stat = None
    print "X : ", X
    del temp, indices, d, nb_samp, col, t, l, l_ind 
    
    return X,Y 