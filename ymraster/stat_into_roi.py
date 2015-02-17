# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:54:42 2015

@author: 
"""
import scipy as sp
from osgeo import gdal


def get_samples_from_roi(in_rst_label,in_rst_roi,in_rst_stat ):
    '''
    The function, thanks to a label image, picks the index of one pixel per 
    sample in a sample raster. Then it takes for each sample the statistic
    features present in a "statistic raster" and casts it in a 2d matrix. The
    three rasters should be of the same size and the samples should exactly 
    match with the segmentation objects of the label image. The input rasters 
    could be any file that GDAL can open.
    
    :param in_rst_label: name of the label image, supposedly created previously 
                        during a segmentation.
    :param in_rst_roi: name of the sample raster, all the samples should
                        correspond to a object in the label image
    :param in_rst_stat: name of the statistic features raster of the
                        segmentation objects
    :returns: 
            X: the sample matrix. A nXd matrix, where n is the number of 
            referenced samples and d is the number of features. Each line of 
            the matrix is sample.
            
            Y: the classe of the samples in a vertical n matrix. 
    ''' 
    
    ## Open data
    stat = gdal.Open(in_rst_stat,gdal.GA_ReadOnly)
    if stat is None:
        print 'Impossible to open '+ in_rst_stat
        exit()
        
    roi = gdal.Open(in_rst_roi,gdal.GA_ReadOnly)
    if roi is None:
        print 'Impossible to open '+ in_rst_roi
        exit()
        
    label = gdal.Open(in_rst_label, gdal.GA_ReadOnly)
    if label is None:
        print 'Impossible to open '+ in_rst_label
        exit()
    ##Test the size
    if not((stat.RasterXSize == roi.RasterXSize) and (stat.RasterYSize ==
    roi.RasterYSize) and (stat.RasterXSize == label.RasterXSize) and 
    (stat.RasterYSize == label.RasterYSize) ):
        print 'Images should be of the same size'
        exit()
            
    ## Get the number of features
    d  = stat.RasterCount
    
    ## load the ROI array
    ROI = roi.GetRasterBand(1).ReadAsArray()
    t = (ROI == 0).nonzero()
    
    ##load the label array and set negative value where the objects don't 
    #correspond to samples
    LABEL = label.GetRasterBand(1).ReadAsArray()
    LABEL[t] = -99   
    
    ##get the indices of one pixel per sample
    #sort the label by their id and get the indices of the first occurrences 
    #of the unique values in the (flattened) original array (LABEL)
    l, l_ind  = sp.unique(LABEL,return_index = True)
    #Delete the first id, corresponding to -99
    l_ind = l_ind[1:len(l_ind)]
    nb_samp = len(l_ind)
    col = LABEL.shape[1]
    #Get the index of each sample in the non-flattened original array
    indices = [sp.empty(nb_samp),sp.empty(nb_samp)]    
    indices[0] = [l_ind // col]#the rows
    indices[1] = [l_ind % col]#the columns
    
    ##set the Y array, ie taking the classes values of each sample
    Y = ROI[indices].reshape((nb_samp,1))
    del ROI
    roi = None    
    
    ##set the X array, ie taking all the statistic features for each sample
    try:
        X = sp.empty((nb_samp,d))
        
    except MemoryError:
        print 'Impossible to allocate memory: roi too big'
        exit()
    
    for i in range(d):
        temp = stat.GetRasterBand(i+1).ReadAsArray()
        X[:,i] = temp[indices]   
    
    #close the files and release memory    
    stat = None
    del temp, indices, d, nb_samp, col, t, l, l_ind 
    
    return X,Y

def get_samples_from_label_img(in_rst_label, in_rst_stat):
    """
    The function, given a label and statistic image, compute in a 2d array the 
    feature per label.The two input rasters should be of the same size. The 
    input rasters could be any file that GDAL can open. The function also 
    compute a reverse matrix that can permit to rebuild an image from the 
    result of an object classification
    
    :param in_rst_label: name of the label image, supposedly created previously 
                        during a segmentation.
    :param in_rst_stat: name of the statistic features raster of the
                        segmentation objects
    :returns: 
            X: the sample matrix. A nXd matrix, where n is the number of 
            label and d is the number of features. Each line of 
            the matrix is label.
            
            reverse: everse matrix that can permit to rebuild an image from the 
            result of an object classification. 
    """
    ## Open data
    stat = gdal.Open(in_rst_stat,gdal.GA_ReadOnly)
    if stat is None:
        print 'Impossible to open '+ in_rst_stat
        exit()
        
    label = gdal.Open(in_rst_label, gdal.GA_ReadOnly)
    if label is None:
        print 'Impossible to open '+ in_rst_label
        exit()
    
    ##Test the size
    if not((stat.RasterXSize == label.RasterXSize) and 
    (stat.RasterYSize == label.RasterYSize) ):
        print 'Images should be of the same size'
        exit()
    
    ## Get the number of features
    d  = stat.RasterCount
    
    ##load the label array 
    LABEL = label.GetRasterBand(1).ReadAsArray()
    
    ##get the indices of one pixel per label
    #sort the label by their id and get the indices of the first occurrences 
    #of the unique values in the (flattened) original array (LABEL). Compute
    #also the reverse matrix that can permit to rebuild the original array. 
    l, l_ind, reverse  = sp.unique(LABEL,return_index = True, 
                                   return_inverse = True)
    nb_samp = len(l_ind)
    
    #Get the index of each sample in the non-flattened original array
    col = LABEL.shape[1]
    indices = [sp.empty(nb_samp),sp.empty(nb_samp)]    
    indices[0] = [l_ind // col]#the rows
    indices[1] = [l_ind % col]#the columns
    
    ##set the X array, ie taking all the statistic features for each label
    try:
        X = sp.empty((nb_samp,d))
        
    except MemoryError:
        print 'Impossible to allocate memory: label image too big'
        exit()
    
    for i in range(d):
        temp = stat.GetRasterBand(i+1).ReadAsArray()
        X[:,i] = temp[indices]   
    
    #close the files and release memory    
    stat = None
    del temp, indices, d, nb_samp, col, l, l_ind 
    
    return X, reverse
    