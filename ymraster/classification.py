# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 14:54:42 2015

@author:
"""
import numpy as np
from osgeo import gdal
from ymraster import Raster, write_file
from raster_dtype import RasterDataType
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report,\
                             accuracy_score

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
            
            Y: the classes of the samples in a vertical n matrix. 
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
    l, l_ind  = np.unique(LABEL,return_index = True)
    #Delete the first id, corresponding to -99
    l_ind = l_ind[1:len(l_ind)]
    nb_samp = len(l_ind)
    col = LABEL.shape[1]
    #Get the index of each sample in the non-flattened original array
    indices = [np.empty(nb_samp),np.empty(nb_samp)]
    indices[0] = [l_ind // col]#the rows
    indices[1] = [l_ind % col]#the columns

    ##set the Y array, ie taking the classes values of each sample
    Y = ROI[indices].reshape((nb_samp,1))
    del ROI
    roi = None

    ##set the X array, ie taking all the statistic features for each sample
    try:
        X = np.empty((nb_samp,d))

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
    l, l_ind, reverse  = np.unique(LABEL,return_index = True,
                                   return_inverse = True)
    nb_samp = len(l_ind)

    #Get the index of each sample in the non-flattened original array
    col = LABEL.shape[1]
    indices = [np.empty(nb_samp),np.empty(nb_samp)]
    indices[0] = [l_ind // col]#the rows
    indices[1] = [l_ind % col]#the columns

    ##set the X array, ie taking all the statistic features for each label
    try:
        X = np.empty((nb_samp,d))

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

def decision_tree(X_train, Y_train, X_test, X_img, reverse_array, raster, 
                  out_filename, ext = 'Gtiff' ):
    """
    :param X_train: The sample-features matrix used to train the model, a n*d 
                    array where n is the number of referenced samples and d is 
                    the number of features.
    :param Y_train: The classes of the samples in a vertical n matrix.
    :param X_test: The sample-features matrix corresponding to the validation 
                    dataset on which the model is applied, a n*d array where n 
                    is the number of referenced samples and d is the number of 
                    features.
    :param X_img: The sample-features matrix corresponding to the wole image
                    on which the model is applied, a n*d array where n is the 
                    number of referenced samples and d is the number of 
                    features.
    :param reverse_array:The reverse matrix use to rebuild into the origin
                        dimension the result of the classification. This matrix
                        is supposed to be computed previously (cf. 
                        get_samples_from_label_img() #TODO)
    :param raster: The raster object that contains all the meta-data that should
                    be set on the classification image written, eg : it could be
                    the raster object of the labelled image.
    :param out_filename: Name of the classification image to be written.
    :param ext: Format of the output image to be written. Any formats
                supported by GDAL. The default value is 'Gtiff'.
    :returns:
            y_predict: The classes predicted from the validation dataset, in a 
                        vertical n matrix. It is useful to compute prediction 
                        error metrics.
    """
    #Get some parameters    
    rows = raster.meta['height']
    col = raster.meta['width']   
    
    #Set the DecisionTreeClassifier
    clf = tree.DecisionTreeClassifier()
    
    #Train the model
    clf = clf.fit(X_train, Y_train)
    
    #Perform the prediction on the whole label image
    classif = clf.predict(X_img)
    
    #Perform the prediction on the test sample
    y_predict = clf.predict(X_test)
    
    #Rebuild the image from the classif flat array with the given reverse array
    classif = classif[reverse_array]
    classif = classif.reshape(rows,col)
    
    #write the file
    meta = raster.meta
    meta['driver'] = gdal.GetDriverByName(ext)
    meta['dtype'] = RasterDataType(numpy_dtype = np.uint32)
    meta['count'] = None
    write_file(out_filename, overwrite=True, array=classif, **meta)
                 
    return y_predict

def pred_error_metrics(Y_predict, Y_test, target_names = None):
    """This function calcul the main classification metrics and compute and 
    display confusion matrix.
    
    :param Y_predict: Vertical n matrix correspind to the estimated targets as
                    returned by a classifier.
    :param Y_test: Vertical n matrix corresponding to the ground truth 
                    (correct) target values.
    :param target_names: List of string that contains the names of the classes.
                        Default value is None. 
    
    :returns:
            cm : Array of the confusion matrix.
            report : Text summary of the precision, recall, F1 score for each 
                    class.
            accuracy: float, If normalize == True, return the correctly 
                        classified samples (float), else it returns the number
                        of correctly classified samples (int).
    """
    
    
    #Compute the main classification metrics 
    report = classification_report(Y_test, Y_predict, target_names = target_names)
    accuracy = accuracy_score(Y_test, Y_predict)
    
    #Compute the confusion matrix
    cm = confusion_matrix(Y_test, Y_predict)
        
    # Show confusion matrix in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    return cm, report, accuracy

