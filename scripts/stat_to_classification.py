# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 15:14:42 2015

@author: sig
"""
import argparse
import os
from ymraster import classification as cla
from ymraster import Raster
from sklearn.cross_validation import train_test_split

def command_line_arguments():
    #Set of the parse arguments
    desc = "From a roi(ground truth), a labeled and a satistic raster, "+\
            "perform a decision tree classification, supply the main "+\
            "classification metrics, and display a confusion matrix. Before "+\
            "the very classification, ground truth samples are randomly split"+\
            " into a training dataset and a validation one. The proportion "+\
            "between the two of them can be specified."
    parser = argparse.ArgumentParser(description= desc)
    parser.add_argument("--roi_file", "-roi", help="Path of the roi raster.",
                        required = True)
    parser.add_argument("--label_file", "-lab", help="Path of the labeled "
                        "raster.", required = True)
    parser.add_argument("--stat_file", "-stat", help="Path of the statistic "
                        "raster. See get_stats.py help for further details.",
                        required = True)
    parser.add_argument("--format","-f", help = " Format in wich the output "+
                        "image is written. Any formats supported by GDAL. "+
                        " Default value is 'Gtiff'.", default = "Gtiff")
    parser.add_argument("--target_names","-tar", help = "List of string that "
                        "contains the names of the classes.Default value is "
                        "None.", default = None, nargs = '+')
    parser.add_argument("--train_size","-trs", help = "Should be between 0.0 "+
                        "and 1.0 and represent the proportion of the dataset "+
                        "used for training the classifier. Default value "+
                        "is 0.75.", default = 0.75, type = float)
    parser.add_argument("-out", "--out_file", help ="Name of the output file",
                        required = True, type = str)
    parser.add_argument("-d","--dir", default = "", help = "Path of the " +
                        "folder where the output will be written.")
    return parser.parse_args() 

def stat_to_classification(args):
    
    #Get the sample-feature matrix from the roi 
    X, Y = cla.get_samples_from_roi(args.label_file,args.roi_file,
                                    args.stat_file)
    
    #Get the sample-feature matrix from the labeled raster                                                 
    X_label, reverse = cla.get_samples_from_label_img(args.label_file,
                                                      args.stat_file )
    
    #Split the roi into two dataset 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,train_size =\
                                                        args.train_size)
    #set some parameters
    stat = Raster(args.stat_file)
    out_filename = os.path.join(args.dir,args.out_file)
    
    #Initialize the classifier, train it and perform it
    Y_predict = cla.decision_tree(X_train, Y_train, X_test, X_label, reverse, 
                                  stat,out_filename, ext = args.format)
    
    #Compute the classification metrics and the confusion matrix
    cm, report, accuracy = cla.pred_error_metrics(Y_predict, Y_test,
                                                   target_names = \
                                                   args.target_names)

    #Print the results
    print "confusion matrix\n", cm
    print report
    print "OA :", accuracy

def main():
    args = command_line_arguments()
    stat_to_classification(args)
    
if __name__ == "__main__":
    main()
   
    