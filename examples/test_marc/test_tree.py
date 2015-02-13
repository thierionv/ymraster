# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:13:31 2015

@author: sig
"""
from ymraster import Raster, write_file, get_samples_from_roi
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot #TODO install the module


rst_file = "data/new_set2/Spot6_MS_31072013_masked.tif"
rst = Raster(rst_file)
rst_array = rst.array()

roi_file =""
label_file = ""
stat_file = ""

statX_array, statY_array = get_samples_from_roi(label_file,roi_file,
                                                stat_file )

clf = tree.DecisionTreeClassifier()
clf = clf.fit(statX_array, statY_array)
classif = clf.predict(rst_array)

 
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("data/new_set2/mon_arbre.pdf")
 
