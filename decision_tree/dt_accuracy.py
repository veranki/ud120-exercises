#!/usr/bin/env python2
import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

##########################################################################

########################## DECISION TREE #################################

from sklearn.tree import DecisionTreeClassifier
##
clf_2 = DecisionTreeClassifier(min_samples_split=2)
pred_2 = clf_2.fit(features_train, labels_train)
acc_min_samples_split_2 = clf_2.score(features_test, labels_test)

clf_50 = DecisionTreeClassifier(min_samples_split=50)
pred_50 = clf_50.fit(features_train, labels_train)
acc_min_samples_split_50 = clf_50.score(features_test, labels_test)


def submitAccuracies():
    return {"acc_min_samples_split_2": round(acc_min_samples_split_2, 3),
            "acc_min_samples_split_50": round(acc_min_samples_split_50, 3)}
