#!/usr/bin/python

"""
This is the code to accompany the Lesson 2 (SVM) mini-project.

Use a SVM to identify emails from the Enron corpus by their authors:
Sara has label 0
Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here 4###
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

##
from sklearn.svm import SVC
t0 = time()
classifier = SVC(C=10000, kernel="rbf")
our_pred = classifier.fit(features_train, labels_train)
print "training time is ", round((time() - t0), 3), "s"

##
t1 = time()
our_pred = classifier.predict(features_test)
print "prediction time is ", round(time() - t1, 3), "s"

##
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, our_pred)
print accuracy
print classifier.score(features_test, labels_test)
print our_pred[10]
print our_pred[26]
print our_pred[50]

##
from collections import Counter
print Counter(our_pred)
from scipy.stats import itemfreq
print itemfreq(our_pred)
#########################################################
