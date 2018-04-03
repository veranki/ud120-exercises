#!/usr/bin/python
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import cross_validation
from sklearn import metrics

# Task 1: Select what features you'll use.
# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
data_dict.pop('TOTAL', 0)


def computeFraction(poi_messages, all_messages):
    """ given a number messages to/from POI (numerator)
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    if poi_messages == "NaN":
        fraction = 0
    else:
        fraction = float(poi_messages) / all_messages
    return fraction
submit_dict = {}
for name in data_dict:
    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)

    data_point["fraction_from_poi"] = fraction_from_poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)

    submit_dict[name] = {"from_poi_to_this_person": fraction_from_poi,
                         "from_this_person_to_poi": fraction_to_poi}
    data_point["fraction_to_poi"] = fraction_to_poi


def submitDict():
    return submit_dict
#####################
data_dict.pop('from_messages', 0)
data_dict.pop('to_messages', 0)
data_dict.pop('from_poi_to_this_person', 0)
data_dict.pop('from_this_person_to_poi', 0)
my_dataset = data_dict

# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'fraction_from_poi', 'fraction_to_poi', 'shared_receipt_with_poi']  # You will need to use more features


# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest
select = SelectKBest(k=10)
features_red = select.fit_transform(features, labels)

# Task 4: Try a varity of classifiers
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
steps = [('feature_selection', select), ('ada_boost', clf)]
pipeline = Pipeline(steps)

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42)

# fit your pipeline on features_train and labels_train
pipeline.fit(features_train, labels_train)
# call pipeline.predict() on your features_test data to make a set of test
# predictions
labels_prediction = pipeline.predict(features_test)
# test your predictions using sklearn.classification_report()
report = metrics.classification_report(labels_test, labels_prediction)
# and print the report
print(report)

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
import sklearn.grid_search


parameters = dict(feature_selection__k=[5, 10, 15],
                  ada_boost__n_estimators=[50, 100, 200])

cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid=parameters)

cv.fit(features_train, labels_train)
labels_predictions = cv.predict(features_test)
report = sklearn.metrics.classification_report(labels_test, labels_predictions)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
dump_classifier_and_data(clf, my_dataset, features_list)
