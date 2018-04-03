#!/usr/bin/python

from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
print reg.coef_
print reg.intercept_


reg.score(ages_train, net_worths_train)
reg.score(ages_test, net_worths_test)
