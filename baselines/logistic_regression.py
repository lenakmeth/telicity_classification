#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 17:57:53 2021

@author: lena
"""

from utils import *
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

label_marker = 'duration'
max_len = 128

X_train, y_train = make_sets('train', label_marker, max_len)
X_val, y_val = make_sets('val', label_marker, max_len)
X_test, y_test = make_sets('test', label_marker, max_len)

# Logistic regression baseline with scaled data
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)

# score = pipe.score(X_val, y_val)
# print(score)

val_preds = pipe.predict(X_val)
test_preds = pipe.predict(X_test)

# classification report 
print('\nValidation:')
print(classification_report(y_val, val_preds))

print('Test:')
print(classification_report(y_test, test_preds))