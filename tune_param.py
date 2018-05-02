#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:58:49 2018

@author: Siyu Zhang
"""

runfile('read_images.py')

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import RandomizedPCA, PCA


clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X_train, Y_train,n_jobs=4, cv=5)
print("Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#
#pca = PCA().fit(X_train)
#plt.plot(np.cumsum(pca.explained_variance_ratio_))


pca = RandomizedPCA(n_components=300, whiten=True, random_state=23)
svc = svm.SVC(class_weight='balanced')
model = make_pipeline(pca,svc)

param_grid = {'svc__C':[0.0001, 0.001, 0.01, 0.1, 1, 10,100],
              'svc__gamma':[0.0001, 0.001, 0.01, 0.1, 1],
              'svc__kernel':['linear', 'rbf']}
grid = GridSearchCV(model, param_grid, n_jobs=8)
grid.fit(X_train, Y_train)
print(grid.best_params_)
print(grid.best_score_)
# svc__C=0.001, svc__gamma=0.0001, svc__kernel=linear
# score=0.88125
clf = grid.best_estimator_

#
#clf.fit(X_train, Y_train)
#
#joblib.dump(clf, 'm_model.pkl') 