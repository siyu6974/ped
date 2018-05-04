#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 01:14:43 2018

@author: laurent
"""
from sklearn.externals import joblib 
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

symbols_to_keep = dir()


z_scaler = StandardScaler()
pca = PCA(svd_solver='randomized', n_components=350, whiten=True, random_state=233)
svc = svm.SVC(class_weight='balanced', gamma=0.01, kernel='rbf', C=80)
model = make_pipeline(z_scaler, pca, svc)

s = cross_val_score(model,X_train,Y_train,n_jobs=4,cv=8)
print("Accuracy: %0.3f (+/- %0.3f)" % (s.mean(), s.std() * 2))

model.fit(X_train, Y_train)

joblib.dump(model, 'm_model.pkl')  


# garbage collector
symbols_to_keep += ['model']
internal_vars = ['name', 'internal_vars','symbols_to_keep','In', 'Out', 'get_ipython', 'exit', 'get_ipython', 'quit']
#
for name in dir():
    if not name.startswith('_') and name not in internal_vars:
        if name not in symbols_to_keep:
            del globals()[name]
del symbols_to_keep
del internal_vars
del name