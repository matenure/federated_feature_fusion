# This is a simple baseline for metr-la.npz. It treats each example as
# a 207-variate 12-step time series and classifies it by using a
# sktime multivariate classification method.
#
# It takes about 20 minutes to run. Example output:
#
# Done ndarray2dataframe. Time 47.34 seconds.
# Done cross validation. Time 1167.72 seconds. Accuracy 0.76 +/- 0.01

__author__ = "Jie Chen"

from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.classification.dictionary_based import BOSSEnsemble
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import time

def ndarray2dataframe(x):
    df = pd.DataFrame()
    instance_list = []
    for j in range(x.shape[1]):
        instance_list.append([])
    for j in range(x.shape[1]):
        for i in range(x.shape[0]):
            instance_list[j].append(pd.Series(x[i,j,:,0]))
    for j in range(x.shape[1]):
        df[j] = instance_list[j]
    return df

start = time.time()
dat = np.load('metr-la.npz')
X = ndarray2dataframe(dat['x']) # needed by sktime
y = dat['y']
end = time.time()
print('Done ndarray2dataframe. Time {:.2f} seconds.'.format(end-start))

start = time.time()
classifier = ColumnEnsembleClassifier(estimators=[
    ("TSF0", TimeSeriesForestClassifier(n_estimators=100), [0]),
    ("BOSSEnsemble3", BOSSEnsemble(max_ensemble_size=5), [3]),
])
scores = cross_val_score(classifier, X, y, cv=5)
end = time.time()
print('Done cross validation. Time {:.2f} seconds. Accuracy {:.2f} +/- {:.2f}'
      .format(end-start, scores.mean(), scores.std()))
