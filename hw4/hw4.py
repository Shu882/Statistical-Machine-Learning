import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNC
from dataGeneration import *
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import pandas as pd

###############################################################
# problem 5: start
# zip code data

# load and clean data
zip_train = pd.read_csv("./zip.train", delimiter=' ', header=None)
zip_train = zip_train.drop(axis=1, columns=257, inplace=False)
zip_test = pd.read_csv("./zip.test", delimiter=' ', header=None)

zip_train = np.array(zip_train)
zip_test = np.array(zip_test)

x_train = zip_train[:, 1:]
y_train = zip_train[:, 0]

x_test = zip_test[:, 1:]
y_test = zip_test[:, 0]

#(a)
# pick the correct data for the 5 classes
y_labels = [0, 3, 5, 6, 9]
mask_train = np.isin(zip_train[:, 0], y_labels)
mask_test = np.isin(zip_test[:, 0], y_labels)
zip_train_a = zip_train[mask_train]
zip_test_a = zip_test[mask_test]
x_train_a = zip_train_a[:, 1:]
y_train_a = zip_train_a[:, 0]
x_test_a = zip_test_a[:, 1:]
y_test_a = zip_test_a[:, 0]

# LDA
model_lda_a = LDA()
model_lda_a.fit(x_train_a, y_train_a)
lda_train_err_a = 1-model_lda_a.score(x_train_a, y_train_a)
lda_test_err_a = 1-model_lda_a.score(x_test_a, y_test_a)
print(lda_train_err_a)
print(lda_test_err_a)
# 0.022335844994617826
# 0.06201550387596899

# KNN
ks5 = np.array([1, 3, 5, 7, 15])
knn_train_err_a = np.zeros(len(ks5))
knn_test_err_a = np.zeros(len(ks5))

for i, k in enumerate(ks5):
    model_knn_a = KNC(n_neighbors=k)
    model_knn_a.fit(x_train, y_train)
    knn_train_err_a[i] = 1 - model_knn_a.score(x_train_a, y_train_a)
    knn_test_err_a[i] = 1 - model_knn_a.score(x_test_a, y_test_a)

print(knn_train_err_a)
print(knn_test_err_a)
# [0.         0.01157158 0.0164155  0.01991389 0.02556512]
# [0.04360465 0.04748062 0.04748062 0.04554264 0.05523256]

# (b)
# LDA
model_lda_b = LDA()
model_lda_b.fit(x_train, y_train)
lda_train_err_b = 1-model_lda_b.score(x_train, y_train)
lda_test_err_b = 1-model_lda_b.score(x_test, y_test)
print(lda_train_err_b)
print(lda_test_err_b)
# 0.061994239473323276
# 0.11459890383657201

# KNN
ks5 = np.array([1, 3, 5, 7, 15])
knn_train_err_b = np.zeros(len(ks5))
knn_test_err_b = np.zeros(len(ks5))

for i, k in enumerate(ks5):
    model_knn_b = KNC(n_neighbors=k)
    model_knn_b.fit(x_train, y_train)
    knn_train_err_b[i] = 1 - model_knn_b.score(x_train, y_train)
    knn_test_err_b[i] = 1 - model_knn_b.score(x_test, y_test)

print(knn_train_err_b)
print(knn_test_err_b)
# [0.         0.01330407 0.02084762 0.02578521 0.03716911]
# [0.05630294 0.05530643 0.05530643 0.05829596 0.06975585]


###############################################################
# problem 6: start

# (a) function LDA5cv()
def LDA5cv(x, y, random_state):
    """
    # I could have implemented these myself.
    # But to get familiar with the sklearn APIs, I chose to use sklearn
    # x,y are predictors and targets
    # return: 5-fold CV error of lda
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    estimator = LDA()
    return (1 - cross_val_score(estimator=estimator, X=x, y=y, cv=cv)).mean()

# (b)
s1_train_x, s1_train_y = scenario1(size=100, seed=2023)
s1_test_x, s1_test_y = scenario1(size=500, seed=2024)
print(LDA5cv(s1_train_x, s1_train_y, random_state=1))
# 0.3

#(c) do (b) 20 times with different seeds
n_times = 20
rng = np.random.default_rng(2023)
seeds = rng.integers(low=1, high=3000, size=n_times, endpoint=True)

errs = np.zeros(n_times)
for i, seed in enumerate(seeds):
    errs[i] = LDA5cv(s1_train_x, s1_train_y, random_state=seed)

print(errs)
# [0.285 0.27  0.305 0.275 0.275 0.29  0.28  0.27  0.28  0.27  0.28  0.285
#  0.28  0.285 0.28  0.295 0.28  0.265 0.28  0.28 ]
fig6, ax6 = plt.subplots()
ax6.hist(errs)
ax6.set_xlabel('CV errors')
ax6.set_ylabel('counts')
fig6.show()
fig6.savefig('p6LDA5CVHist.png')

# (d)
model6 = LDA()
model6_fit = model6.fit(s1_train_x, s1_train_y)
print(1-model6.score(s1_test_x, s1_test_y))
# 0.237


###############################################################
# problem 7: start
# (a) logLDA5cv() function
def logLDA5cv(x, y, random_state):
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    lda_err = (1 - cross_val_score(estimator=LDA(), X=x, y=y, cv=cv)).mean()
    log_err = (1 - cross_val_score(estimator=LR(), X=x, y=y, cv=cv)).mean()
    return lda_err, log_err

#(b)
seed7 = 7
print(logLDA5cv(s1_train_x, s1_train_y, random_state=seed7))
# (0.285, 0.2899999999999999)
# conclusion: lda gives slightly lower 5-fold CV error

###############################################################
# problem 8: start
# (a) function KNN5cv()
def KNN5cv(x, y, ks, random_state):
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    errs = np.zeros(len(ks))
    ses = np.zeros(len(ks))
    # for each k, calculate the 5fold CV error and standard error
    for i, k in enumerate(ks):
        neigh = KNC(n_neighbors=k)
        cv_errors = 1 - cross_val_score(estimator=neigh, X=x, y=y, cv=cv)
        errs[i] = cv_errors.mean()
        ses[i] = cv_errors.std()
        # debugging
        # print("i: ", i)
        # print("k: ", k)
        # print("err: ", errs[i])
        # print("se: ", ses[i])
    return errs, ses

# (b)
n_train = 100 # actually half train size
n_test = 500
s2_train_x, s2_train_y = scenario2(size=n_train, seed=2000)
s2_test_x, s2_test_y = scenario2(size=500, seed=2014)

ks = np.arange(start=1, stop=n_train+1, step=1)
errs, ses = KNN5cv(s2_train_x, s2_train_y, ks, random_state=8)

fig8, ax8 = plt.subplots()
ax8.errorbar(ks, errs, yerr=ses, marker='o', markersize=3, linestyle='dotted', c='b', label='5-fold CV')
ax8.set_xlabel('k')
ax8.set_ylabel("CV errors")

print(np.argmin(errs))
# 14
# err min at k=14+1=15

# (c)
errs_test = np.zeros(len(ks))
for i, k in enumerate(ks):
    neigh = KNC(n_neighbors=k)
    neigh.fit(s2_train_x, s2_train_y)
    errs_test[i] = 1-neigh.score(s2_test_x, s2_test_y)
ax8.plot(ks, errs_test, marker='s', markersize=3, linestyle='dotted', c="g", label="Test")
ax8.legend()
fig8.show()
fig8.savefig('p8KNN5CV.png')