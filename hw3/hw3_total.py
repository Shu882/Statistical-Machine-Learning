from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from ISLP import confusion_table
import statsmodels.api as sm
from sklearn.metrics import (confusion_matrix, accuracy_score)


## generating data: train and test
train_size = 100
rng = np.random.default_rng(seed=2023)
train_green = rng.multivariate_normal(mean=(2, 1), cov=[[1, 0], [0, 1]], size=train_size)
train_red = rng.multivariate_normal(mean=(1, 2), cov=[[1, 0], [0, 1]], size=train_size)
# plot
fig0, ax0 = plt.subplots(2, 1, figsize=(4, 8))
ax0[0].scatter(train_green[:, 0], train_green[:, 1], c='r')
ax0[0].scatter(train_red[:, 0], train_red[:, 1], c='g')
fig0.show()

# put all the data points in a single plot
fig1, ax1 = plt.subplots()
ax1.scatter(train_green[:, 0], train_green[:, 1], c='r')
ax1.scatter(train_red[:, 0], train_red[:, 1], c='g')

test_size = 500
test_rng = np.random.default_rng(seed=2024)
test_green = rng.multivariate_normal(mean=(2, 1), cov=[[1, 0], [0, 1]], size=test_size)
test_red = rng.multivariate_normal(mean=(1, 2), cov=[[1, 0], [0, 1]], size=test_size)
fig2, ax2 = plt.subplots()
ax2.scatter(test_green[:, 0], test_green[:, 1], c='r')
ax2.scatter(test_red[:, 0], test_red[:, 1], c='g')

train_x = np.concatenate((train_green, train_red), axis=0)
train_y = np.concatenate((np.ones(train_size), (-1) * np.ones(train_size)), axis=0)
test_x = np.concatenate((test_green, test_red), axis=0)
test_y = np.concatenate((np.ones(test_size), (-1) * np.ones(test_size)), axis=0)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# p5: Linear Method for Classiﬁcation: Scenario 1
# a LDA
lda = LDA(store_covariance=True) # create lda classifier
lda.fit(train_x, train_y)
lda_test_pred = lda.predict(test_x)
lda_test_err = 1-np.mean(lda_test_pred==test_y) #misclassification rate
lda_train_pred = lda.predict(train_x)
lda_train_err = 1-np.mean(lda_train_pred==train_y) #misclassification rate

print("Train and test errors from LDA: ", lda_train_err, lda_test_err)

print("\nIgnore any output from here\n")
# (b) logistic regression
train_y_logit = train_y
train_y_logit[train_y_logit==-1]=0

test_y_logit = test_y
test_y_logit[test_y_logit==-1]=0

# 5b logistic regression
log_reg = sm.Logit(train_y_logit, train_x).fit()
print(log_reg.summary())
""" 
Optimization terminated successfully.
         Current function value: 0.503154
         Iterations 6
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                  200
Model:                          Logit   Df Residuals:                      198
Method:                           MLE   Df Model:                            1
Date:                Thu, 05 Oct 2023   Pseudo R-squ.:                  0.2741
Time:                        11:24:10   Log-Likelihood:                -100.63
converged:                       True   LL-Null:                       -138.63
Covariance Type:            nonrobust   LLR p-value:                 2.841e-18
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.9487      0.146      6.481      0.000       0.662       1.236
x2            -0.9383      0.147     -6.386      0.000      -1.226      -0.650
==============================================================================
"""

#convert the predicted probabilities to labels
log_pred_train = np.array(list(map(round, log_reg.predict())))
log_pred_test = np.array(list(map(round, log_reg.predict(test_x))))

log_train_err = 1-accuracy_score(train_y_logit, log_pred_train)
log_test_err = 1-accuracy_score(test_y_logit, log_pred_test)

log_train_cm = confusion_matrix(train_y_logit, log_pred_train)
log_test_cm = confusion_matrix(test_y_logit, log_pred_test)
print("Confusion matrix for train : \n", log_train_cm)
print("Confusion matrix for test : \n",log_test_cm)
print("Train and test errors from logistic regression: ", log_train_err, log_test_err)
"""
Confusion matrix for train : 
 [[76 24]
 [29 71]]
Confusion matrix for test : 
 [[379 121]
 [116 384]]
Train and test errors from logistic regression:  0.265 0.237
"""

###############################################################################
# problem 6
# (a)
seed_center = 16
rng = np.random.default_rng(seed=seed_center)
mean_mu = [1, 0]
mean_nu = [0, 1]
cov = np.array([[1, 0], [0, 1]])

# center for green and red classes
mu = rng.multivariate_normal(mean=mean_mu, cov=cov, size=10)
nu = rng.multivariate_normal(mean=mean_nu, cov=cov, size=10)

# define a function to generate data
def gendata2(n, mu1, mu2, sig1, sig2, myseed):
    rng2 = np.random.default_rng(seed=myseed)
    mean1 = rng2.choice(mu1, size=n, replace=True)
    mean2 = rng.choice(mu2, size=n, replace=True)
    green = np.zeros([n, 2])
    red = np.zeros([n, 2])
    for i in range(n):
        green[i, ] = rng2.multivariate_normal(mean=mean1[i, ], cov=sig1)
        red[i, ] = rng2.multivariate_normal(mean=mean2[i, ], cov=sig2)

    x = np.concatenate([green, red], axis=0)
    return x


# generate the training set
seed_train = 2000
ntrain = 100 # half sample size
train2 = gendata2(ntrain, mu, nu, cov/5, cov/5, seed_train)
ytrain = np.concatenate(([1]*ntrain, [0]*ntrain), axis=0)

# (b) plot the training set
fig6b, ax6b = plt.subplots()
ax6b.scatter(train2[:ntrain,0], train2[:ntrain,1], c='g', label="Green")
ax6b.scatter(train2[ntrain:,0], train2[ntrain:,1], c='r', label="Red")
fig6b.legend()
ax6b.set_xlabel('X1')
ax6b.set_ylabel('X2')
fig6b.show()

# (c) plot the training set
ntest = 500
seed_test = 2014
test2 = gendata2(ntest, mu, nu, cov/5, cov/5, seed_test)
ytest = np.concatenate(([1]*ntrain, [0]*ntrain), axis=0)

###############################################################
# problem 7
# (a) linear model
train2new = sm.add_constant(train2)
olsmod = sm.OLS(ytrain, train2new)
olsres = olsmod.fit()
print(olsres.summary())
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.302
Model:                            OLS   Adj. R-squared:                  0.295
Method:                 Least Squares   F-statistic:                     42.59
Date:                Thu, 05 Oct 2023   Prob (F-statistic):           4.26e-16
Time:                        15:33:41   Log-Likelihood:                -109.23
No. Observations:                 200   AIC:                             224.5
Df Residuals:                     197   BIC:                             234.4
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.5576      0.038     14.535      0.000       0.482       0.633
x1             0.1158      0.025      4.568      0.000       0.066       0.166
x2            -0.1966      0.029     -6.666      0.000      -0.255      -0.138
==============================================================================
Omnibus:                       17.910   Durbin-Watson:                   0.610
Prob(Omnibus):                  0.000   Jarque-Bera (JB):                6.133
Skew:                           0.001   Prob(JB):                       0.0466
Kurtosis:                       2.142   Cond. No.                         2.20
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

olsres.get_prediction(train2new)
ols_values_train = olsres.fittedvalues
ols_pred_train = np.zeros(2*ntrain)
ols_pred_train[ols_values_train>=0.5] = 1

test2new = sm.add_constant(test2)
ols_values_test = olsres.predict(test2new)
ols_pred_test = np.zeros(2*ntest)
ols_pred_test[ols_values_test>=0.5] = 1

# linear fitting errors
ols_train_err = 1-accuracy_score(ols_pred_train, ytrain)
ols_test_err = 1-accuracy_score(ols_pred_test, ytest)
print("Train and test errors from linear model: ", ols_train_err, ols_test_err)
# Train and test errors from linear model: 0.275 0.32199999999999995


# (b) LDA
lda = LDA(store_covariance=True) # create lda classifier
lda.fit(train2, ytrain)
lda_test_pred = lda.predict(test2)
lda_test_err = 1-np.mean(lda_test_pred==ytest) #misclassification rate

lda_train_pred = lda.predict(train2)
lda_train_err = 1-np.mean(lda_train_pred==ytrain) #misclassification rate
print("Train and test errors from LDA: ", lda_train_err, lda_test_err)
# Train and test errors from LDA:  0.275 0.32199999999999995

# (c) logistic regression
log_reg = sm.Logit(ytrain, train2).fit()
print(log_reg.summary())

#convert the predicted probabilities to labels
log_pred_train = np.array(list(map(round, log_reg.predict())))
log_pred_test = np.array(list(map(round, log_reg.predict(test2))))

log_train_err = 1-accuracy_score(ytrain, log_pred_train)
log_test_err = 1-accuracy_score(ytest, log_pred_test)

log_train_cm = confusion_matrix(ytrain, log_pred_train)
log_test_cm = confusion_matrix(ytest, log_pred_test)
print("Confusion matrix for train : \n", log_train_cm)
print("Confusion matrix for test : \n",log_test_cm)
print("Train and test errors from logistic regression: ", log_train_err, log_test_err)
# Train and test errors from logistic regression:  0.28500000000000003 0.33999999999999997


##########################################################################################
# p8
# (a) KNN Scenario 1
ks = [1, 4, 7, 10, 13, 16, 30, 45, 60, 80, 100, 150, 200]
nk = len(ks)
train_err = np.zeros(nk)
test_err = np.zeros(nk)


for i, k in enumerate(ks):
    neigh = KNeighborsClassifier(n_neighbors=k)

    train_pred = neigh.fit(X=train_x, y=train_y).predict(train_x)
    train_err[i] = 1 - np.mean(train_pred == train_y)

    test_pred = neigh.fit(X=train_x, y=train_y).predict(test_x)
    test_err[i] = 1 - np.mean(test_pred == test_y)
    # print(f'round: i = {i} and k = {k}')
    # print(f'train_err = {train_err}')
    # print(f'test_err = {test_err}')

dof = 2 * train_size / np.array(ks)

fig3, ax3 = plt.subplots()
ax3.semilogx(dof, train_err, c='r', label='Train', marker='o', linestyle='--')
ax3.semilogx(dof, test_err, c='g', label='Test', marker='^', linestyle='--')
ax3.legend()
ax3.set_xlabel('Degrees of Freedom - N/k')
ax3.set_ylabel('Misclassification curves')
fig3.show()


# (b) KNN Scenario 2
train_err2 = np.zeros(nk)
test_err2 = np.zeros(nk)

for i, k in enumerate(ks):
    neigh = KNeighborsClassifier(n_neighbors=k)
    train_pred = neigh.fit(X=train2, y=ytrain).predict(train2)
    train_err2[i] = 1 - np.mean(train_pred == ytrain)
    test_pred = neigh.fit(X=train2, y=ytrain).predict(test2)
    test_err2[i] = 1 - np.mean(test_pred == ytest)

dof2 = 2 * ntrain / np.array(ks)

fig4, ax4 = plt.subplots()
ax4.semilogx(dof2, train_err2, c='r', label='Train', marker='o', linestyle='--')
ax4.semilogx(dof2, test_err2, c='g', label='Test', marker='^', linestyle='--')
ax4.legend()
ax4.set_xlabel('Degrees of Freedom - N/k')
ax4.set_ylabel('Misclassification curves')
fig4.show()
