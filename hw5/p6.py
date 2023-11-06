import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import (LinearRegression, LassoCV, Lasso)
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.linear_model as skl
import sklearn.model_selection as skm
from sklearn.metrics import mean_squared_error

#######################################################################################################################
#p6 start
# load and clean data
prostate = pd.read_table("prostate.data", index_col=0)
prostate.head()

train = prostate[prostate['train']=='T'].iloc[:, 0:9]
test = prostate[prostate['train']=='F'].iloc[:, 0:9]
trainx = train.iloc[:, 0:8]
trainy = train.iloc[:, 8]
testx = test.iloc[:, 0:8]
testy = test.iloc[:, 8]
print(train.shape, test.shape, testx.shape, testy.shape)

# part 6(a)
ols6a = LinearRegression()
ols6afit = ols6a.fit(trainx, trainy)
ols6a_coef = ols6afit.coef_
# get fitting coefficients using sklearn
print(ols6a_coef)
# [ 0.57654319  0.61402    -0.01900102  0.14484808  0.73720864 -0.20632423 -0.02950288  0.00946516]

# now let's use statsmodels
trainxaug = sm.add_constant(trainx)
testxaug = sm.add_constant(testx)
ols6asm = sm.OLS(trainy, trainxaug)
ols6asmres = ols6asm.fit()



def build_next_model(current_model, feature_choices, trainx, trainy):
    best_feature = 0
    best_r_squre = 0
    for i, f in enumerate(feature_choices):
        # do ols and get R^2
        trainx_new = trainx[np.array([*current_model, f], dtype=object)]
        trainx_new_aug = sm.add_constant(trainx_new)
        ols_model = sm.OLS(trainy, trainx_new_aug)
        ols_res = ols_model.fit()
        ols_res_rsquared = ols_res.rsquared
        if ols_res_rsquared > best_r_squre:
            best_r_squre = ols_res.rsquared
            best_feature = f
    new_model = np.array([*current_model, best_feature], dtype=object)
    return new_model




current_model = np.array([])
p = trainx.shape[1]
features = np.array(trainx.columns)
# models = np.empty_like(p)
models = np.empty(p, dtype=object)
models[:] = [[] for _ in range(p)]
aics = np.zeros(p)
bics = np.zeros(p)
trainerrs = np.zeros(p)
testerrs = np.zeros(p)
# coef_lst = np.empty_like(p)
coef_lst = np.empty(p, dtype=object)
coef_lst[:] = [[] for _ in range(p)]
# build up all the models with sizes from 1 to p
for k in np.arange(p):
    # feature_choices = np.array([f for f in features if f not in current_model])
    # https://stackoverflow.com/questions/41125909/find-elements-in-one-list-that-are-not-in-the-other
    # feature_choices = np.array(list(set(features) - set(current_model)))
    feature_choices = np.setdiff1d(features, current_model)
    current_model = build_next_model(current_model=current_model, feature_choices=feature_choices,
                                 trainx=trainx, trainy=trainy)
    models[k] = current_model
    trainx_new_main = trainx[current_model]
    trainx_new_main_aug = sm.add_constant(trainx_new_main)
    ols_model_main = sm.OLS(trainy, trainx_new_main_aug)
    ols_res_main = ols_model_main.fit()
    coef_lst[k] = np.array(ols_res_main.params)
    aics[k] = ols_res_main.aic
    bics[k] = ols_res_main.bic
    trainerrs[k] = ols_res_main.ssr/ols_res_main.nobs
    # test error
    testx_new_main = testx[current_model]
    testx_new_main_aug = sm.add_constant(testx_new_main)
    testy_pred = ols_res_main.predict(testx_new_main_aug)
    testerrs[k] =  mean_squared_error(testy_pred, testy)

