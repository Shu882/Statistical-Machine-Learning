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
#p4 start
#  part (e)
beta_ols = np.array([1.1, -0.8, 0.3, -0.1]).reshape(4,1)
lamdas = np.arange(start=1e-3, stop=3, step=0.01)
beta_ridge = beta_ols/((1+lamdas).reshape(1,len(lamdas)))
dfs = 4/(1+lamdas) # X is orthonormal therefore, all singular values are 1, df equations follows ESL eqn3.50
fig4, ax4 = plt.subplots()
ax4.plot(dfs, beta_ridge[0, :], 'r*')
ax4.plot(dfs, beta_ridge[1, :], 'g^')
ax4.plot(dfs, beta_ridge[2, :], 'bd')
ax4.plot(dfs, beta_ridge[3, :], 'cs')
ax4.set_xlabel('df($\lambda$)')
ax4.set_ylabel('Coefficients')
# fig4.savefig('p4eSolutionPath.pdf')
fig4.show()

# part (f)
def betaLasso(beta, lam):
    results = np.zeros(len(beta))
    for i, ele in enumerate(beta):
        res = np.sign(ele) * np.max([0,  np.abs(ele)-lam/2])
        results[i] = res
    return results

print(betaLasso([1.1, -0.8, 0.3, -0.1], 1))
# [ 0.6 -0.3  0.  -0. ]
print(betaLasso([1.1, -0.8, 0.3, -0.1], 0.4))
# [ 0.9 -0.6  0.1 -0. ]

betalasso = np.zeros((len(lamdas), len(beta_ols)))
for i, lam in enumerate(lamdas):
    betalasso[i,] = betaLasso([1.1, -0.8, 0.3, -0.1], lam).reshape(4)
betalasso.shape

fig4f, ax4f = plt.subplots()
ax4f.plot(lamdas, betalasso[:, 0], 'r*')
ax4f.plot(lamdas, betalasso[:, 1], 'g^')
ax4f.plot(lamdas, betalasso[:, 2], 'bd')
ax4f.plot(lamdas, betalasso[:, 3], 'cs')
ax4f.set_xlabel('$\lambda$')
ax4f.set_ylabel('Coefficients')
# fig4f.savefig('p4fSolutionPath.pdf')
fig4f.show()

fig4f2, ax4f2 = plt.subplots()
negloglamdas = -np.log(lamdas)
ax4f2.plot(negloglamdas, betalasso[:, 0], 'r*')
ax4f2.plot(negloglamdas, betalasso[:, 1], 'g^')
ax4f2.plot(negloglamdas, betalasso[:, 2], 'bd')
ax4f2.plot(negloglamdas, betalasso[:, 3], 'cs')
ax4f2.set_xlabel('$-\log(\lambda)$')
ax4f2.set_ylabel('Coefficients')
# ax4f2.legend(loc='upper left')
# fig4f2.savefig('p4fSolutionPath2.pdf')
fig4f2.show()

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
print(ols6asmres.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                   lpsa   R-squared:                       0.694
# Model:                            OLS   Adj. R-squared:                  0.652
# Method:                 Least Squares   F-statistic:                     16.47
# Date:                Sat, 04 Nov 2023   Prob (F-statistic):           2.04e-12
# Time:                        12:51:14   Log-Likelihood:                -67.505
# No. Observations:                  67   AIC:                             153.0
# Df Residuals:                      58   BIC:                             172.9
# Df Model:                           8
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# const          0.4292      1.554      0.276      0.783      -2.681       3.539
# lcavol         0.5765      0.107      5.366      0.000       0.361       0.792
# lweight        0.6140      0.223      2.751      0.008       0.167       1.061
# age           -0.0190      0.014     -1.396      0.168      -0.046       0.008
# lbph           0.1448      0.070      2.056      0.044       0.004       0.286
# svi            0.7372      0.299      2.469      0.017       0.140       1.335
# lcp           -0.2063      0.111     -1.867      0.067      -0.428       0.015
# gleason       -0.0295      0.201     -0.147      0.884      -0.432       0.373
# pgg45          0.0095      0.005      1.738      0.088      -0.001       0.020
# ==============================================================================
# Omnibus:                        0.825   Durbin-Watson:                   1.690
# Prob(Omnibus):                  0.662   Jarque-Bera (JB):                0.389
# Skew:                          -0.164   Prob(JB):                        0.823
# Kurtosis:                       3.178   Cond. No.                     1.29e+03
# ==============================================================================
#
# Notes:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
# [2] The condition number is large, 1.29e+03. This might indicate that there are
# strong multicollinearity or other numerical problems.
# save_summary_to_pdf(ols6asmres.summary(), 'summary.pdf')
# save_summary_to_pdf2(ols6asmres.summary(), 'summary.pdf')

# 6(b) and (c)
def build_next_model(current_model, feature_choices, trainx, trainy):
    """
    A single step in forward stepwise selection: choose the best feature among feature choices and add to the current model
    best_feature: a feature with each the model gives highest R^2
    """
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
models = np.empty(p, dtype=object)
models[:] = [[] for _ in range(p)]
aics = np.zeros(p)
bics = np.zeros(p)
trainerrs = np.zeros(p)
testerrs = np.zeros(p)
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

#######################################################################################################################
#p7 start
# 7(a)
scaler = StandardScaler(with_mean=True, with_std=True)
lassocv = LassoCV(alphas=None,cv=5, max_iter=100000)
lassocv.fit(scaler.fit_transform(trainx), trainy)
lasso = Lasso(max_iter=10000)
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(scaler.fit_transform(trainx), trainy)
# print the regularization parameter, train error and test error
print(lassocv.alpha_*len(trainy))
print(mean_squared_error(trainy, lasso.predict(scaler.fit_transform(trainx))))
print(mean_squared_error(testy, lasso.predict(scaler.fit_transform(testx))))
# 0.05888498771532305
# 0.43921972590782493
# 0.5496649751374661

# get the coefficient
lasso_coef = lasso.coef_/scaler.scale_
print(lasso_coef)
# [ 0.69467126  0.98080288 -0.02020694  0.14745558  0.76414961 -0.2022805 -0.02275003  0.01073235]
# make the coefficients pretty
coef_df = pd.DataFrame({"Feature": trainx.columns, "Coefficients": lasso_coef})
coef_df
#    Feature  Coefficients
# 0   lcavol      0.694671
# 1  lweight      0.980803
# 2      age     -0.020207
# 3     lbph      0.147456
# 4      svi      0.764150
# 5      lcp     -0.202280
# 6  gleason     -0.022750
# 7    pgg45      0.010732

# 7(b)
lassocvmse = lassocv.mse_path_
mses_means = lassocvmse.mean(axis=1)
mses_stds = lassocvmse.std(axis=1)
fig7b, ax7b = plt.subplots()
stop = 100
ax7b.errorbar(lassocv.alphas_[0:stop], mses_means[0:stop], yerr=mses_stds[0:stop],
              marker='o', markersize=3, linestyle='dotted', c='b', label='5-fold CV')
ax7b.set_xlabel(r'$\alpha$')
ax7b.set_ylabel("CV errors")
# find the min argument
mseminarg = mses_means.argmin()
# find the one std values
high = mses_means[mseminarg] + mses_stds[mseminarg]
low = mses_means[mseminarg] - mses_stds[mseminarg]
ax7b.plot(lassocv.alphas_[0:stop], high*np.ones(stop), label='One ')
ax7b.plot(lassocv.alphas_[0:stop], low*np.ones(stop))
fig7b.show()
fig7b.savefig('p7bCVErrors1.png')

# https://stackoverflow.com/questions/9868653/find-first-sequence-item-that-matches-a-criterion
# implementation the one sigma rule
ind, mse_ind = next((i, m) for i,m in enumerate(mses_means) if m < high)
print(ind, mse_ind)
alpha_1sigma = lassocv.alphas_[ind]
print(alpha_1sigma*len(trainy))
# 0.2877937122825021 19.282178722927643

lasso_1sigma = Lasso(max_iter=10000)
lasso_1sigma.set_params(alpha=alpha_1sigma)
lasso_1sigma.fit(scaler.fit_transform(trainx), trainy)
# print the regularization parameter, train error and test error
print(mean_squared_error(trainy, lasso_1sigma.predict(scaler.fit_transform(trainx))))
print(mean_squared_error(testy, lasso_1sigma.predict(scaler.fit_transform(testx))))
# 0.6693309767427765
# 0.4878611670193412

# get the coefficient
lasso_1sigma_coef = lasso_1sigma.coef_/scaler.scale_
coef_df_1sigma = pd.DataFrame({"Feature": trainx.columns, "Coefficients": lasso_1sigma_coef})
coef_df_1sigma
#    Feature  Coefficients
# 0   lcavol      0.516378
# 1  lweight      0.431915
# 2      age      0.000000
# 3     lbph      0.000000
# 4      svi      0.111441
# 5      lcp      0.000000
# 6  gleason      0.000000
# 7    pgg45      0.000000

#######################################################################################################################
#p8 start
# 8(a)
# use the OLS coefficients as the initial beta values
ols6a_coef = ols6afit.coef_
beta_init = np.abs(ols6a_coef)
#  basically follow p5
trainx_scaled = trainx/beta_init
testx_scaled = testx/beta_init

# no standardization
lassocv8a = LassoCV(alphas=None,cv=5, max_iter=100000)
lassocv8a.fit(trainx_scaled, trainy)
lasso8a = Lasso(max_iter=10000)
lasso8a.set_params(alpha=lassocv.alpha_)
lasso8a.fit(trainx_scaled, trainy)

print(lassocv8a.alpha_*len(trainy))
print(mean_squared_error(trainy, lasso8a.predict(trainx_scaled)))
print(mean_squared_error(testy, lasso8a.predict(testx_scaled)))

# get the coef back to the original scale
beta_star = lasso8a.coef_/beta_init
coef_df_8a = pd.DataFrame({"features": trainx.columns, "fitting_coef": lasso8a.coef_,
                           "beta_init": beta_init, "beta_star": beta_star})
coef_df_8a

#   features  fitting_coef  beta_init  beta_star
# 0   lcavol      0.332444   0.576543   0.576615
# 1  lweight      0.375487   0.614020   0.611522
# 2      age     -0.000360   0.019001  -0.018945
# 3     lbph      0.020993   0.144848   0.144932
# 4      svi      0.538418   0.737209   0.730347
# 5      lcp     -0.042243   0.206324  -0.204743
# 6  gleason     -0.000889   0.029503  -0.030127
# 7    pgg45      0.000090   0.009465   0.009472

# 8(b)
lassocv8amse = lassocv8a.mse_path_
mses_means = lassocv8amse.mean(axis=1)
mses_stds = lassocv8amse.std(axis=1)
fig8b, ax8b = plt.subplots()
stop = 100
ax8b.errorbar(lassocv8a.alphas_[0:stop], mses_means[0:stop], yerr=mses_stds[0:stop],
              marker='o', markersize=3, linestyle='dotted', c='b', label='5-fold CV')
ax8b.set_xlabel(r'$\alpha$')
ax8b.set_ylabel("CV errors")
# find the min argument
mseminarg = mses_means.argmin()
print(mseminarg)
# find the one std values
high = mses_means[mseminarg] + mses_stds[mseminarg]
low = mses_means[mseminarg] - mses_stds[mseminarg]
ax8b.plot(lassocv8a.alphas_[0:stop], high*np.ones(stop), label='One ')
ax8b.plot(lassocv8a.alphas_[0:stop], low*np.ones(stop))
fig8b.show()
fig8b.savefig('p8bCVErrors1.png')

# fit with largest tuning param
lasso8b_1sigma = Lasso(max_iter=10000)
lasso8b_1sigma.set_params(alpha=lassocv8a.alphas_.max())
lasso8b_1sigma.fit(trainx_scaled, trainy)
# print the regularization parameter, train error and test error
print(mean_squared_error(trainy, lasso8b_1sigma.predict(trainx_scaled)))
print(mean_squared_error(testy, lasso8b_1sigma.predict(testx_scaled)))
# 1.4370364928082315
# 1.0567332280603818

lasso8b_1sigma_coef = lasso8b_1sigma.coef_/beta_init
coef8b_df_1sigma = pd.DataFrame({"Feature": trainx.columns, "Coefficients": lasso8b_1sigma_coef})
coef8b_df_1sigma
#    Feature  Coefficients
# 0   lcavol  0.000000e+00
# 1  lweight  0.000000e+00
# 2      age  0.000000e+00
# 3     lbph  0.000000e+00
# 4      svi  0.000000e+00
# 5      lcp  0.000000e+00
# 6  gleason  0.000000e+00
# 7    pgg45  2.430618e-18