import numpy as np
import pandas as pd
import sys
import datetime
from datetime import datetime
from src.Prep_Class import * 

# REGRESSION

from fireTS.models import NARX
from math import sqrt
from scipy import signal
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, LassoLars
from sklearn.metrics import r2_score, mean_squared_error, make_scorer, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline,  make_pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression import *
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller, acf, arma_order_select_ic, pacf_ols, pacf
from statsmodels.tsa.holtwinters import *

#VISUALIZATION 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6


# =============================================================================
# REGRESSION PREP
# =============================================================================

# === TRAIN TEST =========================================

def train_test_lag(df, lag_len=4, Xy=True):
    '''
    ==Function==
    Creates a new df with 3 lagged columns
    Splits data into X_train, y_train, X_test, y_test sets
        -OR- into train and test sets 
    
    ==Parameters==
    |lag_len| : include how many lags to include in data
                default is 4
    |xy|      : if True, splits data into X & y sets and again into train & test sets
                if False, splits data into train & test sets
                 default = True
            
    ==Returns==
    |X_train| = first 80% of df's lagged data 
    |y_train| = first 80% of y values
    |X_test|  = last 20% of df's lagged data
    |y_test|  = last 20% of df's y values
    -OR-
    |train| = first 80% of df's data 
    |test| = last 20% of df's data
    '''
    lag_df = (pd.concat([df.shift(i) for i in range(lag_len+1)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, lag_len+1)])).dropna() 
    idx = round(len(lag_df)*.8)
    if Xy==False:
        train, test = lag_df[:idx], lag_df[idx:]
        return train, test
    elif Xy==True:
        lag_y, lag_X  = lag_df.y, lag_df.drop(columns='y')    
        X_train, y_train, X_test, y_test = lag_X[:idx], lag_y[:idx], lag_X[idx:], lag_y[idx:]
        return X_train, y_train, X_test, y_test
    
# =============================================================================
# PRIMARY REGRESSION MODELS 
# =============================================================================

# === Multiple Regressions =========================================    
def multiple_regressors(df, lag_len=4, print_mses=True):
    '''
    ==Function==
    Uses train_test_xy(df) to split into (X/y)train/(X/y)test sets
    
    Runs data through 
    - Random Forest Regressor
    - Linear Regression
    - Bagging Regressor
    - AdaBoost Regressor
    - sm OLS Linear Regression
    - smf ols Regression
   
   if print_mses==True:
        ==Prints==
        MSE scores for each 
        default=True
    
    ==Returns==
    rf, lr, br, abr, ols_lin, ols models
    '''
    X_train, y_train, X_test, y_test = train_test_lag(df, lag_len=lag_len)
    rf= RandomForestRegressor(n_jobs=-1).fit(X_train,y_train).predict(X_test)
    ols_lin = sm.OLS(y_train, X_train).fit().predict(X_test)  
    ols_train, ols_test= train_test_lag(df, lag_len=lag_len, Xy=False)
    ols_str = 'y ~ '
    for i in range(1, lag_len+1):
        ols_str+= ' Lag{} +'.format(i)
    ols_str = ols_str.rstrip(' +')
    ols = smf.ols(ols_str, data=ols_train).fit().predict(ols_test)
    if print_mses == True:
        print(' ---- MSE Scores ----'.center(31))
        print('Random Forest Regressor  ', round(mean_squared_error(y_test, rf),5))
        print('sm OLS Linear            ', round(mean_squared_error(y_test, ols_lin),5))

        print('smf ols                  ', round(mean_squared_error(ols_test.y,ols),5))
        return rf, ols_lin, ols
        # removed br, abr
    else:
        return rf, ols_lin, ols 

# =============================================================================
# MODEL REGRESSION PLOTS AND PREP 
# =============================================================================
    
    
def regres_dfs(df):
    '''
    ==Function==
    Creates a new df with y_test values and forecasted values for all regression models
    
    ==Uses== 
    |train_test_lag| from Regression_Helper_Funcs
    |multiple_regressors| from Regression_Helper_Funcs
    
    ==Returns==
    |y_preds| : new df
    '''
    
    y_preds = train_test_lag(df, Xy=True)[3]
    rf, ols_lin, ols_smf = multiple_regressors(df, print_mses=False)
    y_preds.rename(columns={'cost_per_watt':'actual'}, inplace=True)
    y_preds['randomforest'] = rf
#     y_preds['linear'] = lr
#     y_preds['bagging'] = br
#     y_preds['adaboost'] = abr
    y_preds['olslinear'] = ols_lin
    y_preds['olssmf'] = ols_smf
    return y_preds

def regression(df):    
    '''
    completes all prep and outputs regression results
    returns df and stationary df
    '''
    #df, diff = prep()
    y_preds = regres_dfs(df)
    y_train = train_test_lag(df, Xy=True)[1]
    fig, axs = plt.subplots(3, figsize= (20,15), constrained_layout=True)
    pred_s, pred_e = y_preds.index.date[0], y_preds.index.date[-1]
    train_s, train_e = y_train.index.date[0], y_train.index.date[-1]
    
    axs[0].plot(y_preds.actual, label= 'Actual')
    axs[0].plot(y_preds.randomforest, label= 'Random Forest', linewidth=2)
    axs[0].plot(y_train[-30:], label='Train', color='gray')
    axs[0].set_title('Random Forest \n  MSE= {}'.format(round(mean_squared_error(y_preds.actual, y_preds.randomforest),5)), fontsize=18)
    axs[0].legend(loc='best')
    axs[0].set_xlim(left= y_train.index.date[-31])
    fig.suptitle(' Regression Models \n Forecast For:     [{}] - [{}] \n Trained On:       [{}] - [{}]\n '.format(pred_s, pred_e, train_s, train_e), fontsize=20)
    
    axs[1].plot(y_preds.actual, label= 'Actual')
    axs[1].plot(y_preds.olslinear, label= 'OLS Linear', linewidth=2)
    axs[1].plot(y_train[-30:], label='Train',color='gray')
    axs[1].set_title('OLS Linear \n MSE = {}'.format(round(mean_squared_error(y_preds.actual, y_preds.olslinear),5)), fontsize=18)
    axs[1].legend(loc='best')
    axs[1].set_xlim(left= y_train.index.date[-31])
    
    axs[2].plot(y_preds.actual, label= 'Actual', alpha=.75)
    axs[2].plot(y_preds.olssmf, label= 'OLS', linewidth=2)
    axs[2].plot(y_train[-30:], label='Train',color='gray')
    axs[2].set_title('OLS smf \n MSE= {}'.format(round(mean_squared_error(y_preds.actual, y_preds.olssmf),5)), fontsize=18)
    axs[2].legend(loc='best')  
    axs[2].set_xlim(left= y_train.index.date[-31])
    plt.show() 
    #return df, diff
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# =============================================================================
# OTHER REGRESSION MODELS
# =============================================================================     
# === ROBUST LINEAR  =========================================
def rob_lin(df):
    '''
    ==Function==
    robust linear model
    ==Returns==
    |roblin_model| : model.fit()
    |roblin_trend| : model.fit().predict()
    '''
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    roblin_model = sm.RLM(y,X, ).fit()
    roblin_trend = roblin_model.predict()
    return roblin_model, roblin_trend 
 

# === OLS and GLS =========================================    
def least_squares(df):
    
    '''     
    ==Function==
    Difference Least Squares Regression Models
    Uses OLS and GLS the average between the two
    
    ==Returns==
    |lst_sq_mods| : table with predictions from each model
    
    ==Note==
    Results seem to be nearly identical
    
    '''
    y = df
    X = add_constant(np.arange(1, len(y) + 1))
    lst_sq_mods = pd.DataFrame()
    lst_sq_mods['OLS'] = sm.OLS(y, X).fit().predict(X)
    lst_sq_mods['GLS'] = sm.GLS(y, X).fit().predict(X)
    lst_sq_mods['avg'] = linear_plots.mean(axis=1)
    lst_sq_mods['orig'] = np.array(y.cost_per_watt)
    return lst_sq_mods


# === LINEAR MODEL RESIDUALS =========================================
def lm_resids(df, linear_trend):    
    '''
    ==Function==
    Linear Model Residuals
    Takes in df and linear trend
    '''
    lm_residuals = pd.Series(df.cost_per_watt - linear_trend, index=df.index)
    fig, axs = plt.subplots(3, figsize=(16, 8))
    # The model forecasts zero for the first few datapoints, so the residuals
    # are the actual values.
    axs[0].plot(lm_residuals.index, lm_residuals)
    plot_acf_and_pacf(lm_residuals, axs[1:])
    plt.tight_layout()

    
# === HOLT LINEAR REGRESSION =========================================    
def holt_linear_model(df):
    '''
    ==Function==
    Holt's Linear Regression Model
    
    ==Returns==
    RMS score 
    '''
    idx = round(len(df) * .8)
    train = df[:idx]
    test = df[idx:]
    fit1 = Holt(np.asarray(train['cost_per_watt'])).fit()
    test['forecast'] = fit1.forecast(len(test))
#     mse = mean_squared_error(test['cost_per_watt'],test['forecast'])
    model_plot(test['cost_per_watt'], train, test['forecast'], 'Holt Linear')
    
#     plt.figure(figsize=(16,8))
#     plt.plot(train['cost_per_watt'], label='Train')
#     plt.plot(test['cost_per_watt'], label='Test')
#     plt.plot(test['Holt_linear'], label='Holt_linear')
#     plt.legend(loc='best')
#     mse = mean_squared_error(test['cost_per_watt'],test['Holt_linear'])
#     plt.title('MSE = {}'.format(round(mse,5)))
#     plt.show()
    
    
# === RANDOM FOREST GRID SEARCH =========================================    
def rf_gs(df): 
    '''
    ==Function==
    Random Forest Grid Search
    - Specifies hyperparameter space
    - Creates a hyperparameter grid.
    - Creates train and test sets
    - Creates GridSearchCV object
    - Fits to the training set
    - Computes and prints the metrics
    
    ==Returns==
    Best Parameters
    Test R Squared
    Grid Search results    
    '''
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    rf = RandomForestRegressor(oob_score=True,n_jobs=-1)
    rf.fit(X,y)
    thePipe = Pipeline([('RFR', RandomForestRegressor())])
    thePipe.get_params()
    # Specify the hyperparameter space.
    num_estimators_space = np.array(range(5, 25, 5))
    max_depth_space = np.array(range(5, 25, 5))
    # Create the hyperparameter grid.
    param_grid = {'RFR__n_estimators': num_estimators_space,
              'RFR__max_depth': max_depth_space}
    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Create the GridSearchCV object: gm_cv
    gs_cv = GridSearchCV(thePipe, param_grid, cv=3, return_train_score=True, verbose=2)
    # Fit to the training set
    gs_cv.fit(X_train, y_train)  
    # Compute and print the metrics
    theR2 = gs_cv.score(X_test, y_test)
    print("Best parameters: {}".format(gs_cv.best_params_))
    print("test R squared: {}".format(theR2))    
    return gs_cv.cv_results_


# === NARX RANDOM FOREST GRID SEARCH =========================================
from fireTS.models import NARX

def narx_rf(df):
    
    x = df
    y = df

    mdl = NARX(RandomForestRegressor(), auto_order=2, exog_order=[2], exog_delay=[1])
    para_grid = {'n_estimators': [10, 30, 100]}
    mdl.grid_search(x, y, para_grid, verbose=2)

    # Best hyper-parameters are set after grid search, print the model to see the difference
    print(mdl)
    mdl.fit(x, y)
    ypred = mdl.predict(x, y, step=3)
    return ypred

# === DIRECTAUTOREGRESSOR OF RF =========================================
from fireTS.models import DirectAutoRegressor

def dir_autoreg(df):
    x = df
    y = df
    mdl = DirectAutoRegressor(RandomForestRegressor(), 
                              auto_order=2, 
                              exog_order=[2], 
                              exog_delay=[1], 
                              pred_step=3)
    mdl.fit(x, y)
    ypred = mdl.predict(x, y)
    return ypred

# === NARX =========================================

def narx(df):
    x = df
    y = df
    mdl = NARX(RandomForestRegressor(), auto_order=2, exog_order=[2], exog_delay=[1])
    mdl.fit(x, y)
    ypred = mdl.predict(x, y, step=3)
    return ypred

# =============================================================================
# COEFFICIENTS & COVARIANCE
# =============================================================================

# === COEFFICIENTS =========================================
def model_coefs_params(model_list):
    [print('Params for {} \n {}'.format(i, i.params)) for i in model_list]   
    
# === COVARIANCE =========================================    
        
def cov_table(df):
    '''
    ==Function==
    Estimate a covariance matrix
    
    ==Input Suggestion==
    Can Enter in lag_cost from ols
    '''
    plt.figure(figsize=(20,12))
    plt.plot(np.cov(df))
    plt.show()
    return np.cov(df)



