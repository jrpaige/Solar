import numpy as np
import pandas as pd
import sys
import datetime
from datetime import datetime
from src.Plot import * 

# REGRESSION
from math import sqrt
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, LassoLars
from sklearn.metrics import r2_score, mean_squared_error, make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
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

def train_test_xy(df):
    '''
    ==Function==
    Creates 4 values: X_train, y_train, X_test, y_test   
    
    ==Returns==
    |X_train| = 2d array of [sample#, 1] = length of first 80% of data
    |y_train| = array of first 80% of data values
    |X_test|  = 2d array of [sample#, 1] = length of last 20% of data
    |y_test|  = arrat of last 20% of data values
    '''
    count_cons, count, idx, df_vals = [], 1, round(len(df)*.8), df.cost_per_watt.values
    for i in range(len(df)):
        count_cons.append((count, 1))
        count +=1
    X_train, y_train, X_test, y_test = count_cons[:idx], df_vals[:idx], count_cons[idx:],df_vals[idx:]
    return X_train, y_train, X_test, y_test


# =============================================================================
# PRIMARY REGRESSION MODELS 
# =============================================================================

# === Multiple Regressions =========================================    
def multiple_regressors(df):
    '''
    ==Function==
    Uses train_test_xy(df) to split into (X/y)train/(X/y)test sets
    
    Runs data through 
    - Random Forest Regressor
    - Linear Regression
    - Bagging Regressor
    - AdaBoost Regressor
   
    ==Prints==
    MSE scores for each 
    '''
    X_train, y_train, X_test, y_test = train_test_xy(df)
    rf_trend = RandomForestRegressor(n_jobs=-1).fit(X_train,y_train).predict(X_test)
    print(' ---- MSE Scores ----'.center(31))
    print('Random Forest Regressor  ', round(mean_squared_error(y_test, rf_trend),5))
    lr_trend = LinearRegression().fit(X_train, y_train).predict(X_test)
    print('Linear Regression        ', round(mean_squared_error(y_test, lr_trend),5))
    br_trend = BaggingRegressor().fit(X_train, y_train).predict(X_test)
    print('Bagging Regressor        ', round(mean_squared_error(y_test, br_trend),5))
    abr_trend = AdaBoostRegressor().fit(X_train, y_train).predict(X_test)
    print('AdaBoost Regressor       ', round(mean_squared_error(y_test, abr_trend),5))

        
# === NEW LAG OLS ========================================= 
def smf_ols(df):
    '''
    ==Function==
    uses smf.ols on data split into train test
    
    ==Returns==
    plot with MSE
    '''
    lag_df = (pd.concat([df.shift(i) for i in range(4)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, 4)])).dropna()    
    idx = round(len(lag_df)* .8)
    ols_train, ols_test= lag_df[:idx], lag_df[idx:]
    ols_predict = smf.ols('y ~ Lag1 + Lag2 + Lag3', data=ols_train).fit().predict(ols_test)
    print('smf ols                  ', round(mean_squared_error(ols_test.y,ols_predict),5))
    
    
# === NEW LAG LINEAR OLS ========================================= 
def sm_OLS(df):
    '''
    ==Function==
    linear OLS that uses sm.OLS on data split into X_train, y_train, X_test, y_test
    X = array [1-len(df)]
    y = df's values
    
    ==Returns==
    plot with MSE
    '''
    lag_df = (pd.concat([df.shift(i) for i in range(4)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, 4)])).dropna() 
    lag_df.index = np.arange(1,len(lag_df)+1)
    idx = round(len(lag_df)* .8)
    lag_y,lag_X = list(lag_df.values), list(lag_df.index)
    lag_y_train, lag_y_test,lag_X_train, lag_X_test  = lag_y[:idx], lag_y[idx:], lag_X[:idx], lag_X[idx:]
    predict = sm.OLS(lag_y_train, lag_X_train).fit().predict(lag_X_test)  
    print('sm OLS Linear            ', round(mean_squared_error(lag_y_test, predict),5))
    
    
    
    
    
# =============================================================================
# OTHER REGRESSION MODELS
# =============================================================================    

# === ROLLING OLS =========================================
def rolling_ols(df):  #not helpful
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    rolols_model = RollingOLS(y, X, window=3).fit()
    #rolols_trend = rolols_model.predict(X)
    return rolols_model
   
    
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
