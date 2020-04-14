import numpy as np
import pandas as pd
import sys
import datetime
from datetime import datetime
from Solar.src.Plot import * 

# MATH
from math import sqrt
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline,  make_pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

#TIME
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression import *
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.tools import add_constant
from statsmodels.tsa import stattools
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.arima_model import *
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.holtwinters import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, arma_order_select_ic, pacf_ols, pacf
import pyramid
from pmdarima.arima import auto_arima
from sktime.forecasters import ARIMAForecaster
from sktime.highlevel.tasks import ForecastingTask
from sktime.highlevel.strategies import ForecastingStrategy
from sktime.highlevel.strategies import Forecasting2TSRReductionStrategy
from sktime.pipeline import Pipeline
from sktime.transformers.compose import Tabulariser

#VISUALIZATION 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6



# =============================================================================
# INITIAL REGRESSION MODELS 
# =============================================================================


# === TRAIN TEST =========================================
def reg_test_train(df, train_size=.8):
    '''
    ==Function==
    Basic manual 80/20 split
    
     ==Parameters==
    |train_size| : can input specific % of data to use for training set
                    default set to 80%
    
    ==Returns==
    |train| : % of data from beginning 
    |test| : remaining % of data until end
    '''
    idx = round(len(df)*train_size)
    train, test = df[:idx], df[idx:]
    return train, test

# === TIME SERIES SPLIT =========================================
def time_train_test_split():
    '''
    ==Function==
    Completes sklearn's TimeSeriesSplit using kfolds on data
    
    ==Returns==
    |train| : array of values 
    |test| : array of values
    '''
    tss = TimeSeriesSplit()
    dfn = df.values
    for strain_index, stest_index in tss.split(dfn):
        train, test = dfn[strain_index], dfn[stest_index]
        print('Observations: %d' % (len(train) + len(test)))
        print('Training Observations: %d' % (len(train)))
        print('Testing Observations: %d' % (len(test)))
        pyplot.plot(train)
        pyplot.plot([None for i in train] + [x for x in test])
        pyplot.show()
    return train, test

#ARMA(train,order).fit().predict(start=test.index.date[0],end=test.index.date[-1])


    
# === LAGGED OLS =========================================    
# need to add constant 
# need to split into train test

def lag_ols_model(df):    
    '''
    ==Function==
    Creates lag table and processes through OLS
    
    ==Returns==
    |ols_model| : ols of 3 lagged colummns]
    |ols_trend| : df of fitted values]
    '''
    train, test = reg_test_train(df)
    X_train = add_constant(np.arange(1, len(train) + 1))
    X_test = add_constant(np.arange(1, len(test) + 1))
    lag_cost_train = (pd.concat([X_train.shift(i) for i in range(4)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, 4)])).dropna()
    lag_cost_test = (pd.concat([X_test.shift(i) for i in range(4)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, 4)])).dropna()
    y = df 
    ols_model = smf.ols('y ~ Lag1 + Lag2 + Lag3', data=lag_cost_train).fit() 
    ols_trend = ols_model.fittedvalues
    return ols_model, ols_trend


# === LINEAR OLS =========================================
# need to split into train test
def linear_ols_model(df):
    '''
    ==Function==
    Creates X & y
    
    ==Returns==
    |linear_model| : model of ols]
    |linear_trend| : df of fitted values] 
    '''
    train, test = reg_test_train(df)
    X_train = add_constant(np.arange(1, len(train) + 1))
    X_test = add_constant(np.arange(1, len(test) + 1))
    y = df
    
    linear_model = sm.OLS(y, X_train).fit()
    linear_trend = linear_model.predict(X_train)
    return linear_model ,linear_trend

    
# === RANDOM FOREST =========================================    



def random_forest_model(df):

    '''
    ==Function==
    Uses simple Random Forest Regressor to forecast
   
    ==Returns==
    |rf_trend| : df of fitted values]
    
    ==Prints== 
    MSE score
    '''

    count_cons, count, idx, df_vals = [], 1, round(len(df)*.8), df.cost_per_watt.values
    for i in range(1, len(df)+1):
        count_cons.append((count, 1))
        count +=1
    X_train, y_train, X_test, y_test = count_cons[:idx], df_vals[:idx], count_cons[idx:],df_vals[idx:]
    rf_model = RandomForestRegressor(n_jobs=-1).fit(X_train,y_train)
    rf_trend = rf_model.predict(X_test)
    print('MSE =', mean_squared_error(y_test, rf_trend) )
    return rf_trend


# need to re-evalute RF code. should not have constant
# need to split into train test
# def randomforest_model(df):
#     '''
#     ==Function==
#     Uses simple Random Forest Regressor to forecast
   
#     ==Returns==
#     |rf_model| : the model of the rf regressor]
#     |rf_trend| : df of fitted values]
#     '''
    
#     counter_r, count = [], 1
#     for i in range(1, len(df)+1):
#         counter_r.append((count,count))
#         count +=1
#     ct_arr = np.array(counter_r)
#     X = ct_arr
#     y = df.cost_per_watt.values
#     rf_model = RandomForestRegressor(n_jobs=-1).fit(X,y)
#     rf_trend = rf_model.predict(X)
#     return rf_model,rf_trend

# def randomforest_model(df):
#     count_cons, count = [], 1
#     for i in range(1, len(df)+1):
#         count_cons.append((count,1))
#         count +=1
#     ct_con = np.array(count_cons)
#     df_vals = df.cost_per_watt.values
#     idx = round(len(df)*.8)
#     X_train, y_train, X_test, y_test = ct_con[:idx], df_vals[:idx], ct_con[idx:],df_vals[idx:]
#     rf_model = RandomForestRegressor(n_jobs=-1).fit(X_train,y_train)
#     rf_trend = rf_model.predict(X_test)
#     return rf_model,rf_trend, mean_squared_error(y_test, rf_trend)


def random_forest_model(df):
    count_cons, count, idx, df_vals = [], 1, round(len(df)*.8), df.cost_per_watt.values
    for i in range(1, len(df)+1):
        count_cons.append((count, 1))
        count +=1
    X_train, y_train, X_test, y_test = count_cons[:idx], df_vals[:idx], count_cons[idx:],df_vals[idx:]
    rf_model = RandomForestRegressor(n_jobs=-1).fit(X_train,y_train)
    rf_trend = rf_model.predict(X_test)
    print('MSE =', mean_squared_error(y_test, rf_trend) )
    return rf_trend


# =============================================================================
# SCORING/ TESTING/ PLOTTING
# =============================================================================   

# === MSE OF REGRESSION MODELS =========================================
def score_table(df, ols_model, linear_model, rf_model):
    '''
    ==Returns==
    Table with MSE scores for each regression model 
    ''' 
    rf_trend = rf_model.predict(add_constant(np.arange(1,len(df)+ 1)))
    models = ['OLS', 'LINEAR', 'RF',]
    reg_scores = pd.DataFrame(models)
    reg_scores.rename(columns={0:'Models'}, inplace=True)
    reg_scores.set_index('Models', drop=True, inplace= True)
    reg_scores['MSE'] = [round(mean_squared_error(df[3:], ols_model.fittedvalues),5), round(mean_absolute_error(df, linear_model.fittedvalues),5), round(mean_squared_error(df,rf_trend),5)]
    return reg_scores
    
    
# === MSE OF STATIONARY REGRESSION MODELS =========================================   

def stat_score_table(df, ols_model, linear_model, rf_model, reg_scores):
    '''
    ==Function==
    Specifically for after using regression models on differenced/stationary data
    
    ==Parameters==
    |df| : differenced/stationary data
    ols, linear, rf models :
    |reg_scores| :  previous score table from regression models on original data
    
    ==Returns==
    Table with MSE scores for each regression model 
    '''
    df = df.dropna()
    rf_trend = rf_model.predict(add_constant(np.arange(1,len(df)+ 1)))
    models = ['OLS_DIFF', 'LINEAR_DIFF', 'RF_DIFF',]
    reg_scores = pd.DataFrame(models)
    reg_scores.rename(columns={0:'Models'}, inplace=True)
    reg_scores.set_index('Models', drop=True, inplace= True)
    tsreg_scores['MSE'] = [mean_squared_error(df[3:], tsols_model.fittedvalues), mean_absolute_error(df, tslinear_model.fittedvalues), mean_squared_error(df,tsrf_trend)]
    diff_reg_scores = (stat_score_table(diff, ols_model, linear_model, rf_model).T)
    diff_reg_scores = diff_reg_scores.T
    scores_for_all = pd.concat([reg_scores, diff_reg_scores])
    return scores_for_all
    
# === TEST ON STATIONARY MODELS =========================================
def stationary_test_on_models(ols_model, linear_model, rf_trend):
    ols_df, lin_df, rf_df = pd.DataFrame(ols_model.fittedvalues), pd.DataFrame(linear_model.fittedvalues), pd.DataFrame(rf_trend)
    model_list = [ols_df, lin_df, rf_df]
    print('p-value of original data (ols, linear,rf)')
    [print((adfuller(i, autolag='AIC')[1]))for i in model_list]
    print('-------')
    print('p-value of differenced data(ols, linear,rf)')
    [print(adfuller(i.diff(periods=1).dropna(), autolag='AIC')[1]) for i in model_list]

    
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

    
# === ARIMA ON LINEAR MODEL RESIDUALS =========================================    
def lm_residual_model(lm_residuals):
    '''
    ==Function==
    ARIMA on LM residuals
    
    ==Note==
    for use in other funcs
    '''
    lm_residual_model = ARIMA(
    lm_residuals, order=( )).fit()

    
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
