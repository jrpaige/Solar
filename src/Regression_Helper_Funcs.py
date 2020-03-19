import numpy as np
import pandas as pd
import sys
import datetime

# MATH
from math import sqrt
from scipy import signal
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, make_scorer, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline,  make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

#VISUALIZATION 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6

#TIME
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.tsa import stattools
from statsmodels.tools.tools import add_constant
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression import *
import pyramid
from pmdarima.arima import auto_arima

   
# === INITIAL REGRESSION MODELS =========================================
def lag_ols_model(df):    
    '''
    Creates lag table and processes through OLS
    Returns:
        [ols_model: ols of 3 lagged colummns]
        [ols_trend: df of fitted values]
    '''
    lag_cost = (pd.concat([df.shift(i) for i in range(4)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, 4)])).dropna()
    ols_model = smf.ols('y ~ Lag1 + Lag2 + Lag3', data=lag_cost).fit() 
    ols_trend = ols_model.fittedvalues
    return ols_model, ols_trend


def linear_ols_model(df):
    '''
    creates X & y
    plots linear regression line
    returns: linear_model ,linear_trend
    '''
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    linear_model = sm.OLS(y, X).fit()
    linear_trend = linear_model.predict(X)
    return linear_model ,linear_trend

    
def randomforest_model(df):
    '''
    Uses simple Random Forest Regressor to forecast
    '''
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    rf_model = RandomForestRegressor(n_jobs=-1).fit(X,y)
    rf_trend = rf_model.predict(X)
    return rf_model,rf_trend

def score_table(df, ols_model, linear_model, rf_model):
    rf_trend = rf_model.predict(add_constant(np.arange(1,len(df)+ 1)))
    models = ['OLS', 'LINEAR', 'RF',]
    reg_scores = pd.DataFrame(models)
    reg_scores.rename(columns={0:'Models'}, inplace=True)
    reg_scores.set_index('Models', drop=True, inplace= True)
    reg_scores['MAE'] = [mean_absolute_error(df[3:], ols_model.fittedvalues), mean_absolute_error(df, linear_model.fittedvalues), mean_absolute_error(df,rf_trend)]
    reg_scores['MSE'] = [mean_squared_error(df[3:], ols_model.fittedvalues), mean_absolute_error(df, linear_model.fittedvalues), mean_squared_error(df,rf_trend)]
    reg_scores['RMSE'] = [np.sqrt(reg_scores.MSE[0]), np.sqrt(reg_scores.MSE[1]), np.sqrt(reg_scores.MSE[2])]
    
    ols_df, lin_df, rf_df = pd.DataFrame(ols_model.fittedvalues), pd.DataFrame(linear_model.fittedvalues), pd.DataFrame(rf_trend)
    #reg_scores['P_VALUE'] = [ adfuller(ols_df, autolag='AIC')[1],adfuller(lin_df, autolag='AIC')[1], adfuller(rf_df, autolag='AIC')[1]]   
    return reg_scores
    
def plot_regres_model(df, model_trend, model_string_name):    
    fig, ax = plt.subplots(1, figsize=(16, 3))
    ax.plot(df.index, df, label= 'cost_per_watt')
    ax.plot(df.index, model_trend, label= model_string_name)
    plt.ylabel('Cost Per Watt ($)')
    plt.xlabel('Year')
    plt.legend(loc='best')
    ax.set_title("Weekly Median Cost Per Watt Over Time with Trendline via {}".format(model_string_name))
    plt.show()
    print(model_name.summary())
    
    
    
# === STATIONARY REGRESSION =========================================   


def stat_lag_ols_model(df):    
    '''
    Creates lag table and processes through OLS
    Returns:
        [ols_model: ols of 3 lagged colummns]
        [ols_trend: df of fitted values]
    '''
    df = df[1:]
    tslag_cost = (pd.concat([df.shift(i) for i in range(4)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, 4)])).dropna()
    tsols_model = smf.ols('y ~ Lag1 + Lag2 + Lag3', data=tslag_cost).fit() 
    tsols_trend = tsols_model.fittedvalues
    return tsols_model, tsols_trend
    
def stat_linear_ols_model(df):
    '''
    creates X & y
    plots linear regression line
    returns: linear_model ,linear_trend
    '''
    df = df[1:]
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    tslinear_model = sm.OLS(y, X).fit()
    tslinear_trend = tslinear_model.predict(X)
    return tslinear_model ,tslinear_trend

    
def stat_randomforest_model(df):
    '''
    Uses simple Random Forest Regressor to forecast
    '''
    df = df[1:]
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    tsrf_model = RandomForestRegressor(n_jobs=-1).fit(X,y)
    tsrf_trend = tsrf_model.predict(X)
    return tsrf_model,tsrf_trend

def stat_score_table(df, tsols_model, tslinear_model, tsrf_model):
    df = df[1:]
    tsrf_trend = tsrf_model.predict(add_constant(np.arange(1,len(df)+ 1)))
    tsmodels = ['OLS', 'LINEAR', 'RF',]
    tsreg_scores = pd.DataFrame(tsmodels)
    tsreg_scores.rename(columns={0:'Models'}, inplace=True)
    tsreg_scores.set_index('Models', drop=True, inplace= True)
    tsreg_scores['MAE'] = [mean_absolute_error(df[3:], tsols_model.fittedvalues), mean_absolute_error(df, tslinear_model.fittedvalues), mean_absolute_error(df,tsrf_trend)]
    tsreg_scores['MSE'] = [mean_squared_error(df[3:], tsols_model.fittedvalues), mean_absolute_error(df, tslinear_model.fittedvalues), mean_squared_error(df,tsrf_trend)]
    tsreg_scores['RMSE'] = [np.sqrt(tsreg_scores.MSE[0]), np.sqrt(tsreg_scores.MSE[1]), np.sqrt(tsreg_scores.MSE[2])]
    ols_df, lin_df, rf_df = pd.DataFrame(tsols_model.fittedvalues), pd.DataFrame(tslinear_model.fittedvalues), pd.DataFrame(tsrf_trend)
    #tsreg_scores['P_VALUE'] = [ adfuller(ols_df, autolag='AIC')[1],adfuller(lin_df, autolag='AIC')[1], adfuller(rf_df, autolag='AIC')[1]]   
    return tsreg_scores
    
def stat_plot_regres_model(df, model_trend, model_string_name):    
    fig, ax = plt.subplots(1, figsize=(16, 3))
    ax.plot(df.index, df, label= 'cost_per_watt')
    ax.plot(df.index, model_trend, label= model_string_name)
    plt.ylabel('Cost Per Watt ($)')
    plt.xlabel('Year')
    plt.legend(loc='best')
    ax.set_title("Weekly Median Cost Per Watt Over Time with Trendline via {}".format(model_string_name))
    plt.show()
    print(model_name.summary())



# === REGRESSION MODELS =========================================
def stationary_test_on_models(ols_model, linear_model, rf_trend):
    ols_df, lin_df, rf_df = pd.DataFrame(ols_model.fittedvalues), pd.DataFrame(linear_model.fittedvalues), pd.DataFrame(rf_trend)
    model_list = [ols_df, lin_df, rf_df]
    print('p-value of original data (ols, linear,rf)')
    [print((adfuller(i, autolag='AIC')[1]))for i in model_list]
    print('-------')
    print('p-value of differenced data(ols, linear,rf)')
    [print(adfuller(i.diff(periods=1).dropna(), autolag='AIC')[1]) for i in model_list]

def rolling_ols(df):  #not helpful
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    rolols_model = RollingOLS(y, X, window=3).fit()
    #rolols_trend = rolols_model.predict(X)
    return rolols_model
   
# LINEAR  ___

def rob_lin(df):
    '''
    robust linear
    '''
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    roblin_model = sm.RLM(y,X, ).fit()
    roblin_trend = roblin_model.predict()
    return roblin_model, roblin_trend 
 
def least_squares(df):
    y = df
    X = add_constant(np.arange(1, len(y) + 1))
    lst_sq_mods = pd.DataFrame()
    lst_sq_mods['OLS'] = sm.OLS(y, X).fit().predict(X)
    lst_sq_mods['GLS'] = sm.GLS(y, X).fit().predict(X)
    lst_sq_mods['avg'] = linear_plots.mean(axis=1)
    lst_sq_mods['orig'] = np.array(y.cost_per_watt)
    return lst_sq_mods

def lm_resids(df, linear_trend):    
    '''
    takes in df and linear trend
    '''
    lm_residuals = pd.Series(df.cost_per_watt - linear_trend, index=df.index)
    fig, axs = plt.subplots(3, figsize=(16, 8))
    # The model forecasts zero for the first few datapoints, so the residuals
    # are the actual values.
    axs[0].plot(lm_residuals.index, lm_residuals)
    plot_acf_and_pacf(lm_residuals, axs[1:])
    plt.tight_layout()

def lm_residual_model(lm_residuals):
    lm_residual_model = ARIMA(
    lm_residuals, order=( )).fit()

def lm_preds(df): 
    X = np.column_stack([df,
                     add_constant(np.arange(1, len(df) + 1))])
    lm_preds = pd.Series(
    linear_model.predict(X),
    index=df.index) 
    #lm_preds= lm_preds[arima_preds.index.min():]
    
def holt_linear_model(df):
    y_hat_avg = df.copy()
    fit1 = Holt(np.asarray(df['cost_per_watt'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
    y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
    plt.figure(figsize=(16,8))
    plt.plot(df['cost_per_watt'], label='Cost Per Watt')
    plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
    plt.legend(loc='best')
    plt.show()
    model_type = 'Holt_linear'
    print('RMS Score:',  rms_score(df, model_type))
    rms = sqrt(mean_squared_error(len(df), y_hat.model_type))
    return rms
    
    
# RF ___
def rf_gs(df): 
    '''
    Random Forest Grid Search
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



# === COEFFICIENTS =========================================
def model_coefs_params(model_list):
    [print('Params for {} \n {}'.format(i, i.params)) for i in model_list]   
    
# === Covariance =========================================    
        
def cov_table(df):
    '''
    Estimate a covariance matrix
    Can Enter in lag_cost from ols
    '''
    plt.figure(figsize=(20,12))
    plt.plot(np.cov(df))
    plt.show()
    return np.cov(df)
