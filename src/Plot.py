import numpy as np
import pandas as pd
import sys
import datetime
from datetime import datetime
from src.Regression_Helper_Funcs import train_test_lag, multiple_regressors

# MATH
from math import sqrt
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, make_scorer, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline,  make_pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

#TIME
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
# PLOT
# =============================================================================

# === GENERAL MODEL PLOT FUNCTION =========================================
def model_plot(test_data,train_data,forecasts,method, order=None):
    '''
     ==Function==
    Plots the regression model entered
    
    ==Parameters==
    |method| : string
            name of regression or time series model used
            ex: 'Random Forest Regressor', 'ARIMA'
    |order| : tuple or None (default is set to None)
            if time series method is being used, enter in order used 
    '''
    test_start, test_end = test_data.index.year[0], test_data.index.year[-1]
    forcst_start, forcst_end = train_data.index.year[0], train_data.index.year[-1]
    fig, ax = plt.subplots(1, figsize=plt.figaspect(.25))
    train_data.plot(ax=ax, label='Train')
    test_data.plot(ax=ax, label='Test')
    forecasts.plot(ax=ax, label='{} Forecast'.format(method))
    ax.set(ylabel='cost_per_watt')
    if order==None:
        plt.title('Forecasted [{} - {}] Data \n Based On [{} - {}] Data\n {} Method  MSE= {}'.format(
                                    test_start, test_end, 
                                    forcst_start, forcst_end,method,
                                    round(mean_squared_error(test_data, forecasts),5)))
        
    else:
        plt.title('Forecasted [{} - {}] Data \n Based On [{} - {}] Data\n {} {} MSE= {}'.format(
                                    test_start, test_end, 
                                    forcst_start, forcst_end,method,order,
                                    round(mean_squared_error(test_data, forecasts),5)))
    plt.legend(loc='best')
    plt.show()    
    
    
# === REGRESSION MODEL PLOT WITH TRENDLINE =========================================
def plot_regres_model(df, model_trend, model_name):  
    '''
    ==Function==
    Plots the regression model entered
    
    ==Parameters==
    |model_name| : should be entered as a string
    '''
    fig, ax = plt.subplots(1, figsize=(16, 3))
    ax.plot(df.index, df, label= 'cost_per_watt')
    ax.plot(df.index, model_trend, label= model_name)
    plt.ylabel('Cost Per Watt ($)')
    plt.xlabel('Year')
    plt.legend(loc='best')
    ax.set_title("Weekly Median Cost Per Watt Over Time with Trendline via {}".format(model_name))
    plt.show()

    
    


# === PLOT REGRESSION MODELS =========================================    
      
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

def plot_regression(df):    
    y_preds = regres_dfs(df)
    fig, axs = plt.subplots(3, figsize= (20,15))
    
    axs[0].plot(y_preds.actual, label= 'actual')
    axs[0].plot(y_preds.randomforest, label= 'Random Forest')
    axs[0].set_title('Random Forest \n MSE = {}'.format(round(mean_squared_error(y_preds.actual, y_preds.randomforest),5)))
    axs[0].legend(loc='best')
    
    axs[1].plot(y_preds.actual, label= 'actual')
    axs[1].plot(y_preds.olslinear, label= 'OLS Linear')
    axs[1].set_title('OLS Linear \n MSE = {}'.format(round(mean_squared_error(y_preds.actual, y_preds.olslinear),5)))
    axs[1].legend(loc='best')
    
    axs[2].plot(y_preds.actual, label= 'actual')
    axs[2].plot(y_preds.olssmf, label= 'OLS')
    axs[2].set_title('OLS smf \n MSE = {}'.format(round(mean_squared_error(y_preds.actual, y_preds.olssmf),5)))
    axs[2].legend(loc='best')
                                
    plt.show()


def plot_regs(df):
    '''
    ==Function==
    plots 6 regression models' forecasted values with the actual values
    
    ==Returns==
    6 subplots with MSE scores in each table's title
    '''
    y_preds = regres_dfs(df)
    fig, axs = plt.subplots(3, 2, figsize= (40,20))
    axs[0,0].plot(y_preds.actual, label= 'actual')
    axs[0,0].plot(y_preds.randomforest, label= 'Random Forest')
    axs[0,0].set_title('Random Forest \n MSE = {}'.format(round(mean_squared_error(y_preds.actual, y_preds.randomforest),5)))
    axs[0,0].legend(loc='best')

    axs[0,1].plot(y_preds.actual , label= 'actual')
    axs[0,1].plot(y_preds.linear, label= 'Linear')
    axs[0,1].set_title('Linear \n MSE = {}'.format(round(mean_squared_error(y_preds.actual, y_preds.linear),5)))
    axs[0,1].legend(loc='best')
    
    axs[1,0].plot(y_preds.actual, label= 'actual')
    axs[1,0].plot(y_preds.bagging, label= 'Bagging')
    axs[1,0].set_title('Bagging \n MSE = {}'.format(round(mean_squared_error(y_preds.actual, y_preds.bagging),5)))
    axs[1,0].legend(loc='best')
    
    axs[1,1].plot(y_preds.actual, label= 'actual')
    axs[1,1].plot(y_preds.adaboost, label= 'AdaBoost')
    axs[1,1].set_title('AdaBoost \n MSE = {}'.format(round(mean_squared_error(y_preds.actual, y_preds.adaboost),5)))
    axs[1,1].legend(loc='best')               
    
    axs[2,0].plot(y_preds.actual, label= 'actual')
    axs[2,0].plot(y_preds.olslinear, label= 'OLS Linear')
    axs[2,0].set_title('OLS Linear \n MSE = {}'.format(round(mean_squared_error(y_preds.actual, y_preds.olslinear),5)))
    axs[2,0].legend(loc='best')
    
    axs[2,1].plot(y_preds.actual, label= 'actual')
    axs[2,1].plot(y_preds.olssmf, label= 'OLS')
    axs[2,1].set_title('OLS smf \n MSE = {}'.format(round(mean_squared_error(y_preds.actual, y_preds.olssmf),5)))
    axs[2,1].legend(loc='best')                  
    plt.show()
    
  

