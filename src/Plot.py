import numpy as np
import pandas as pd
import sys
import datetime
from datetime import datetime

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
