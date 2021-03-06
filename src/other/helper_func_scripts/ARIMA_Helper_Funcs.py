import numpy as np
import pandas as pd
import sys
import datetime
from datetime import datetime
from src.Time_Series_Helper_Funcs import *

# MATH
from math import sqrt
from scipy import signal
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score, mean_squared_error, make_scorer, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, KFold, GridSearchCV
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
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
plt.style.use('ggplot')

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def train_test(df):
    """    
    ==Function==
    Splits data into train and test sets 

    ==Returns==   
    |train| = first 80% of df's data 
    |test| = last 20% of df's data
    """
    idx = round(len(df)*.8)
    train, test = df[:idx], df[idx:]
    return train, test

def ARIMA_predict(df, order):
    
    train, test = train_test(df)
    test_s, test_e = test.index.date[0], test.index.date[-1]
    train_s, train_e = train.index.date[0], train.index.date[-1]
    res = ARIMA(train, order=order).fit()
    fig, ax = plt.subplots(1, figsize=(14, 4))
    ax.plot(test.index, test)
    ax.plot(train.index[-20:], train[-20:])
    fig = res.plot_predict(test_s,test_e, ax=ax, plot_insample=True)
    
    # plt.title('MSE {}'.format(round(mean_squared_error(test,res.predict('2015-06-14','2019-1-6')),5)))
    plt.title('Forecasted [{} - {}] Data \n Based On [{} - {}] Data\n ARIMA {} MSE= {}'.format(
                                test_s, test_e, 
                                train_s, train_e,order,
                                round(mean_squared_error(test,res.predict(test_s, test_e)),5)))
    plt.show()

    
    # =============================================================================
# ARIMA PARAMETERS
# ============================================================================= 

    
    

# === GET PDQ VIA AUTO ARIMA =========================================
def auto_arima_pdq(df,trace_list=False):
    '''
    ==Function==
    Uses Auto ARIMA to obtain best parameters for data
    ==Parameters==
    |trace_list| : bool
        if True, function will return list of all searched pairs
        default=False
    ==Returns==
    printed pdq variable
    auto_arima variable to use in other functions
    '''
    arima_pdq = auto_arima(df, trace=trace_list, stepwise=False, max_p=8,max_P = 8, max_order=12).order
    print('P, D, Q parameters to use in ARIMA model =', arima_pdq)
    return arima_pdq
 
# === TEST VARIOUS PDQ'S MSE =========================================    
def evaluate_arima_model(X, arima_order):
    '''
    ==Function ==
    Splits data into training/test
    Pushes through ARIMA models 
   
    ==Returns==
    MSE
    
    ==Note==
    Only used in arima_order_mses function
    '''
    # prepare training dataset
    train_size = int(len(X) * 0.8)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order, missing='drop')
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# === FIND BEST PARAMETERS BY RUNNING THROUGH DIFFERENT PDQ'S =========================================      
def arima_order_mses(df):
    '''
    ==Function==
    Uses various p,d,qs within below range
    Tests out each combination 
    
    ==Returns== 
    Params with the best cfg + best MSE
    
    ==Input Suggestion==
    Use [evaluate_models(df.values.dropna(), p_values, d_values, q_values)]
    
    ==Note==
    Computationally expensive! 
    '''
    df = df.dropna().values
    p_values = [0, 1, 2, 4,]
    d_values = range(0, 3)
    q_values = range(0, 3)
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(df, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.4f' % (order,mse))
                except:
                    continue
    print('Best ARIMA %s MSE=%.4f' % (best_cfg, best_score))
    
          

# =============================================================================
# ARIMA TIME SERIES
# ============================================================================= 
  
# === ARIMA MODEL =========================================

def arima_model(df, order, years_off, plot, use_years):
    '''   
    ==Parameters==
    |use_years| - bool
        if use_years =True, function will use years_off parameter
        if use_years = False, function will use a simple train/test 80/20 split                     
    |years_off| - integer
        if use_years=True, years_off = number of years to set aside as test set 
        if use_years=False, feel free to enter 0 or 'NA'
    |order| - tuple
        should be entered in as a tuple ex. "order=(1,1)"
    |plot| - bool
        if plot=True, function will return a plot with data in the title
        if plot=False, function will print out ARMA order used and resulting MSE 
    ==Returns==
    plot and/or MSE score
    '''
    df = df.dropna()
    if use_years == True:
        train = df[:len(df) - (52*years_off)]
        test = df[len(df) - (52*years_off):]
        pred = ARIMA(train, order=order, freq='W').fit().predict(start=test.index.date[0], end=test.index.date[-1])
    else:
        train= df.dropna()[:round(len(df.dropna())*.8)]
        test = df[len(train):]
        order=order
        pred = ARIMA(train, order, freq='W').fit().predict(start=test.index.date[0],end=test.index.date[-1])

    if plot==True:
        #plot_arima(test, pred,train, order)
        arima_plot(test, train, pred, order)
    else:
        print('ARIMA Order Used: {}'.format(order))
        print('MSE:',round(mean_squared_error(test, pred),5))

          



# === ARIMA MODELS VIA SKTIME =========================================   

def skt_arima(df, order):
    '''
    ==Function==
    Splits dataset into 70/30 train test split
    Applies the ARIMAForecaster from sktime 
    Collects Forecast predictions 
    
    ==Parameters==
    |order| should be entered in as a tuple ex. "order=(1,1,1)"
    If no preferred order, enter in "auto_arima(df.dropna()).order"
    
    ==Returns==
    Plot with the train, test, and forecast data
    The model's MSE score
    '''
    idx = round(len(df) * .8)
    tsdf = df['cost_per_watt'].dropna()
    tsdf.reset_index(drop=True, inplace=True)
    train = pd.Series([tsdf[:idx]])
    test = pd.Series([tsdf[idx:]])
    m = ARIMAForecaster(order=order)
    m.fit(train)
    fh = np.arange(1, (len(tsdf)-idx)+1)
    y_pred = m.predict(fh=fh)
    skt_mse = m.score(test, fh=fh)**2
    fig, ax = plt.subplots(1, figsize=plt.figaspect(.25))
    train.iloc[0][-30:].plot(ax=ax, label='train')
    test.iloc[0].plot(ax=ax, label='test')
    y_pred.plot(ax=ax, label='forecast')
    ax.set(ylabel='cost_per_watt')
    plt.title('SKT ARIMA Model MSE ={}'.format(round(skt_mse,5)))
    plt.legend(loc='best')
    plt.show()

# === PLOT ARIMA MODELS VIA SKTIME =========================================             
def skt_arima_plot(test,train,y_pred,fh,  skt_mse):    
    
