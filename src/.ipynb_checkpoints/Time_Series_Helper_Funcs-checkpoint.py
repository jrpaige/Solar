import numpy as np
import pandas as pd
import sys
import datetime
from datetime import datetime

# MATH
from math import sqrt
from scipy import signal
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
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
# TIME SERIES PREP
# =============================================================================

# === REGULAR TRAIN TEST SPLIT =========================================     
def train_test(df):
    '''
    ==Function==
    Splits data into train and test sets 
    
     ==Returns==   
    |train| = first 80% of df's data 
    |test| = last 20% of df's data
    '''
    idx = round(len(df)*.8)
    train, test = df[:idx], df[idx:]
    return train, test
        
# === ROLLING PLOTS =========================================
def rolling_plot(df):
    '''
    ==Returns==
    Plot with 
    - Original data 
    - Data with a rolling mean window of 3 
    - Data with a rolling median window of 3
    - Data with a rolling standard deviation window of 3
    '''
    ywn = pd.DataFrame(df.cost_per_watt).dropna()
    rollingmedian = ywn.rolling(window=3).median()
    rollingmean = ywn.rolling(window=3).mean()
    rollingstd = ywn.rolling(window=3).std() 
    orig = plt.plot(df, color='blue',label='Original')
    mean = plt.plot(rollingmean, color='green', label='Rolling Mean')
    med = plt.plot(rollingmedian, color='red', label='Rolling Median')
    std = plt.plot(rollingstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Weekly Rolling Mean, Median, \
              & Standard Deviation with Window 3')
    plt.ylabel('Cost Per Watt $')
    plt.show()
        
# === DICKEY FULLER TEST =========================================        
def dfuller_test(df):
    '''
    ==Function ==
    Performs Dickey-Fuller test using AIC
    
    ==Returns==
    Results including:
    Test Statistic, p-value, #Lags Used, #Observations Used, and Critical Values
    '''
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(df, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                             'p-value','#Lags Used',
                                             '#Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

    
# === AUTOCORRELATION PLOTS =========================================    
def autocor_plots(df):
    '''
    ==Returns==
    Autocorrelation and Partial Autocorrelation plots
    '''
    fig, ax = plt.subplots(2, figsize=(15,7))
    sm.graphics.tsa.plot_acf(df, ax=ax[0])
    sm.graphics.tsa.plot_pacf(df, ax=ax[1])
    plt.show()   

    
# === GET DIFFERENCED DATA ========================================= 
def get_differences(df):
    '''
    ==Function ==
    Differences the data to attempt to achieve stationarity
    Each data point is representative of the change in value from the previous data point
    
    ==Returns==
    weekly_differences
    '''
    weekly_differences = df.diff(periods=1)
    plt.plot(weekly_differences.index, weekly_differences)
    # The first entry in the differenced series is NaN.
    plot_acf(weekly_differences[1:]) #lags=lags)
    plot_pacf(weekly_differences[1:]) #lags=lags) 
    plt.tight_layout()
    plt.show()
    return weekly_differences


# === STATIONARITY TESTING ========================================= 
def test_for_stationarity(df):  
    '''
    ==Function ==
    Tests data for stationarity 
    
    ==Returns==
    p-value and result
    
    ==Input Suggestion==
    weekly_differences
    
    '''
    test = adfuller(df)
    print("ADF p-value: {0:2.2f}".format(test[1]))
    if test[1] < 0.51:
        print('Achieved stationarity! Reject ADF H0.')
    else:
        print('Time Series is not stationary. Fail to reject ADF H0')
        
        
# === TIME SERIES TRAIN TEST SPLIT =========================================
def time_train_test_split(df):
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
        plt.plot(train)
        plt.plot([None for i in train] + [x for x in test])
        plt.show()
    return train, test       




# =============================================================================
# CORRELATION & COEFFICIENTS
# =============================================================================        


# === CREATE LAG =========================================
 
def series_lag(series, lag=1):
    '''
    ***For use within plot_ac_scat function***
    '''
    truncated = np.copy(series)[lag:]
    lagged = np.copy(series)[:(len(truncated))]
    return truncated, lagged


# === COMPUTE AUTOCORRELATION NUMBERS =========================================
def compute_autocorrelation(series, lag=1):
    '''
    ***for use within plot_ac_scat function***
    '''
    series, lagged = series_lag(series, lag=lag)
    autocorr = np.corrcoef(series, lagged)[0, 1]
    return autocorr 


# === PLOT AC, PARTIAL AC, HIST, & LINE  =========================================
def tsplot(y, lags=None, title='', figsize=(14, 8)):
    '''
    ==Input Suggestion==
    tsplot(ts_train, title='', lags=)
    '''
    
    
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax   = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax  = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    
    y.plot(ax=ts_ax)
    ts_ax.set_title(title)
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    fig.tight_layout()
    return ts_ax, acf_ax, pacf_ax


# === PLOT AUTOCORRELATION ON A SCATTER PLOT =========================================
def plot_ac_scat(df):
    '''
    ==Returns==
    Autocorrelation scatter plot
    
    ==Input Suggestion==
    weekly_differences
    '''
    df = np.array(df)
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))

    lags = [1,2,3,4,5,6,7,26,52]

    for i, ax in zip(lags,axs.flatten()):
        series, lagged = series_lag(df, lag=i)
        autocorr = compute_autocorrelation(df, lag=i)
        ax.scatter(series, lagged, alpha=0.5)
        ax.set_title("Lag {0}".format(i))
        #ax.set_title("Lag {0} AC: {1:2.2f}".format(i, autocorr))
        plt.tight_layout()

# === PLOT COEFFICIENTS =========================================        
def plotcoefficients(model):
    ''' 
    ==Parameters==
    |model| = model.fit() variable
    
    ==Returns==
    Plot with sorted coefficient values of the model
    '''
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');

    
# =============================================================================
# SIMPLE TIME SEIRES
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

    
# =============================================================================
# VALIDATE/SCORE
# =============================================================================           

# === GRID SEARCH EXPONENTIAL SMOOTHING  ===============================================  
def walk_forward_validation(df, n_test, cfg):
    '''
    ==Function==
    used with score_model function and grid_search function
    walk-forward validation for univariate data
    
    ==Note==
    # forecst = list() 
    '''
    train, test = train_test_split(df, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = exp_smoothing_forecast(history, cfg) 
                      # store forecast in list of predictions        
        forecst.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate forecast error
    error = measure_rmse(test, forecast)
    return error
    
# === SCORE ===============================================  
def score_model(df, n_test, cfg, debug=False):
    '''
    ==Function==
    Scores a model
    
    ==Returns==
    cfg key and result
    or None on failure
    '''
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(df, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(df, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)

# === GRID SEARCH ===============================================  
def grid_search(df, cfg_list, n_test, parallel=True):
    '''
    ==Function==
    Grid searches for configs
    '''
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(df, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(df, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores   
        
#scores = grid_search(data, cfg_list, n_test)