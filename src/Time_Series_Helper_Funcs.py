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

# === REMOVE FIRST FEW YEARS DUE TO NON-CONTINUITY =========================================
def time_frame(df):
    '''
    ==Function==
    Removes the volatile data points in the beginning 
    Resamples data into the weekly medians 
    Time Frame = 1/2002 - 12/2018 
    
    ==Returns== 
    |y| : resampled and cropped df 
    '''
    sdf = df.loc[df.index > '2000-12-31']
    y = pd.DataFrame(sdf.cost_per_watt)
    y = pd.DataFrame(y['cost_per_watt'].resample('W').median())
    return y


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
    test = adfuller(df.dropna())
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
    
# === SIMPLE TIME SERIES MODELS =========================================
def simple_move(df): 
    '''
    ==Returns== 
    Simple forecast based on shifted data of 1 week and 3 week lags
    '''
    forcst = pd.DataFrame(df.loc[df.index.year>2017])
    forcst['cost_1weekago'] = df['cost_per_watt'].shift(1)
    forcst['cost_3weeksago'] = df['cost_per_watt'].shift(3)
    print('MSE for cost_1weekago =', mean_squared_error(forcst['cost_1weekago'],forcst['cost_per_watt']).round(4))
    print('MSE for cost_3weeksago =', mean_squared_error(forcst['cost_3weeksago'],forcst['cost_per_watt']).round(4))


# =============================================================================
# ARMA MODELS
# =============================================================================         
    
# === GET PD FOR ARMA VIA AUTO ARIMA =========================================
def auto_arma_pd(df,trace_list=False):
    '''
    ==Function==
    Uses Auto ARIMA to obtain best parameters for data
    ==Parameters==
    |trace_list| : bool
        if True, function will return list of all searched pairs
        default=False
    ==Returns==
    printed pd variable
    arma_pd variable to use in other functions
    '''
    arma_pd = auto_arima(df, trace=trace_list, stepwise=False, max_p=6, max_P =6, max_order=6).order[:2]
    print('P, D parameters to use in ARMA model =', arma_pd)
    return arma_pd

    
# === ARMA MODEL  =========================================
def arma_model(df, order, years_off, plot, use_years):
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
        m = ARMA(train, order=order).fit(train)
        pred = ARMA(train, order=order).fit().predict(start=test.index.date[0], end=test.index.date[-1])
    else:
        df = df.dropna()
        idx = round(len(df) * .8)
        train= df[:idx]
        test = df[idx:]
        order=order
        pred = ARMA(train, order).fit().predict(start=test.index.date[0],end=test.index.date[-1])
    
    if plot==True:
        arma_plot(test, train,pred, order)
    else:
        print('ARMA Order Used: {}'.format(order))
        #print('MSE:',round(.red_error(test, pred),5))
    
# === PLOT ARMA =========================================    
def plot_arma(test_data, ARMA_preds,train_data, order):    
    test_start, test_end = test_data.index.year[0], test_data.index.year[-1]
    forcst_start, forcst_end = train_data.index.year[0], train_data.index.year[-1]
    plt.plot(test_data, label='Actual', alpha=0.5)
    plt.plot(ARMA_preds, label= 'Forecast', color='black')
    plt.legend(loc='best')
    plt.title('Forecasted [{} - {}] Data \n Based On [{} - {}] Data\n ARMA {} MSE= {}'.format(
                                test_start, test_end, 
                                forcst_start, forcst_end,order,
                                round(mean_squared_error(test_data, ARMA_preds),5)))
    plt.show()
    
    
def arma_plot(test_data,train_data,ARMA_preds, order):    
    test_start, test_end = test_data.index.year[0], test_data.index.year[-1]
    forcst_start, forcst_end = train_data.index.year[0], train_data.index.year[-1]
    fig, ax = plt.subplots(1, figsize=(10,6))
    train_data.plot(ax=ax, label='Train', linewidth=1)
    test_data.plot(ax=ax, label='Test', linewidth=1)
    ARMA_preds.plot(ax=ax, label='Forecast', color='black', alpha=0.7,linewidth=2)
    ax.set(ylabel='cost_per_watt')
    plt.title('Forecasted [{} - {}] Data \n Based On [{} - {}] Data\n ARMA {} MSE= {}'.format(
                                test_start, test_end, 
                                forcst_start, forcst_end,order,
                                round(mean_squared_error(test_data, ARMA_preds),5)))
    plt.legend(loc='best')
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
    #p_values = [0, 1, 2, 4, 6, 8, 10]
    p_values = [0, 1, 2, 4, 5, 6, 7]
    d_values = range(0, 3)
    q_values = range(0, 4)
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

          
# === PLOT ARIMA MODEL =========================================    
def plot_arima(test_data, ARIMA_preds,train_data, order):    
    test_start, test_end = test_data.index.year[0], test_data.index.year[-1]
    forcst_start, forcst_end = train_data.index.year[0], train_data.index.year[-1]
    plt.plot(test_data, label='Actual', alpha=0.5)
    plt.plot(ARIMA_preds, label= 'Forecast')
    plt.legend(loc='best')
    plt.title('Forecasted [{} - {}] Data \n Based On [{} - {}] Data\n ARIMA {} MSE= {}'.format(
                                test_start, test_end, 
                                forcst_start, forcst_end,order,
                                round(mean_squared_error(test_data, ARIMA_preds),5)))


def arima_plot(test_data,train_data,ARIMA_preds, order):    
    test_start, test_end = test_data.index.year[0], test_data.index.year[-1]
    forcst_start, forcst_end = train_data.index.year[0], train_data.index.year[-1]
    fig, ax = plt.subplots(1, figsize=plt.figaspect(.25))
    train_data.plot(ax=ax, label='Train')
    test_data.plot(ax=ax, label='Test')
    ARIMA_preds.plot(ax=ax, label='Forecast')
    ax.set(ylabel='cost_per_watt')
    plt.title('Forecasted [{} - {}] Data \n Based On [{} - {}] Data\n ARIMA {} MSE= {}'.format(
                                test_start, test_end, 
                                forcst_start, forcst_end,order,
                                round(mean_squared_error(test_data, ARIMA_preds),5)))
    plt.legend(loc='best')
    plt.show()    
    
    
# === SPECIFIC FORECAST =========================================    
def arima_model_forecast(df):
    '''
    ==Function==
    Forecasts for 2016-2019 using ARIMA
    
    ==Returns==
    model.fit()
    '''
    st_date = '2016-01-10'
    y_hat_avg = df[1:df.index.get_loc(st_date)-1]
    new_dates = pd.DataFrame(pd.date_range(start='2016-01-10', end='2019-01-06', freq='W'))
    new_dates['cost_per_watt'] = 0
    new_dates.set_index(0, drop=True, inplace=True)
    
    y_hat_avg = pd.concat([y_hat_avg, new_dates])
    
    fit1 = ARIMA(y_hat_avg['cost_per_watt'], order=(auto_arima(df.dropna()).order)).fit()
    fit_preds = pd.DataFrame(fit1.predict(start="2016-01-10", end="2019-01-06"))
    y_hat_avg['ARIMA'] = fit_preds
    plt.figure(figsize=(12,8))
    plt.plot(df['cost_per_watt'], label='Cost Per Watt')
    plt.plot(y_hat_avg['ARIMA'], label='ARIMA')
    plt.legend(loc='best')
    plt.title('ARIMA Model Predictions Beginning 1-10-2016')
    plt.show()
    print(' Mean Absolute Error =       {}\n Mean Squared Error =        {}\n Root Mean Squared Error =   {}'.format(round(mean_absolute_error(fit_preds,df[731:]),6), round(mean_squared_error(fit_preds,df[731:]),6), round(np.sqrt(mean_squared_error(fit_preds,df[731:]))),6))
    return fit1

          
# === GET ERROR SCORES FROM ARIMA MODEL RESULTS =========================================             
def arima_scores(res):
    '''
    ==Parameters==
    |res| = variable from model.fit() 
    '''
    print('standard errors :',res.bse)
    print('----------')
    print('pvalues :',res.pvalues)
    print('----------')
    print('residuals :',res.resid)
    print('----------')
    print('plot diagnositcs :', res.plot_diagnostics())


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
    skt_arima_plot(test,train,y_pred, fh, skt_mse)

# === PLOT ARIMA MODELS VIA SKTIME =========================================             
def skt_arima_plot(test,train,y_pred, skt_mse):    
    fig, ax = plt.subplots(1, figsize=plt.figaspect(.25))
    train.iloc[0].plot(ax=ax, label='train')
    test.iloc[0].plot(ax=ax, label='test')
    y_pred.plot(ax=ax, label='forecast')
    ax.set(ylabel='cost_per_watt')
    plt.title('ARIMA Model MSE ={}'.format(round(skt_mse,5)))
    plt.legend(loc='best')
    plt.show()

          
          
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