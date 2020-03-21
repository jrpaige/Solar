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
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import *
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, arma_order_select_ic, pacf_ols, pacf
import pyramid
from pmdarima.arima import auto_arima
from sktime.forecasters import ARIMAForecaster
from sktime.highlevel.tasks import ForecastingTask
from sktime.highlevel.strategies import ForecastingStrategy
from sktime.highlevel.strategies import Forecasting2TSRReductionStrategy
from sktime.transformers.compose import Tabulariser
from sktime.pipeline import Pipeline



#VISUALIZATION 
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
plt.style.use('ggplot')



# === TS PREP =========================================
def time_frame(df):
    '''
    ==Function==
    Removes the volatile data points in the beginning 
    Resamples data into the weekly medians 
    Time Frame = 1/2002 - 12/2018 
    
    ==Returns== 
    |y| : resampled and cropped df 
    '''
    sdf = df.loc[df.index > '2001-12-31']
    y = pd.DataFrame(sdf.cost_per_watt)
    y = pd.DataFrame(y['cost_per_watt'].resample('W').median())
    return y

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
    
def autocor_plots(df):
    '''
    ==Returns==
    Autocorrelation and Partial Autocorrelation plots
    '''
    fig, ax = plt.subplots(2, figsize=(15,7))
    sm.graphics.tsa.plot_acf(df, ax=ax[0])
    sm.graphics.tsa.plot_pacf(df, ax=ax[1])
    plt.show()   

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

        
# === CORRELATION & COEFFICIENTS =========================================
 
def series_lag(series, lag=1):
    '''
    ***For use within plot_ac_scat function***
    '''
    truncated = np.copy(series)[lag:]
    lagged = np.copy(series)[:(len(truncated))]
    return truncated, lagged

def compute_autocorrelation(series, lag=1):
    '''
    ***for use within plot_ac_scat function***
    '''
    series, lagged = series_lag(series, lag=lag)
    autocorr = np.corrcoef(series, lagged)[0, 1]
    return autocorr 

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

    
# === ARMA MODELS =========================================
def auto_arma_pd(df):
    '''
    ==Function==
    Uses Auto ARIMA to obtain best parameters for data
    ==Returns==
    printed pd variable
    arma_pd variable to use in other functions
    '''
    arma_pd = auto_arima(df).order[:2]
    print('P, D parameters to use in ARMA model =', arma_pd)
    return arma_pd

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
        pred = ARMA(train, order=order, freq='W').fit().predict(start=test.index.date[0], end=test.index.date[-1])
    else:
        train= df.dropna()[:round(len(df.dropna())*.8)]
        test = df[len(train):]
        order=order
        pred = ARMA(train, order, freq='W').fit().predict(start=test.index.date[0],end=test.index.date[-1])

    if plot==True:
        plot_arma(test, pred,train, order)
    else:
        print('ARMA Order Used: {}'.format(order))
        print('MSE:',round(mean_squared_error(test, pred),5))

def arma_years_model(df,order, years_off, plot):
    '''
    ==Parameters==
    |years_off| = number of years at the end of the df that the forecast should predict
    |order| should be entered in as a tuple ex. "order=(1,1,1)"
    |plot| if plot=True, function will return a plot with data in the title
            if plot=False, function will print out ARMA order used and resulting MSE
    
    ==Returns==
    ARMAmodel.fit()
    '''
    df = df.dropna()
    y_hat = df[:len(df) - (52*years_off)]
    actual = df[len(df) - (52*years_off):]
    arma_pred = ARMA(y_hat, order=order, freq='W').fit().predict(start=actual.index.date[0], end=actual.index.date[-1])
    if plot==True:
        plot_arma(actual, arma_pred,y_hat, order)
    else:
        print('ARMA Order Used: {}'.format(order))
        print('MSE:',round(mean_squared_error(actual, arma_pred),5))
    return ARMA(y_hat, order=order, freq='W').fit()

def simple_arma_model(df, plot):
    
    '''
     ==Parameters==
    |plot| if plot=True, function will return a plot with data in the title
            if plot=False, function will print out ARMA order used and resulting MSE
    ==Returns==
    ARMAmodel.fit()
    '''
    train= df.dropna()[:round(len(df.dropna())*.8)]
    test = df[len(train):]
    order=auto_arima(train).order[:2]
    mod = ARMA(train, order)
    result = mod.fit()
    pred = result.predict(start=test.index.date[0],end=test.index.date[-1])
    if plot==True:
        plot_arma(test, pred,train, order)
    else:
        print('ARMA Order Used: {}'.format(auto_arima(train).order[:2]))
        print('MSE:',round(mean_squared_error(test, pred),5))
    return result
    
def plot_arma(test_data, ARMA_preds,train_data, order):    
    test_start, test_end = test_data.index.year[0], test_data.index.year[-1]
    forcst_start, forcst_end = train_data.index.year[0], train_data.index.year[-1]
    plt.plot(test_data, label='Actual', alpha=0.5)
    plt.plot(ARMA_preds, label= 'Forecast')
    plt.legend(loc='best')
    plt.title('Forecasted [{} - {}] Data \n Based On [{} - {}] Data\n ARMA {} MSE= {}'.format(
                                test_start, test_end, 
                                forcst_start, forcst_end,order,
                                round(mean_squared_error(test_data, ARMA_preds),5)))
    plt.show()

    
# === ARIMA TIME SERIES MODEL=========================================

# === ARIMA PARAMS
def auto_arima_pdq(df):
    '''
    ==Function==
    Uses Auto ARIMA to obtain best parameters for data
    ==Returns==
    printed pdq variable
    auto_arima variable to use in other functions
    '''
    arima_pdq = auto_arima(df, seasonal=False, stationary=True).order
    print('P, D, Q parameters to use in ARIMA model =', arima_pdq)
    return arima_pdq
 
    
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
    p_values = [0, 1, 2, 4]
    d_values = range(0, 4)
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
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA %s MSE=%.3f' % (best_cfg, best_score))


# === ARIMA MODELS   

def basic_arima_model(df):  
    '''
    ==Function==
    Uses auto_arima's parameter suggestions and runs a no frills ARIMA
    train/test = 80/20 split
    
    ==Returns==
    Model Summary
    '''
    trn= df.dropna()[:round(len(df.dropna())*.8)]
    act = df[len(trn)+1:]
    mod = ARIMA(trn, order=auto_arima(trn).order)
    res = mod.fit()
    pred = res.predict(start=act.index.date[0],end=act.index.date[-1] )
    print('ARIMA Order Used: {}'.format(auto_arima(trn).order))
    print('MSE:',round(mean_squared_error(act, pred),5))

def arima_model_forecast(df):
    '''
    ==Function==
    Forecasts for 2016-2019 using ARIMA
    
    ==Returns==
    model.fit()
    '''
    st_date = '2016-01-10'
    y_hat_avg = df[1:df.index.get_loc(st_date)]
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

# === ARIMA MODELS VIA SKTIME

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
    
def skt_arima_plot(test,train,y_pred,fh, skt_mse):    
    fig, ax = plt.subplots(1, figsize=plt.figaspect(.25))
    train.iloc[0].plot(ax=ax, label='train')
    test.iloc[0].plot(ax=ax, label='test')
    y_pred.plot(ax=ax, label='forecast')
    ax.set(ylabel='cost_per_watt')
    plt.title('ARIMA Model MSE ={}'.format(round(skt_mse,5)))
    plt.legend(loc='best')
    plt.show()

# === VALIDATE/SCORE ===============================================  
 
    #gridsearch(ETS)===
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

def forecast_accuracy(forecast, actual):
    '''
    ==Returns==
    MAPE, ME, MAE, MPE, RMSE, and Corr(Actual,Forecast)
    '''
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    print('Mean Absolute Percentage Error:  ', mape, '\nMean Error:                      ',me, '\nMean Absolute Error :            ', mae, 
            '\nMean Percentage Error:           ', mpe, '\nRoot Mean Squared Error :        ',rmse, 
            '\nCorrelation between the \nActual and the Forecast:         ',corr)    


def mean_average_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true-y_pred)/ y_true)) * 100
                 
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))                                   
                        
def rms_score(df, model_type):
    '''
    ==Function==
    Calculates RMSE to check accuracy of model on data set
    
    ==Parameters==
    |model_type| = [moving_avg_forecast, Holt_linear, ARIMA, 
        OLS, RF, Linear Regression]
    
    ==Returns==
    MAPE, ME, MAE, MPE, RMSE, Correlation(Actual, Forecast)
    '''
    #rms = sqrt(mean_squared_error(len(df), y_hat.model_type))
    #return rms                     
    # actual = w_diff.cost_per_watt
    forecast = pd.DataFrame(df[1:])
    forecastt =  forecast.loc[forecast.index < '2019-01-07']
    mins = np.amin(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             #minmax
    acf1 = acf(fc-test)[1]                      #ACF1
    return({'Mean Absolute Percentage Error':mape, 'Mean Error':me, \
            'Mean Absolute Error ': mae, 'Mean Percentage Error': mpe,\
            'Root Mean Squared Error ':rmse, \
            #'Lag 1 Autocorrelation of Error':acf1, \
            'Correlation between the Actual and the Forecast':corr}) \
            #'Min-Max Error ':minmax})
