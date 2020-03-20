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

#VISUALIZATION 
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
plt.style.use('ggplot')



# === TS PREP =========================================
def time_frame(df):
    '''
    removes the volatile data points in the beginning. 
    df is now only data beginning 1/1/2002
    returns: 
        y : shortened timeline to remove volatility in the beginning\ 
        of the data and resamples the data weekly by median
    '''
    sdf = df.loc[df.index > '2001-12-31']
    y = pd.DataFrame(sdf.cost_per_watt)
    y = pd.DataFrame(y['cost_per_watt'].resample('W').median())
    return y

def rolling_plot(df):
    '''
    Plots original data, and data with a rolling 
    window of 3 using mean, median, and standard deviation
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
    Performs Dickey-Fuller test 
    '''
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(df, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                             'p-value','#Lags Used',
                                             'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
def autocor_plots(df):
    fig, ax = plt.subplots(2, figsize=(15,7))
    sm.graphics.tsa.plot_acf(df, ax=ax[0])
    sm.graphics.tsa.plot_pacf(df, ax=ax[1])
    plt.show()   

def get_differences(df):
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
    Tests data for stationarity 
    Should pass in weekly_differences
    Prints result
    '''
    test = sm.tsa.stattools.adfuller(df[1:])
    print("ADF p-value: {0:2.2f}".format(test[1]))
    if test[1] < 0.51:
        print('Achieved stationarity! Reject ADF H0.')
    else:
        print('Time Series is not stationary. Fail to reject ADF H0')

        
# === CORRELATION =========================================
 
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
    Plots autocorrelation scatter plot
    Should like use weekly differences array
    '''
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))

    lags = [1,2,3,4,5,6,7,26,52]

    for i, ax in zip(lags,axs.flatten()):
        series, lagged = series_lag(df, lag=i)
        autocorr = compute_autocorrelation(df, lag=i)
        ax.scatter(series, lagged, alpha=0.5)
        ax.set_title("Lag {0}".format(i))
        #ax.set_title("Lag {0} AC: {1:2.2f}".format(i, autocorr))
        plt.tight_layout()

    
# === COEFFICIENTS =========================================
def plotCoefficients(model):
    ''' 
    Plots sorted coefficient values of the model
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
    Returns simple forecast based on shifted data of 1 week and 3 week lags
    '''
    forcst = pd.DataFrame(df.loc[df.index.year>2017])
    forcst['cost_1weekago'] = df['cost_per_watt'].shift(1)
    forcst['cost_3weeksago'] = df['cost_per_watt'].shift(3)
    print(precision(forcst,'cost_1weekago','cost_per_watt'), precision(forcst,'cost_3weeksago','cost_per_watt'))

def precision(data,forecast,origin):
    '''
    Returns MAE, MSE, and RMSE scoring for simple move model
    '''
    MAE = mean_absolute_error(data[origin],data[forecast]).round(2)
    MSE = mean_squared_error(data[origin],data[forecast]).round(2)
    RMSE = np.sqrt(mean_squared_error(data[origin],data[forecast])).round(2)
    print(forecast,'[\n MAE:',MAE, '\n MSE:',MSE, '\n RMSE:',RMSE,']')
    
    
def ARMA_model(df, begin_year):
    '''
    Forecasts beginning from begin year through end of df
    '''
    trunc_df = df.loc[df.index.year <begin_year][1:]   
    mod1 = ARMA(trunc_df, order=(1,0), freq='W', )
    res1 = mod1.fit()
    res1.plot_predict(end=str(df.index.year[-1]))
    return res1
        
def ARMA_plots(df):
    '''
    Plot + Confidence Interval + Model Summary
    Plot1 = ARMA forecasted data through 2030 based on data 
        through end of 2016 using .plotpredict
    Plot2 = ARMA forecasted data through 2030 based on full 
        data using .plotpredict
    Df should be stationary, likely the differenced values.
    Prints confidence intervals and model summary of each ARMA model
    Returns models from each 
    '''
    ts_diff = df[1:] #remove the NaN
    first_14 = ts_diff.loc[ts_diff.index.year <2017]
    last_few = ts_diff.loc[ts_diff.index.year >2016]
    
    #Plot1
    mod1 = ARMA(first_14, order=(1,0), freq='W', )
    res1 = mod1.fit()
    res1.plot_predict(end='2030', alpha=0.5)
    plt.title('ARMA Forecast through 2030 on Data from 2002-2016')
    plt.show()
    print('Confidence Intervals for ARMA Forecast through 2030 on Data from 2002-2016', res1.conf_int())
    print(res1.summary())

  
    #Plot2
    mod2 = ARMA(ts_diff, order=(1,0), freq='W', )
    res2 = mod2.fit()
    res2.plot_predict(end='2030', alpha=0.5)
    plt.title('ARMA Forecast through 2030 on Full Data ')
    plt.show()
    print('Confidence Intervals for ARMA Forecast through 2030 on Full Data', res2.conf_int())
    print(res2.summary())
    return res1, res2
    
# === ARIMA TIME SERIES MODEL=========================================

# === ARIMA PARAMS
def auto_arima_pdq(df):
    '''
    Auto ARIMA to obtain best parameters for data
    '''
    df = np.array(df)
    print('P, D, Q parameters to use in ARIMA model =', auto_arima(df[1:]).order)
    
def evaluate_arima_model(X, arima_order):
    '''
    Splits data into training/test
    Pushes through ARIMA models
    Obtains and returns MSE 
    '''
    # prepare training dataset
    train_size = int(len(X) * 0.7)
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
    
def evaluate_models(dataset):
    '''
    Uses various p,d,qs within below range
    Tests out each combination 
    Returns params with the best cfg + best MSE
    use [evaluate_models(df.values.dropna(), p_values, d_values, q_values)]  
    '''
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
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA %s MSE=%.3f' % (best_cfg, best_score))


# === ARIMA MODELS   
def basic_arima_model(df):    
    mod = ARIMA(df[1:], order=auto_arima(df[1:]).order)
    res = mod.fit()
    print(res.summary())

def arima_model_forecast(df):
    '''
    Forecasts for 2016-2019
    '''
    st_date = '2016-01-10'
    y_hat_avg = df[1:df.index.get_loc(st_date)]
    new_dates = pd.DataFrame(pd.date_range(start='2016-01-10', end='2019-01-06', freq='W'))
    new_dates['cost_per_watt'] = 0
    new_dates.set_index(0, drop=True, inplace=True)
    y_hat_avg = pd.concat([y_hat_avg, new_dates])
    fit1 = ARIMA(y_hat_avg['cost_per_watt'], order=(auto_arima(df[1:]).order)).fit()
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
    print('standard errors :',res.bse)
    print('----------')
    print('pvalues :',res.pvalues)
    print('----------')
    print('residuals :',res.resid)
    print('----------')
    print('plot diagnositcs :', res.plot_diagnostics())

    
# === VALIDATE/SCORE ===============================================  
 
    #gridsearch(ETS)===
def walk_forward_validation(data, n_test, cfg):
    '''
    walk-forward validation for univariate data
    #forecst = list() 
    '''
    train, test = train_test_split(data, n_test)
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
    

def score_model(data, n_test, cfg, debug=False):
    '''
    Score a model, return None on failure
    '''
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


def grid_search(data, cfg_list, n_test, parallel=True):
    '''
    Grid search for configs
    '''
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores   
        
#scores = grid_search(data, cfg_list, n_test)

def forecast_accuracy(forecast, actual):
    '''
    Prints MAPE, ME, MAE, MPE, RMSE, and Corr(Actual,Forecast)
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
    
                     
                        
def RMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))                                   
                        
def rms_score(df, model_type):
    '''
    Calculates RMSE to check accuracy of model on data set
    model_type = [moving_avg_forecast, Holt_linear, ARIMA, 
        OLS, RF, Linear Regression]
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
