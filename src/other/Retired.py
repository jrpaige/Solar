# UNUSED OF PREVIOUS VERSIONS OF CODE

import pandas as pd
import numpy as np
import datetime
import sys
import gc
gc.collect()
import sys, os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import itertools

# MATH
from math import sqrt
from scipy import signal
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

from sklearn.model_selection import TimeSeriesSplit, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline,  make_pipeline, FeatureUnion
# DEPRECIATED from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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
import seaborn as sns



from pandas.plotting import lag_plot
import statsmodels.api as sm
import math
from multiprocessing import cpu_count
from joblib import Parallel,delayed
from warnings import catch_warnings,filterwarnings



def station_plots(df):
    '''
    check for stationarity through acf and pacf plots
    '''
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(211)
    plot_acf(df.cost_per_watt, lags=3, ax=ax1)
    ax2 = fig.add_subplot(212)
    plot_pacf(df.cost_per_watt,lags=3,ax=ax2)
    plt.show()
    
    
def plot_acf_and_pacf(df, axs):
    """
    *** For use in get_differences function***
    Plot the autocorrelation and partial autocorrelation plots of a series
    on a pair of axies.
    """
    _ = plot_acf(df, ax=axs[0]) #lags=lags)
    _ = plot_pacf(df, ax=axs[1]) #lags=lags) 
    
def get_differences(df):
    weekly_differences = df.diff(periods=1)
    fig, axs = plt.subplots(3, figsize=(16, 8))
    axs[0].plot(weekly_differences.index, weekly_differences)
    # The first entry in the differenced series is NaN.
    plot_acf_and_pacf(weekly_differences[1:], axs[1:])
    plt.tight_layout()
    test = sm.tsa.stattools.adfuller(weekly_differences[1:])
    print("ADF p-value: {0:2.2f}".format(test[1]))
    if test[1] < 0.51:
        print('Achieved stationarity! Reject ADF H0.')
    else:
        print('Time Series is not stationary. Fail to reject ADF H0')
    return weekly_differences




def lag_plots(df):
    plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})
    fig, axes = plt.subplots(3, 8, figsize=(15,15), sharex=True, sharey=True, dpi=100)
    for i, ax in enumerate(axes.flatten()[:24]):
        lag_plot(df, lag=i+1, ax=ax, c='firebrick', alpha=0.5, s=3)
        ax.set_title('Lag ' + str(i+1))

    fig.suptitle('Lag Plot', y=1.15)    



def precision(data,predict,origin):
    MAE = mean_absolute_error(data[origin],data[predict]).round(2)
    MSE = mean_squared_error(data[origin],data[predict]).round(2)
    RMSE = np.sqrt(mean_squared_error(data[origin],data[predict])).round(2)
    #MSLE = mean_squared_log_error(data[origin],data[predict]).round(2)
    #RMLSE = np.sqrt(mean_squared_log_error(data[origin],data[predict])).round(2)
    MAPE = round(np.mean((abs(data[origin]-data[predict]))/(data[origin]))*100,2)
    MAPE_adjust = round(np.mean((abs(data[origin]-data[predict]))/(data[origin]+1))*100,2)
    sMAPE = round(np.mean(200*(abs(data[origin]-data[predict]))/(data[origin]+data[predict])),2)
    print(predict,'[Rsquared:', Rsquared, '\n MAE:',MAE, '\n MSE:',MSE, '\n RMSE:',RMSE,\
                     '\n MAPE:',MAPE,'\n MAPE_adjust:',MAPE_adjust,'\n sMAPE:',sMAPE,']')




def measure_rmse(actual, predicted):
	return math.sqrt(mean_squared_error(actual, predicted))

def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]


#gridsearch(ETS)===============================================
# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = exp_smoothing_forecast(history, cfg)# store forecast in list of predictions		
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error
    
# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
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

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
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
    use weekly differences array
    '''
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))

    lags = [1,2,3,4,5,6,7,26,52]

    for i, ax in zip(lags,axs.flatten()):
        series, lagged = series_lag(df, lag=i)
        autocorr = compute_autocorrelation(df, lag=i)
        ax.scatter(series, lagged, alpha=0.5)
        ax.set_title("Lag {0}".format(i))
#        ax.set_title("Lag {0} AC: {1:2.2f}".format(i, autocorr))

    plt.tight_layout()
    
    
    
def format_list_of_floats():
    return ["{0:2.2f}".format(f) for f in L]
    
    
    
    
# ============= UNUSED TAKEN FROM TIME SERIES HELPER FUNCS 



     
        
        
        
# === RESIDUALS =========================================
    
def residual_plot(ax, x, y, y_hat, n_bins=50):
    residuals = y_hat - y
    ax.axhline(0, color="black", linestyle="--")
    ax.scatter(x, residuals, color="grey", alpha=0.5)
    ax.set_ylabel("Residuals ($\hat y - y$)")

def plot_many_residuals(df, var_names, y_hat, n_bins=50):
    fig, axs = plt.subplots(len(var_names), figsize=(12, 3*len(var_names)))
    for ax, name in zip(axs, var_names):
        x = df[name]
        residual_plot(ax, x, df['cost_per_watt'], y_hat)
        ax.set_xlabel(name)
        ax.set_title("Model Residuals by {}".format(name))
    return fig, axs





    
#df = weekly_differences
def arima_coefs(df):
    mod = ARIMA(weekly_differences[1:], order=auto_arima(df[1:]).order)
    res = mod.fit()
    print("ARIMA(2, 0, 4) coefficients from model:\n  Intercept {0:2.2f}\n  AR {1}".format(
    res.params[0], 
        format_list_of_floats(list(res.params[1:]))))
    
#start_date_str = '2019-01-06'   
def arima_preds(df, start_date_str):
    arima_preds = lm_residual_model.predict(df.index.max(), pd.to_datetime(start_date_str).tz_localize('UTC'), dynamic=True)
    return arima_preds    

def arima_res_ind(res):
    print('Variance/Covariance Matrix', ARIMAResults.cov_params(res))

#52 steps
#start = '2018-12-09', end= '2020'
def arima_forecast_predict_plot(res, steps, start_date_str, end_date_str):
    print('ARIMA forecast') 
    ARIMAResults.forecast(res, steps =steps,).plot()
    plt.title('ARIMA forecast for {} steps'.format(steps))
    plt.show()
    
    print('ARIMA forecast')
    ARIMAResults.predict(res,start = start_date_str, end= end_date_str, dynamic=True).plot()
    plt.show()
    
    
    
def auto_regressive_process(size, coefs, init=None):
    """Generate an autoregressive process with Gaussian white noise.  The
    implementation is taken from here:
    
      http://numpy-discussion.10968.n7.nabble.com/simulate-AR-td8236.html
      
    Exaclty how lfilter works here takes some pen and paper effort.
    """
    coefs = np.asarray(coefs)
    if init == None:
        init = np.array([0]*len(coef))
    else:
        init = np.asarray(init)
    init = np.append(init, np.random.normal(size=(size - len(init))))
    assert(len(init) == size)
    a = np.append(np.array([1]), -coefs)
    b = np.array([1])
    return pd.Series(signal.lfilter(b, a, init))

#start = '2018-12-09', end= '2020'
# start = '2016-01-03'
def see_preds_plot(res, start_date_str, end_date_str):
    pred = res.predict(start = start_date_str, end= end_date_str)
    all_year_preds = res.predict(end = end_date_str)
    last_four_preds = res.predict(start= start_date_str, end= end_date_str)
    plt.figure(figsize=(16,8))
    plt.plot(res.predict(), label='Full Forecast')
    plt.plot(w_diff, label='Weekly_Differences')
    plt.plot(pred, label = 'Future Forecast')
    plt.plot(preds, label= 'Forecast for 2016 -2020')
    #plt.plot(syw, label = 'Full Data')
    plt.legend(loc='best')
    plt.show()

#target = weekly_differences['cost_per_watt']
def see_fitted(df, target):
    plt.plot(df)
    plt.plot(yt_res.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((yt_res.fittedvalues - target)**2))
    
# MOVING AVG __ 
def moving_avg_model(df):
    y_hat_avg = df.copy()
    y_hat_avg['moving_avg_forecast'] = df['cost_per_watt'].rolling(3).median().iloc[-1]
    plt.figure(figsize=(16,8))
    plt.plot(df['cost_per_watt'], label='Cost Per Watt')
    plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
    plt.legend(loc='best')
    plt.show()
    model_type = 'moving_avg_forecast'
    print('RMS Score:', np.sqrt(mean_squared_error(df, model_type)))
    
    
    
    num = 3    
tscv = TimeSeriesSplit(n_splits=num)

def timeseries_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test


def plotModelResults(model, X_train, X_test, plot_intervals=False, plot_anomalies=False, scale=1.96):
    """
    Plots modeled vs fact values, forecast intervals and anomalies
    
    """
    
    forcst = model.predict(X_test)
    plt.figure(figsize=(15, 7))
    plt.plot(forcst, "g", label="forecast", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, 
                                    cv=tscv, 
                                    scoring="neg_mean_squared_error")
        #mae = cv.mean() * (-1)
        deviation = np.sqrt(cv.std())
        
        lower = forcst - (scale * deviation)
        upper = forcst + (scale * deviation)
        
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    error = mean_absolute_percentage_error(forecast, y_test)
    print("Mean absolute percentage error", error)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    
    

def spec_arma_plots(df):
    '''
    ==Parameters==
    df should be stationary, likely the differenced values.
    
    ==Plots Included:==
    Plot1 = ARMA forecasted data through 2030 based on data 
        through end of 2016 using .plotpredict
    Plot2 = ARMA forecasted data through 2030 based on full 
        data using .plotpredict
   
    ==Returns==
    Plots
    Confidence Intervals 
    Model Summary
    Model variable
    '''
    ts_diff = df[1:] #remove the NaN
    first_14 = ts_diff.loc[ts_diff.index.year <2017]
    last_few = ts_diff.loc[ts_diff.index.year >2016]
    
    #Plot1
    mod1 = ARMA(first_14, order=(8,0), freq='W', )
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
    print(precision(forcst,'cost_1weekago','cost_per_watt'), precision(forcst,'cost_3weeksago','cost_per_watt'))

def precision(data,forecast,origin):
    '''
    Returns MAE, MSE, and RMSE scoring for simple move model
    '''
    MAE = mean_absolute_error(data[origin],data[forecast]).round(2)
    MSE = mean_squared_error(data[origin],data[forecast]).round(2)
    RMSE = np.sqrt(mean_squared_error(data[origin],data[forecast])).round(2)
    print(forecast,'\n MSE :',MSE)
    print(forecast,'[\n MAE:',MAE, '\n MSE:',MSE, '\n RMSE:',RMSE,']') 
        
        
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
        
   




# === STATIONARY REGRESSION =========================================   


def stat_lag_ols_model(df):    
    '''
    ==Function==
    OLS Regression for differenced/stationary data
    Creates lag table and processes through OLS
    ==Returns==
    |ols_model| : ols of 3 lagged colummns on the differenced data]
    |ols_trend| : df of fitted values for the differenced data]
    '''
    df = df[1:]
    tslag_cost = (pd.concat([df.shift(i) for i in range(4)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, 4)])).dropna()
    tsols_model = smf.ols('y ~ Lag1 + Lag2 + Lag3', data=tslag_cost).fit() 
    tsols_trend = tsols_model.fittedvalues
    return tsols_model, tsols_trend
    
def stat_linear_ols_model(df):
    '''
    ==Function==
    Linear Regression for differenced/stationary data
    Creates X & y
    Plots linear regression line
    ==Returns== 
    |linear_model| : model of ols on differenced data]
    |linear_trend| : df of fitted values for differenced data] 
    '''
    df = df[1:]
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    tslinear_model = sm.OLS(y, X).fit()
    tslinear_trend = tslinear_model.predict(X)
    return tslinear_model ,tslinear_trend

    
def stat_randomforest_model(df):
    '''
    ==Function==
    Random Forest Regressor for differenced/stationary data
    Uses simple Random Forest Regressor to forecast
    ==Returns==
    |rf_model| : the model of the rf regressor on the differenced data]
    |rf_trend| : df of fitted values for the differenced data]        
    '''
    df = df[1:]
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    tsrf_model = RandomForestRegressor(n_jobs=-1).fit(X,y)
    tsrf_trend = tsrf_model.predict(X)
    return tsrf_model,tsrf_trend

def stat_score_table(df, tsols_model, tslinear_model, tsrf_model):
    '''
    ==Function==
    Specifically for after using differenced/stationary data
    
    ==Returns==
    Table with MSE scores for each regression model 
    '''
    df = df[1:]
    tsrf_trend = tsrf_model.predict(add_constant(np.arange(1,len(df)+ 1)))
    tsmodels = ['OLS', 'LINEAR', 'RF',]
    tsreg_scores = pd.DataFrame(tsmodels)
    tsreg_scores.rename(columns={0:'Models'}, inplace=True)
    tsreg_scores.set_index('Models', drop=True, inplace= True)
    #tsreg_scores['MAE'] = [mean_absolute_error(df[3:], tsols_model.fittedvalues), mean_absolute_error(df, tslinear_model.fittedvalues), mean_absolute_error(df,tsrf_trend)]
    tsreg_scores['MSE'] = [mean_squared_error(df[3:], tsols_model.fittedvalues), mean_absolute_error(df, tslinear_model.fittedvalues), mean_squared_error(df,tsrf_trend)]
    #tsreg_scores['RMSE'] = [np.sqrt(tsreg_scores.MSE[0]), np.sqrt(tsreg_scores.MSE[1]), np.sqrt(tsreg_scores.MSE[2])]
    #ols_df, lin_df, rf_df = pd.DataFrame(tsols_model.fittedvalues), pd.DataFrame(tslinear_model.fittedvalues), pd.DataFrame(tsrf_trend)
    #tsreg_scores['P_VALUE'] = [ adfuller(ols_df, autolag='AIC')[1],adfuller(lin_df, autolag='AIC')[1], adfuller(rf_df, autolag='AIC')[1]]   
    return tsreg_scores















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
    
    
    
    
    
    
    
    tsdf = df['cost_per_watt'].dropna()
    idx = round(len(tsdf) * .8)
    tsdf.reset_index(drop=True, inplace=True)
    train = pd.Series([tsdf[:idx]])
    test = pd.Series([tsdf[idx:]])
    m = ARIMAForecaster(order=order)
    m.fit(train)
    fh = np.arange(1, (len(tsdf)-idx))
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

    
    
        df = df.dropna()
    if use_years == True:
        train = df[:len(df) - (52*years_off)]
        test = df[len(df) - (52*years_off):]
        pred = ARMA(train, order=order, freq='W').fit().predict(start=test.index.date[0], end=test.index.date[-1])
    else:
        df = df['cost_per_watt'].dropna()
        idx = round(len(df) * .8)
        train= df[:idx]
        test = df[idx:]
        order=order
        pred = ARMA(train, order, freq='W').fit().predict(start=test.index.date[0],end=test.index.date[-1])
    
    if plot==True:
        plot_arma(test, pred,train, order)
    else:
        print('ARMA Order Used: {}'.format(order))
        print('MSE:',round(mean_squared_error(test, pred),5))
        
        
        
        
        
        
        
        

# === INITIAL REGRESSION MODELS =========================================
        
        def reg_test_train(df):
    train, test, = train_test_split(df ,test_size=0.2)
    return train, test
    
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
    ols_model = smf.ols('y ~ Lag1 + Lag2 + Lag3', data=lag_cost_).fit() 
    ols_trend = ols_model.fittedvalues
    return ols_model, ols_trend

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

    
    
# need to re-evalute RF code. should not have constant
# need to split into train test
def randomforest_model(df):
    '''
    ==Function==
    Uses simple Random Forest Regressor to forecast
   
    ==Returns==
    |rf_model| : the model of the rf regressor]
    |rf_trend| : df of fitted values]
    '''
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    rf_model = RandomForestRegressor(n_jobs=-1).fit(X,y)
    rf_trend = rf_model.predict(X)
    return rf_model,rf_trend

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
    #reg_scores['MAE'] = [mean_absolute_error(df[3:], ols_model.fittedvalues), mean_absolute_error(df, linear_model.fittedvalues), mean_absolute_error(df,rf_trend)]
    reg_scores['MSE'] = [round(mean_squared_error(df[3:], ols_model.fittedvalues),5), round(mean_absolute_error(df, linear_model.fittedvalues),5), round(mean_squared_error(df,rf_trend),5)]
    #reg_scores['RMSE'] = [np.sqrt(reg_scores.MSE[0]), np.sqrt(reg_scores.MSE[1]), np.sqrt(reg_scores.MSE[2])]
    #ols_df, lin_df, rf_df = pd.DataFrame(ols_model.fittedvalues), pd.DataFrame(linear_model.fittedvalues), pd.DataFrame(rf_trend)
    #reg_scores['P_VALUE'] = [ adfuller(ols_df, autolag='AIC')[1],adfuller(lin_df, autolag='AIC')[1], adfuller(rf_df, autolag='AIC')[1]]   
    return reg_scores
    
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
    
    
    
    
# === CREATE MATRIX OF PAIRED P & Q  =========================================            


def pdq_matrix(df)
p_min = 0
d_min = 0
q_min = 0
p_max = 11
d_max = 0
q_max = 6

# Initialize a DataFrame to store the results
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

for p,d,q in itertools.product(range(p_min,p_max+1),
                               range(d_min,d_max+1),
                               range(q_min,q_max+1)):
    if p==0 and d==0 and q==0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
    
    try:
        model = smt.SARIMAX(ts_train, order=(p, d, q),
                               enforce_stationarity=True,
                               enforce_invertibility=False,seasonal_order=None
                              )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.mse
        
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)
    
    
    
    
    
    
    
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
    if order=None:
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
    
    
# === RANDOM FOREST =========================================    
    
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





# CREATED NEW OLS MODELS
    
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



# === OLD TT SPLIT =========================================
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
#==============================================================   





    
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
    
    
    
    
    
    
    
    # === MAPE, ME, MAE, MPE, RMSE, CORRELATION ===============================================  
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

# === MAPE ===============================================  
def mean_average_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true-y_pred)/ y_true)) * 100

# === RMSE ===============================================            
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))                                   

# === RMS ===============================================            
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




# UNKNOWN
#ARMA(train,order).fit().predict(start=test.index.date[0],end=test.index.date[-1])

    



# === ROLLING OLS =========================================
def rolling_ols(df):  #not helpful
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    rolols_model = RollingOLS(y, X, window=3).fit()
    #rolols_trend = rolols_model.predict(X)
    return rolols_model


# FOR WHEN PLOTTING ALL 6 REGRESSION FUNCTIONS
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

    
    
    
    
# === SIMPLE LAG  =========================================
def simple_use_lag(df):
    lag_len=3
    lag_df = (pd.concat([df.shift(i) for i in range(lag_len+1)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, lag_len+1)])).dropna() 
    y = lag_df.pop('y')
    X = lag_df
    print('Lag of 1 Week MSE: ', round(mean_squared_error(X.Lag1, y),4))
    print('Lag of 2 Weeks MSE:', round(mean_squared_error(X.Lag2, y),4))
    print('Lag of 3 Weeks MSE:', round(mean_squared_error(X.Lag3, y),4)) 
    
    
              
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



# === PLOT TS COEFFICIENTS =========================================        
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

#



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


#POTENTIALLY REMOVE
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