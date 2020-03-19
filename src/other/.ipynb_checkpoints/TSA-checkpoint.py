import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import adfuller
import gc
gc.collect()
import sys, os
import warnings
warnings.filterwarnings('ignore')


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


from pandas.plotting import lag_plot

def lag_plots(df):
    plt.rcParams.update({'ytick.left' : False, 'axes.titlepad':10})
    fig, axes = plt.subplots(3, 8, figsize=(15,15), sharex=True, sharey=True, dpi=100)
    for i, ax in enumerate(axes.flatten()[:24]):
        lag_plot(df, lag=i+1, ax=ax, c='firebrick', alpha=0.5, s=3)
        ax.set_title('Lag ' + str(i+1))

    fig.suptitle('Lag Plot', y=1.15)    


from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error


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

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import math
from multiprocessing import cpu_count
from joblib import Parallel,delayed
from warnings import catch_warnings,filterwarnings


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
    
    
    
    
    
    
    
# ============= UNUSED TAKEN FROM TIME SERIES HELPER FUNCS 


    
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
    