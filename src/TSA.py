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
    Rsquared = r2_score(data[origin],data[predict]).round(2)
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
    