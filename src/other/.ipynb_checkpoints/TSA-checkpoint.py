# UNUSED OF PREVIOUS VERSIONS OF CODE


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
import itertools

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



# === NEW LAG OLS PLOT ========================================= 
def smf_ols_plot():
    plt.plot(ols_predict, label='OLS preds')
    plt.plot(ols_test.y, label='actual')
    plt.legend(loc='best')
    plt.title('MSE = {}'.format(round(mean_squared_error(ols_test.y,ols_predict),5)))
    plt.show()


 # === NEW LAG LINEAR OLS PLOT =========================================    
def sm_OLS_plot():
    plt.plot(pd.DataFrame(predict)[0], label= 'OLS preds')
    plt.plot(pd.DataFrame(lag_y_test)[0], label= 'actual')
    plt.legend(loc='best')
    plt.title('MSE = {}'.format(round(mean_squared_error(lag_y_test, predict),5)))
    plt.show()