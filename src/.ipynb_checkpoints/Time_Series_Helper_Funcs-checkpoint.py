import numpy as np
import pandas as pd
import sys
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import datetime

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit 
from math import sqrt
from statsmodels.tsa import stattools
from statsmodels.tools.tools import add_constant
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, acf
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pyramid
from pmdarima.arima import auto_arima
from scipy import signal
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6

# TS PREP ______________________________________________


def rolling_plot(df):
    ywn = pd.DataFrame(df.cost_per_watt).dropna()
    rollingmedian = ywn.rolling(window=3).median()
    rollingmean = ywn.rolling(window=3).mean()
    rollingstd = ywn.rolling(window=3).std() 
    
    orig = plt.plot(df, color='blue',label='Original')
    mean = plt.plot(rollingmean, color='green', label='Rolling Mean')
    med = plt.plot(rollingmedian, color='red', label='Rolling Median')
    std = plt.plot(rollingstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Weekly Rolling Mean, Median, & Standard Deviation with Window 3')
    plt.ylabel('Cost Per Watt $')
    plt.show()
        
def test_stationarity(df):
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(df, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
def log_ma(df):
    ywn = pd.DataFrame(df.cost_per_watt).dropna()
    rollingmedian = ywn.rolling(window=3).median()
    rollingmean = ywn.rolling(window=3).mean()
    ywn_log = np.log(ywn)
    ywn_log_minus_MA = ywn_log - rollingmedian
    return ywn_log_minus_MA


def resamp_lag_ols(df):
    y = pd.DataFrame(df.cost_per_watt)
    yw = pd.DataFrame(y['cost_per_watt'].resample('W').median())
    return yw

def create_lag(df):
    lagg_cost = (pd.concat([df.shift(i) for i in range(4)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, 4)])).dropna()
    return lagg_cost

def ols_table(lagg_cost):
    solar_model = smf.ols('y ~ Lag1 + Lag2 + Lag3', data=lagg_cost).fit()
    print(solar_model.summary())
    return solar_model

def shortened_timeline(df):
    #remove the volitle data points in the beginning. showing only data beginning 1/1/2002
    sdf = df.loc[df.index.date > datetime.date(2001,12,31)]
    syw = resamp_lag_ols(sdf)
    s_lagg_cost = create_lag(syw)
    solar_model_short = ols_table(s_lagg_cost)
    return syw
    return s_lagg_cost 

#     rolling_plot(syw)
#     plt.show()
#     test_stationarity(syw)


def series_and_lagged(series, lag=1):
    truncated = np.copy(series)[lag:]
    lagged = np.copy(series)[:(len(truncated))]
    return truncated, lagged


    

def format_list_of_floats(L):
    return ["{0:2.2f}".format(f) for f in L]
        
# CORRELATION ______________________________________________

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

def compute_autocorrelation(series, lag=1):
    series, lagged = series_and_lagged(series, lag=lag)
    return np.corrcoef(series, lagged)[0, 1]

#df = w_diff
def plot_ac_scat(df):
    fig, axs = plt.subplots(3, 3, figsize=(8, 8))

    lags = [1,2,3,4,5,6,7,8,52]

    for i, ax in zip(lags,axs.flatten()):
        series, lagged = series_and_lagged(df, lag=i)
        autocorr = compute_autocorrelation(df, lag=i)
        ax.scatter(series, lagged, alpha=0.5)
        ax.set_title("Lag {0} AC: {1:2.2f}".format(i, autocorr))

    plt.tight_layout()

    
def box_jenkins_plot(df):
    fig, axs = plt.subplots(2, figsize=(16, 6))
    plot_acf_and_pacf(df, axs)
    plt.tight_layout()
    plt.show()

def plot_acf_and_pacf(df, axs):
    """Plot the autocorrelation and partial autocorrelation plots of a series
    on a pair of axies.
    """
    _ = plot_acf(df, ax=axs[0]) #lags=lags)
    _ = plot_pacf(df, ax=axs[1]) #lags=lags)   

    
# RESIDUALS  ______________________________________________
    
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




# LINEAR  ______________________________________________
def linear_model_trend(df):
    X = add_constant(np.arange(1, len(df) + 1))
    y = df
    linear_model = sm.OLS(y, X).fit()
    linear_trend = linear_model.predict(X)
    fig, ax = plt.subplots(1, figsize=(16, 3))
    ax.plot(df.index, df)
    ax.plot(df.index, linear_trend)
    ax.set_title("Weekly Median Cost Per Watt Over Time with Trendline")
    return linear_trend
    
def lm_resids(df, linear_trend):    
    lm_residuals = pd.Series(df.cost_per_watt - linear_trend, index=df.index)
    fig, axs = plt.subplots(3, figsize=(16, 8))
    # The model predicts zero for the first few datapoints, so the residuals
    # are the actual values.
    axs[0].plot(lm_residuals.index, lm_residuals)
    plot_acf_and_pacf(lm_residuals, axs[1:])
    plt.tight_layout()

def lm_residual_model(lm_residuals):
    lm_residual_model = ARIMA(
    lm_residuals, order=( )).fit()
    
def lm_preds(df): 
    X = np.column_stack([df,
                     add_constant(np.arange(1, len(df) + 1))])
    lm_preds = pd.Series(
    linear_model.predict(X),
    index=df.index) 
    #lm_preds= lm_preds[arima_preds.index.min():]
    

    
    
# ARIMA ______________________________________________

#may add disp=-1 into .fit()
def arima_model(df):
    y_hat_avg = df.copy()
    fit1 = sm.tsa.statespace.ARIMA(df['cost_per_watt'], order=(2, 0, 4)).fit()
    y_hat_avg['ARIMA'] = fit1.predict(start="2019-1-6", end="2020-1-6", dynamic=True)
    plt.figure(figsize=(16,8))
    plt.plot(df['cost_per_watt'], label='Cost Per Watt')
    plt.plot(y_hat_avg['ARIMA'], label='ARIMA')
    plt.legend(loc='best')
    plt.show()
    model_type = 'ARIMA'
    print('RMS Score:',  rms_score(df, model_type))

def arima_pdq_get(df):
    df = np.array(df)
    print('P, D, Q parameters to use in ARIMA model =', auto_arima(df[1:]).order)
    mod = ARIMA(weekly_differences[1:], order=auto_arima(df[1:]).order)
    res = mod.fit()
    print(res.summary())
    
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
    
    print('ARIMA predict')
    ARIMAResults.predict(res,start = start_date_str, end= end_date_str, dynamic=True).plot()
    plt.show()
    
def arima_scores(res):
    print('standard errors :',res.bse)
    print('----------')
    print('pvalues :',res.pvalues)
    print('----------')
    print('residuals :',res.resid)
    print('----------')
    print('plot diagnositcs :', res.plot_diagnostics())

    
# actual = w_diff.cost_per_watt
# forecast = pd.DataFrame(all_year_preds[1:])
# forecast = forecast[0]
# forecastt =  forecast.loc[forecast.index < '2019-01-07']

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    #mins = np.amin(np.hstack([forecast[:,None], 
                              #actual[:,None]]), axis=1)
    #maxs = np.amax(np.hstack([forecast[:,None], 
                              #actual[:,None]]), axis=1)
    #minmax = 1 - np.mean(mins/maxs)             # minmax
    #acf1 = acf(fc-test)[1]                      # ACF1
    #return({'Mean Absolute Percentage Error':mape, 'Mean Error':me, 'Mean Absolute Error ': mae, 
            #'Mean Percentage Error': mpe, 'Root Mean Squared Error ':rmse, #'Lag 1 Autocorrelation of Error':acf1, 
            #'Correlation between the Actual and the Forecast':corr}) #'Min-Max Error ':minmax})
    print('Mean Absolute Percentage Error:  ', mape, '\nMean Error:                      ',me, '\nMean Absolute Error :            ', mae, 
            '\nMean Percentage Error:           ', mpe, '\nRoot Mean Squared Error :        ',rmse, #'Lag 1 Autocorrelation of Error':acf1, 
            '\nCorrelation between the \nActual and the Forecast:         ',corr) #'Min-Max Error ':minmax})    
    
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
    plt.plot(res.predict(), label='Full Predictions')
    plt.plot(w_diff, label='Weekly_Differences')
    plt.plot(pred, label = 'Future Predictions')
    plt.plot(preds, label= 'Predictions from 2016 -2020')
    #plt.plot(syw, label = 'Full Data')
    plt.legend(loc='best')
    plt.show()

#target = weekly_differences['cost_per_watt']
def see_fitted(df, target):
    plt.plot(df)
    plt.plot(yt_res.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((yt_res.fittedvalues - target)**2))

    

# OTHER MODELS  ______________________________________________
def random_forest_model():
    X = add_constant(np.arange(1, len(syw) + 1))
    y = syw
    rf = RandomForestRegressor(oob_score=True,n_jobs=-1)
    rf.fit(X,y)

    print('OOB Score: {}'.format(rf.oob_score_))
    print('r2 score on test: {}'.format(rf.score(X,y)))

    rf_predict = rf.predict(X)
    mse = mean_squared_error(y,rf_predict)
    print('MSE: {}'.format(mse))

def rms_score(df, model_type):
    '''
    calculate RMSE to check to accuracy of model on data set
    model_type = [moving_avg_forecast, Holt_linear, ARIMA, OLS, RF, Linear Regression]
    '''
    rms = sqrt(mean_squared_error(df.Count, y_hat.model_type))
    return rms

def moving_avg_model(df):
    y_hat_avg = df.copy()
    y_hat_avg['moving_avg_forecast'] = df['cost_per_watt'].rolling(3).median().iloc[-1]
    plt.figure(figsize=(16,8))
    plt.plot(df['cost_per_watt'], label='Cost Per Watt')
    plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
    plt.legend(loc='best')
    plt.show()
    model_type = 'moving_avg_forecast'
    print('RMS Score:', rms_score(df, model_type))
    
def holt_linear_model(df):
    y_hat_avg = df.copy()
    fit1 = Holt(np.asarray(df['cost_per_watt'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
    y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
    plt.figure(figsize=(16,8))
    plt.plot(df['cost_per_watt'], label='Cost Per Watt')
    plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
    plt.legend(loc='best')
    plt.show()
    model_type = 'Holt_linear'
    print('RMS Score:',  rms_score(df, model_type))

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
        Plots modelled vs fact values, prediction intervals and anomalies
    
    """
    
    prediction = model.predict(X_test)
    
    plt.figure(figsize=(15, 7))
    plt.plot(prediction, "g", label="prediction", linewidth=2.0)
    plt.plot(y_test.values, label="actual", linewidth=2.0)
    
    if plot_intervals:
        cv = cross_val_score(model, X_train, y_train, 
                                    cv=tscv, 
                                    scoring="neg_mean_squared_error")
        #mae = cv.mean() * (-1)
        deviation = np.sqrt(cv.std())
        
        lower = prediction - (scale * deviation)
        upper = prediction + (scale * deviation)
        
        plt.plot(lower, "r--", label="upper bond / lower bond", alpha=0.5)
        plt.plot(upper, "r--", alpha=0.5)
        
        if plot_anomalies:
            anomalies = np.array([np.NaN]*len(y_test))
            anomalies[y_test<lower] = y_test[y_test<lower]
            anomalies[y_test>upper] = y_test[y_test>upper]
            plt.plot(anomalies, "o", markersize=10, label = "Anomalies")
    
    error = mean_absolute_percentage_error(prediction, y_test)
    print("Mean absolute percentage error", error)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.grid(True);
    
def plotCoefficients(model):
    """
        Plots sorted coefficient values of the model
    """
    
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
    
