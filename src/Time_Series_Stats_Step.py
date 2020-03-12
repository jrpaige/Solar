import numpy as np
import pandas as pd
import sys
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tools.tools import add_constant
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6

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


def plot_acf_and_pacf(df, axs, lags=24*2):
    """Plot the autocorrelation and partial autocorrelation plots of a series
    on a pair of axies.
    """
    _ = plot_acf(df, ax=axs[0], lags=lags)
    _ = plot_pacf(df, ax=axs[1], lags=lags)

#def plot_moving_average(df, window, plot_intervals=False, scale = 

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

    
def arima_model(df):
    y_hat_avg = df.copy()
    fit1 = sm.tsa.statespace.ARIMA(df['cost_per_watt'], order=(2, 1, 4)).fit()
    y_hat_avg['ARIMA'] = fit1.predict(start="2019-1-6", end="2020-1-6", dynamic=True)
    plt.figure(figsize=(16,8))
    plt.plot(df['cost_per_watt'], label='Cost Per Watt')
    plt.plot(y_hat_avg['ARIMA'], label='ARIMA')
    plt.legend(loc='best')
    plt.show()
    model_type = 'ARIMA'
    print('RMS Score:',  rms_score(df, model_type))


