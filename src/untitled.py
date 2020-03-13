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
import pyramid
from pmdarima.arima import auto_arima

def box_jenkins_plot(df):
    fig, axs = plt.subplots(2, figsize=(16, 6))
    plot_acf_and_pacf(df, axs)
    plt.tight_layout()
    plt.show()


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

def arima_pdq_results(df):
    print('P, D, Q parameters to use in ARIMA model =', auto_arima(df[1:]).order)
    mod = ARIMA(weekly_differences[1:], order=auto_arima(df[1:]).order)
    res = mod.fit()
    print(res.summary())
    
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
    
def plot_acf_and_pacf(df, axs):
    """Plot the autocorrelation and partial autocorrelation plots of a series
    on a pair of axies.
    """
    _ = plot_acf(df, ax=axs[0]) #lags=lags)
    _ = plot_pacf(df, ax=axs[1]) #lags=lags)    
    
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