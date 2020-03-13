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