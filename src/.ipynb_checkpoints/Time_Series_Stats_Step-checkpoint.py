import numpy as np
import pandas as pd
import sys
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.tools.tools import add_constant
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
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
    
    #orig = plt.plot(df, color='blue',label='Original')
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
