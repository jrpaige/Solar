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

def resample_weekly(df):
    yw = pd.DataFrame(df['cost_per_watt'].resample('W').median())
    yw['date'] = yw.index.date
    return yw

def resample_daily(df):
    yd = pd.DataFrame(df['cost_per_watt'].resample('D').median())
    yd['date'] = yd.index.date
    return yd

def ols_table(df):
    
    results = smf.ols('cost_per_watt ~ date', data=df).fit()
    solar_model = ols('cost_per_watt ~ date', data=df).fit()
    print(solar_model.summary())
    return solar_model
    
