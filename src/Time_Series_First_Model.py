import numpy as np
import pandas as pd
import sys
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
# from statsmodels.tools.tools import add_constant
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
# from statsmodels.tsa.arima_process import ArmaProcess
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.stats.diagnostic import acorr_ljungbox

def resample_create_lag(df):
    yw = pd.DataFrame(df['cost_per_watt'].resample('W').median())
    yw['date'] = yw.index.date
    lagg_cost = (pd.concat([yw.shift(i) for i in range(4)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, 4)])).dropna()
    return lagg_cost

def ols_table(lagg_cost):
    solar_model = smf.ols('y ~ Lag1 + Lag2 + Lag3', data=lagg_cost).fit()
    print(solar_model.summary())
    return solar_model
