import numpy as np
import pandas as pd
import sys
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import datetime
# from statsmodels.tools.tools import add_constant
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
# from statsmodels.tsa.arima_process import ArmaProcess
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.stats.diagnostic import acorr_ljungbox




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

#     rolling_plot(syw)
#     plt.show()
#     test_stationarity(syw)