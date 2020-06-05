import numpy as np
import pandas as pd
import sys
import datetime
from datetime import datetime
from src.Prep_Class import * 

# MATH
from math import sqrt
from scipy import signal
from fireTS.models import NARX
#from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, LassoLars
from sklearn.metrics import r2_score, mean_squared_error, make_scorer, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline,  make_pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

#TIME
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.rolling import RollingOLS
from statsmodels.regression import *
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.tools import add_constant
from statsmodels.tsa import stattools
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.arima_model import *
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.holtwinters import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, arma_order_select_ic, pacf_ols, pacf
import pyramid
from pmdarima.arima import auto_arima
from sktime.forecasters import ARIMAForecaster
from sktime.highlevel.tasks import ForecastingTask
from sktime.highlevel.strategies import ForecastingStrategy
from sktime.highlevel.strategies import Forecasting2TSRReductionStrategy
from sktime.pipeline import Pipeline
from sktime.transformers.compose import Tabulariser

#VISUALIZATION 
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
plt.style.use('ggplot')

class Models():
    def __init__(self,order=(4,0,0)):
        self.self = self
        self.order = order
        
        
    def train_test(self, df):
        idx = round(len(df)*.8)
        train, test = df[:idx], df[idx:]
        return train, test
        
    def lag_train_test(self,df, Xy=True, lag_len=3):
        lag_df = (pd.concat([df.shift(i) for i in range(lag_len+1)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, lag_len+1)])).dropna() 
        idx = round(len(lag_df)*.8)
        if Xy==False:
            train, test = lag_df[:idx], lag_df[idx:]
            return train, test
        elif Xy==True:
            lag_y, lag_X  = lag_df.y, lag_df.drop(columns='y')    
            X_train, y_train, X_test, y_test = lag_X[:idx], lag_y[:idx], lag_X[idx:], lag_y[idx:]
            return X_train, y_train, X_test, y_test
        
    def multiple_regressors(self,df, lag_len=3, print_mses=True):
        X_train, y_train, X_test, y_test = self.lag_train_test(df, lag_len=lag_len)
        rf= RandomForestRegressor(n_jobs=-1).fit(X_train,y_train).predict(X_test)
        ols_lin = sm.OLS(y_train, X_train).fit().predict(X_test)  
        ols_train, ols_test= self.lag_train_test(df, lag_len=lag_len, Xy=False)
        ols_str = 'y ~ ' +" + ".join([f"Lag{i}" for i in range(1,lag_len+1)])
        ols = smf.ols(ols_str, data=ols_train).fit().predict(ols_test)
        return rf, ols_lin, ols 
    
    def regres_dfs(self, df):
        y_preds = self.lag_train_test(df)[3]
        rf, ols_lin, ols_smf = self.multiple_regressors(df, print_mses=False)
        y_preds.rename(columns={'cost_per_watt':'actual'}, inplace=True)
        y_preds['randomforest'], y_preds['olslinear'],y_preds['olssmf'] = rf, ols_lin, ols_smf
        return y_preds
    
    def formastr(self,str):
        return str.replace(" ","").replace("Regression","").lower()
    
    def regression(self, df):   
        y_preds = self.regres_dfs(df)
        y_train = self.lag_train_test(df, Xy=True)[1]
        pred_s, pred_e = y_preds.index.date[0], y_preds.index.date[-1]
        train_s, train_e = y_train.index.date[0], y_train.index.date[-1]
        model_type = ['Random Forest Regression', 'OLS Linear Regression', 'OLS smf Regression']
        return  y_preds, y_train, [train_s, train_e, pred_s, pred_e], model_type
    
    def skt_ARIMA(self,df): 
        idx = round(len(df)*.8)
        tsdf = df['cost_per_watt']
        tsdf.reset_index(drop=True, inplace=True)
        ts_train, ts_test = pd.Series([tsdf[:idx]]), pd.Series([tsdf[idx:]])
        m = ARIMAForecaster(order=self.order)
        m.fit(ts_train)
        fh = np.arange(1, (len(tsdf)-idx)+1)
        ts_y_pred = m.predict(fh=fh)
        skt_mse = m.score(ts_test, fh=fh)**2
        skt_title = f'SKT ARIMA {self.order}        MSE ={round(skt_mse,5)}'
        return ts_train, ts_test, ts_y_pred, skt_title

    def ARIMA_predict(self, df):
        atrain, atest = self.train_test(df)
        atest_s, atest_e = atest.index.date[0], atest.index.date[-1]
        atrain_s, atrain_e = atrain.index.date[0], atrain.index.date[-1]
        res = ARIMA(atrain, order=self.order).fit()
        a_pred = res.predict(atest_s, atest_e)
        arima_title = f'ARIMA {self.order}         MSE={round(mean_squared_error(atest,a_pred),5)}'
        return res, atrain, atest, arima_title, a_pred    
    
    def all_models(self,df):
        y_preds, y_train, [train_s, train_e, pred_s, pred_e], model_type = self.regression(df)
        ts_train, ts_test, ts_y_pred, skt_title = self.skt_ARIMA(df)
        res, atrain, atest, arima_title, a_pred = self.ARIMA_predict(df)
        idx = round(len(df)*.8)
        
        fig, axs = plt.subplots(5, figsize= (20,20))
        fig.tight_layout(h_pad=5)
        for i in range(3):
            exec(f"axs[{i}].plot(y_preds.{self.formastr(model_type[i])}, label= '{model_type[i]}', linewidth=2)")
            exec(f"axs[{i}].plot(y_preds.actual, label= 'Actual')")
            exec(f"axs[{i}].plot(y_train[-30:], label='Train', color='gray')")
            exec(f"axs[{i}].fill_between(y_preds.index, y_preds.{self.formastr(model_type[i])}, y_preds.actual, color='gray', alpha=.3)")
            exec(f"axs[{i}].set_title('{model_type[i]}        MSE=%s' % round(mean_squared_error(y_preds.actual, y_preds.{self.formastr(model_type[i])}),5), fontsize=18)")
            exec(f"axs[{i}].legend(loc='best')")
            exec(f"axs[{i}].set_xlim(left=y_train.index.date[-31])")

        
        axs[3].plot(a_pred, label='ARIMA Forecast')
        axs[3].plot(atest.index, atest, label='Actual')
        axs[3].plot(atrain.index[-30:], atrain[-30:], label='Train', color='gray')
        axs[3].fill_between(a_pred.index, atest.cost_per_watt.values, 0, color='gray', alpha=.3)
        axs[3].set_title(arima_title, fontsize=18)
        axs[3].legend(loc='best')
        axs[3].set_xlim(left=atrain.index.date[-31])
        
               
        ts_y_pred.plot(ax=axs[4], label='SKT ARIMA Forecast')
        ts_test.iloc[0].plot(ax=axs[4],label='Actual')
        ts_train.iloc[0][-30:].plot(ax=axs[4],label='Train', color='gray')
        axs[4].fill_between(ts_y_pred.index, ts_test.iloc[0].values, 0, color='gray', alpha=.3)
        axs[4].set_title(skt_title, fontsize=18)
        axs[4].legend(loc='best')
        #axs[4].set_ylabel('cost_per_watt')
        fig.suptitle('Forecast For:     [{}] - [{}] \n Trained On:       [{}] - [{}]\n \n \n'.format(pred_s, pred_e, train_s, train_e), y=1.05 ,verticalalignment='top', fontsize=20)
        
        plt.show()
        
        
    def show_models(self,df):
        return self.all_models(df)

    
    
if __name__ == "__main__":   
    Models().show_models(df)
    #Models(order=(0,0,0)).show_models(df)