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




class Run_Models():
    """
    Runs regression and arima models
    """
    
    def __init__(self,df):
        self.self = self
        self.df = df
        
    def train_test(self, Xy=True, lag_len=4):
        df = self.df
        lag_df = (pd.concat([df.shift(i) for i in range(lag_len+1)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, lag_len+1)])).dropna() 
        idx = round(len(lag_df)*.8)
        if Xy==False:
            train, test = lag_df[:idx], lag_df[idx:]
            return train, test
        elif Xy==True:
            lag_y, lag_X  = lag_df.y, lag_df.drop(columns='y')    
            X_train, y_train, X_test, y_test = lag_X[:idx], lag_y[:idx], lag_X[idx:], lag_y[idx:]
            return X_train, y_train, X_test, y_test

        
    def multiple_regressors(self, lag_len=3, print_mses=True):
        '''
        ==Function==
        Uses train_test_xy(df) to split into (X/y)train/(X/y)test sets

        Runs data through 
        - Random Forest Regressor
        - sm OLS Linear Regression
        - smf ols Regression

       if print_mses==True:
            ==Prints==
            MSE scores for each 
            default=True

        ==Returns==
        rf, ols_lin, ols models
        '''
        df = self.df
        X_train, y_train, X_test, y_test = self.train_test(lag_len=lag_len)
        rf= RandomForestRegressor(n_jobs=-1).fit(X_train,y_train).predict(X_test)
        ols_lin = sm.OLS(y_train, X_train).fit().predict(X_test)  
        ols_train, ols_test= train_test(lag_len=lag_len, Xy=False)
        ols_str = 'y ~ ' +" + ".join([f"Lag{i}" for i in range(1,lag_len+1)])
        ols = smf.ols(ols_str, data=ols_train).fit().predict(ols_test)
        return rf, ols_lin, ols 

    def regres_dfs(self):
        '''
        ==Function==
        Creates a new df with y_test values and forecasted values for all regression models

        ==Uses== 
        |train_test_lag| from Regression_Helper_Funcs
        |multiple_regressors| from Regression_Helper_Funcs

        ==Returns==
        |y_preds| : new df
        '''
        df = self.df
        y_preds = self.train_test(df)[3]
        rf, ols_lin, ols_smf = self.multiple_regressors(df, print_mses=False)
        y_preds.rename(columns={'cost_per_watt':'actual'}, inplace=True)
        y_preds['randomforest'] = rf
        y_preds['olslinear'] = ols_lin
        y_preds['olssmf'] = ols_smf
        return y_preds
    
    def forma(str):
        return str.replace(" ","").lower()
    
    def regression(self):    
        '''
        completes all prep and outputs regression results
        returns df and stationary df
        '''
        df = self.df
        y_preds = regres_dfs(df)
        y_train = train_test(df, Xy=True)[1]
        fig, axs = plt.subplots(3, figsize= (20,15), constrained_layout=True)
        pred_s, pred_e = y_preds.index.date[0], y_preds.index.date[-1]
        train_s, train_e = y_train.index.date[0], y_train.index.date[-1]
        nl = '\n'
        def formastr(str):
            return str.replace(" ","").lower()

        model_type = ['Random Forest', 'OLS Linear', 'OLS smf']
        fig.suptitle(' Regression Models \n Forecast For:     [{}] - [{}] \n Trained On:       [{}] - [{}]\n'.format(pred_s, pred_e, train_s, train_e), fontsize=20)

        for i in range(3):
            exec(f"axs[{i}].plot(y_preds.actual, label= 'Actual')")
            exec(f"axs[{i}].plot(y_preds.{formastr(model_type[i])}, label= '{model_type[i]}', linewidth=2)")
            exec(f"axs[{i}].plot(y_train[-30:], label='Train', color='gray')")
            exec(f"axs[{i}].fill_between(y_preds.index, y_preds.{formastr(model_type[i])}, y_preds.actual, color='gray', alpha=.3)")
            exec(f"axs[{i}].set_title('{model_type[i]}        MSE=%s' % round(mean_squared_error(y_preds.actual, y_preds.{formastr(model_type[i])}),5), fontsize=18)")
            exec(f"axs[{i}].legend(loc='best')")
            exec(f"axs[{i}].set_xlim(left=y_train.index.date[-31])")
        plt.show()

        
    def show_models(self):
        return self.regression()
    
if __name__ == "__main__":   
    Run_Models(df).show_models()









    
# # =============================================================================
# # TRAIN TEST SPLITS  
# # =============================================================================


# def train_test(df, lag=True, lag_len=4, Xy=True):
#     '''
#     ==Function==
#     Creates a new df with 3 lagged columns
#     Splits data into X_train, y_train, X_test, y_test sets
#         -OR- into train and test sets 
    
#     ==Parameters==
#     |lag|     : if lag=True:
#                   will return dfs with lags
#                 else:
#                   will return df with only single cost_per_watt values
#                 default= True
#     |lag_len| : include how many lags to include in data
#                 default is 4
#     |xy|      : if True, splits data into X & y sets and again into train & test sets
#                 if False, splits data into train & test sets
#                  default = True
            
#     ==Returns==
#     |X_train| = first 80% of df's lagged data 
#     |y_train| = first 80% of y values
#     |X_test|  = last 20% of df's lagged data
#     |y_test|  = last 20% of df's y values
#     -OR-
#     |train| = first 80% of df's data 
#     |test| = last 20% of df's data
#     '''
#     lag_df = (pd.concat([df.shift(i) for i in range(lag_len+1)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, lag_len+1)])).dropna() 
#     idx = round(len(lag_df)*.8)
#     if lag==True:
#         if Xy==False:
#             train, test = lag_df[:idx], lag_df[idx:]
#             return train, test
#         elif Xy==True:
#             lag_y, lag_X  = lag_df.y, lag_df.drop(columns='y')    
#             X_train, y_train, X_test, y_test = lag_X[:idx], lag_y[:idx], lag_X[idx:], lag_y[idx:]
#             return X_train, y_train, X_test, y_test
#     if lag==False:
#         train, test = df[:idx], df[idx:]
#         return train, test
    
    
    
    
# # =============================================================================
# # MODEL REGRESSION PLOTS AND PREP 
# # =============================================================================
    
#  # === Multiple Regressions =========================================    
# def multiple_regressors(df, lag_len=4, print_mses=True):
#     '''
#     ==Function==
#     Uses train_test_xy(df) to split into (X/y)train/(X/y)test sets
    
#     Runs data through 
#     - Random Forest Regressor
#     - Linear Regression
#     - Bagging Regressor
#     - AdaBoost Regressor
#     - sm OLS Linear Regression
#     - smf ols Regression
   
#    if print_mses==True:
#         ==Prints==
#         MSE scores for each 
#         default=True
    
#     ==Returns==
#     rf, lr, br, abr, ols_lin, ols models
#     '''
#     X_train, y_train, X_test, y_test = train_test(df, lag_len=lag_len)
#     rf= RandomForestRegressor(n_jobs=-1).fit(X_train,y_train).predict(X_test)
#     ols_lin = sm.OLS(y_train, X_train).fit().predict(X_test)  
#     ols_train, ols_test= train_test(df, lag_len=lag_len, Xy=False)
#     ols_str = 'y ~ '
#     for i in range(1, lag_len+1):
#         ols_str+= ' Lag{} +'.format(i)
#     ols_str = ols_str.rstrip(' +')
#     ols = smf.ols(ols_str, data=ols_train).fit().predict(ols_test)
#     if print_mses == True:
#         print(' ---- MSE Scores ----'.center(31))
#         print('Random Forest Regressor  ', round(mean_squared_error(y_test, rf),5))
#         print('sm OLS Linear            ', round(mean_squared_error(y_test, ols_lin),5))

#         print('smf ols                  ', round(mean_squared_error(ols_test.y,ols),5))
#         return rf, ols_lin, ols
#         # removed br, abr
#     else:
#         return rf, ols_lin, ols 

    
# def regres_dfs(df):
#     '''
#     ==Function==
#     Creates a new df with y_test values and forecasted values for all regression models
    
#     ==Uses== 
#     |train_test_lag| from Regression_Helper_Funcs
#     |multiple_regressors| from Regression_Helper_Funcs
    
#     ==Returns==
#     |y_preds| : new df
#     '''
    
#     y_preds = train_test(df, Xy=True)[3]
#     rf, ols_lin, ols_smf = multiple_regressors(df, print_mses=False)
#     y_preds.rename(columns={'cost_per_watt':'actual'}, inplace=True)
#     y_preds['randomforest'] = rf
# #     y_preds['linear'] = lr
# #     y_preds['bagging'] = br
# #     y_preds['adaboost'] = abr
#     y_preds['olslinear'] = ols_lin
#     y_preds['olssmf'] = ols_smf
#     return y_preds

# def regression(df):    
#     '''
#     completes all prep and outputs regression results
#     returns df and stationary df
#     '''
#     #df, diff = prep()
#     y_preds = regres_dfs(df)
#     y_train = train_test(df, Xy=True)[1]
#     fig, axs = plt.subplots(3, figsize= (20,15), constrained_layout=True)
#     pred_s, pred_e = y_preds.index.date[0], y_preds.index.date[-1]
#     train_s, train_e = y_train.index.date[0], y_train.index.date[-1]
    
#     axs[0].plot(y_preds.actual, label= 'Actual')
#     axs[0].plot(y_preds.randomforest, label= 'Random Forest', linewidth=2)
#     axs[0].plot(y_train[-30:], label='Train', color='gray')
#     axs[0].set_title('Random Forest \n  MSE= {}'.format(round(mean_squared_error(y_preds.actual, y_preds.randomforest),5)), fontsize=18)
#     axs[0].legend(loc='best')
#     axs[0].set_xlim(left= y_train.index.date[-31])
#     fig.suptitle(' Regression Models \n Forecast For:     [{}] - [{}] \n Trained On:       [{}] - [{}]\n '.format(pred_s, pred_e, train_s, train_e), fontsize=20)
    
#     axs[1].plot(y_preds.actual, label= 'Actual')
#     axs[1].plot(y_preds.olslinear, label= 'OLS Linear', linewidth=2)
#     axs[1].plot(y_train[-30:], label='Train',color='gray')
#     axs[1].set_title('OLS Linear \n MSE = {}'.format(round(mean_squared_error(y_preds.actual, y_preds.olslinear),5)), fontsize=18)
#     axs[1].legend(loc='best')
#     axs[1].set_xlim(left= y_train.index.date[-31])
    
#     axs[2].plot(y_preds.actual, label= 'Actual', alpha=.75)
#     axs[2].plot(y_preds.olssmf, label= 'OLS', linewidth=2)
#     axs[2].plot(y_train[-30:], label='Train',color='gray')
#     axs[2].set_title('OLS smf \n MSE= {}'.format(round(mean_squared_error(y_preds.actual, y_preds.olssmf),5)), fontsize=18)
#     axs[2].legend(loc='best')  
#     axs[2].set_xlim(left= y_train.index.date[-31])
#     plt.show() 
#     #return df, diff
    
    
    
# def narx_gs(df):    
#     x = df
#     y = df

#     mdl = NARX(RandomForestRegressor(), auto_order=2, exog_order=[2], exog_delay=[1])
#     para_grid = {'n_estimators': [10, 30, 100]}
#     mdl.grid_search(x, y, para_grid, verbose=2)

#     # Best hyper-parameters are set after grid search, print the model to see the difference
#     print(mdl)
#     mdl.fit(x, y)
#     ypred = mdl.predict(x, y, step=3)
#     return ypred

# def narx(df):
#     x = df
#     y = df
#     mdl = NARX(RandomForestRegressor(), auto_order=2, exog_order=[2], exog_delay=[1])
#     mdl.fit(x, y)
#     ypred = mdl.predict(x, y, step=3)
#     return ypred

    
# # =============================================================================
# # ARIMA PARAMETERS
# # ============================================================================= 



# def ARIMA_predict(df, order):
    
#     train, test = train_test(df)
#     test_s, test_e = test.index.date[0], test.index.date[-1]
#     train_s, train_e = train.index.date[0], train.index.date[-1]
#     res = ARIMA(train, order=order).fit()
#     fig, ax = plt.subplots(1, figsize=(14, 4))
#     ax.plot(test.index, test)
#     ax.plot(train.index[-20:], train[-20:])
#     fig = res.plot_predict(test_s,test_e, ax=ax, plot_insample=True)
    
#     # plt.title('MSE {}'.format(round(mean_squared_error(test,res.predict('2015-06-14','2019-1-6')),5)))
#     plt.title('Forecasted [{} - {}] Data \n Based On [{} - {}] Data\n ARIMA {} MSE= {}'.format(
#                                 test_s, test_e, 
#                                 train_s, train_e,order,
#                                 round(mean_squared_error(test,res.predict(test_s, test_e)),5)))
#     plt.show()

    

# # === GET PDQ VIA AUTO ARIMA =========================================
# def auto_arima_pdq(df,trace_list=False):
#     '''
#     ==Function==
#     Uses Auto ARIMA to obtain best parameters for data
#     ==Parameters==
#     |trace_list| : bool
#         if True, function will return list of all searched pairs
#         default=False
#     ==Returns==
#     printed pdq variable
#     auto_arima variable to use in other functions
#     '''
#     arima_pdq = auto_arima(df, trace=trace_list, stepwise=False, max_p=8,max_P = 8, max_order=12).order
#     print('P, D, Q parameters to use in ARIMA model =', arima_pdq)
#     return arima_pdq
 
# # === TEST VARIOUS PDQ'S MSE =========================================    
# def evaluate_arima_model(X, arima_order):
#     '''
#     ==Function ==
#     Splits data into training/test
#     Pushes through ARIMA models 
   
#     ==Returns==
#     MSE
    
#     ==Note==
#     Only used in arima_order_mses function
#     '''
#     # prepare training dataset
#     train_size = int(len(X) * 0.8)
#     train, test = X[0:train_size], X[train_size:]
#     history = [x for x in train]
#     # make predictions
#     predictions = list()
#     for t in range(len(test)):
#         model = ARIMA(history, order=arima_order, missing='drop')
#         model_fit = model.fit(disp=0)
#         yhat = model_fit.forecast()[0]
#         predictions.append(yhat)
#         history.append(test[t])
#     # calculate out of sample error
#     error = mean_squared_error(test, predictions)
#     return error


# # === FIND BEST PARAMETERS BY RUNNING THROUGH DIFFERENT PDQ'S =========================================      
# def arima_order_mses(df):
#     '''
#     ==Function==
#     Uses various p,d,qs within below range
#     Tests out each combination 
    
#     ==Returns== 
#     Params with the best cfg + best MSE
    
#     ==Input Suggestion==
#     Use [evaluate_models(df.values.dropna(), p_values, d_values, q_values)]
    
#     ==Note==
#     Computationally expensive! 
#     '''
#     df = df.dropna().values
#     #p_values = [0, 1, 2, 4, 6, 8, 10]
#     p_values = [0, 1, 2, 4, 5, 6, 7]
#     d_values = 0
#     q_values = range(0, 4)
#     best_score, best_cfg = float("inf"), None
#     for p in p_values:
#         for d in d_values:
#             for q in q_values:
#                 order = (p,d,q)
#                 try:
#                     mse = evaluate_arima_model(df, order)
#                     if mse < best_score:
#                         best_score, best_cfg = mse, order
#                     print('ARIMA%s MSE=%.4f' % (order,mse))
#                 except:
#                     continue
#     print('Best ARIMA %s MSE=%.4f' % (best_cfg, best_score))
    
    
# # === ARIMA MODELS VIA SKTIME =========================================   

# def skt_arima(df, order):
#     '''
#     ==Function==
#     Splits dataset into 70/30 train test split
#     Applies the ARIMAForecaster from sktime 
#     Collects Forecast predictions 
    
#     ==Parameters==
#     |order| should be entered in as a tuple ex. "order=(1,1,1)"
#     If no preferred order, enter in "auto_arima(df.dropna()).order"
    
#     ==Returns==
#     Plot with the train, test, and forecast data
#     The model's MSE score
#     '''
#     idx = round(len(df) * .8)
#     tsdf = df['cost_per_watt'].dropna()
#     tsdf.reset_index(drop=True, inplace=True)
#     train = pd.Series([tsdf[:idx]])
#     test = pd.Series([tsdf[idx:]])
#     m = ARIMAForecaster(order=order)
#     m.fit(train)
#     fh = np.arange(1, (len(tsdf)-idx)+1)
#     y_pred = m.predict(fh=fh)
#     skt_mse = m.score(test, fh=fh)**2
#     skt_arima_plot(test,train,y_pred, fh, skt_mse)

# # === PLOT ARIMA MODELS VIA SKTIME =========================================             
# def skt_arima_plot(test,train,y_pred, skt_mse):    
#     fig, ax = plt.subplots(1, figsize=plt.figaspect(.25))
#     train.iloc[0].plot(ax=ax, label='train')
#     test.iloc[0].plot(ax=ax, label='test')
#     y_pred.plot(ax=ax, label='forecast')
#     ax.set(ylabel='cost_per_watt')
#     plt.title('ARIMA Model MSE ={}'.format(round(skt_mse,5)))
#     plt.legend(loc='best')
#     plt.show()