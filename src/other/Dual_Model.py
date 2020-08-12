import numpy as np
import pandas as pd
import sys
import datetime
from datetime import datetime

# MATH
from math import sqrt
from scipy import signal
from fireTS.models import NARX
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, LassoLars
from sklearn.metrics import r2_score, mean_squared_error, make_scorer, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline,  make_pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy import stats
from scipy.stats import normaltest

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

#VISUALIZATION 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
plt.style.use('ggplot')

class Dual_Models():
    
    '''
    ==Function==
    Run ARIMA and Regression Models
    
    ==Output==
    4 subplots with associated MSE score in plot titles
        - ARIMA
        - Random Forest Regression
        - OLS Linear Regression
        - OLS smf Regression 
        
    ==Parameters==
    |order_method| - selection of various methods of looking for/selecting an ARIMA order
        options:
            - 'predetermined': uses predetermined 
                    (3,0,0) for first order differenced df
                    (3,0,1) for second order differenced df
            - 'auto': uses auto_arima method to look for ARIMA order
            - 'laborious': uses best_order function (COMPUTATIONALLY EXPENSIVE)
            - 'select': allows for user to input order
            DEFAULT= 'predetermined'

    ==Included Functions==
    +train_test
    +lag_train_test
    +multiple_regressors
    +regres_dfs
    +regression
    +evaluate_arima_model
    +best_order
    +auto_pdq
    +ARIMA_predict
    +all_models
    +show_models
    
    
   '''
    
    def __init__(self,order_method='predetermined'):
        self.self = self
        self.order_method=order_method
    
    def train_test(self, df):
        idx = round(len(df)*.8)
        return df[:idx],df[idx:]

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
        model_type = ['ARIMA','Random Forest Regression', 'OLS Linear Regression', 'OLS smf Regression']
        return  y_preds, y_train, [train_s, train_e, pred_s, pred_e], model_type
        
    def evaluate_arima_model(self,X, arima_order):
        '''
        ==Function ==
        Splits data into training/test
        Pushes through ARIMA models 

        ==Returns==
        MSE

        ==Note==
        Only used in arima_order_mses function
        '''
        # prepare training dataset
        train_size = int(len(X) * 0.8)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]
        # make predictions
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=arima_order, missing='drop')
            model_fit = model.fit(disp=0)
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        # calculate error
        error = mean_squared_error(test, predictions)
        return error

    def best_order(self,df):
        '''
        ==Function==
        Uses various p,d,qs within below range
        Tests out each combination 

        ==Prints== 
        Params with the best cfg + best MSE

        ==Returns==
        best order in format: (p,d,q)
        
        ==Input Suggestion==
        Use [evaluate_models(df.values.dropna(), p_values, d_values, q_values)]

        ==Note==
        Computationally expensive! 
        '''
        df = df.dropna().values
        p_values = [0, 1, 2, 3, 4]
        d_values = range(0, 3)
        q_values = range(0, 3)
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        mse = self.evaluate_arima_model(df, order)
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                        #print('ARIMA%s MSE=%.4f' % (order,mse))
                    except:
                        continue
        return best_cfg

    def auto_pdq(self, df):
        '''
        ==Function==
        Uses Auto ARIMA to obtain best parameters for data
        ==Parameters==
        |trace_list| : bool
            if True, function will return list of all searched pairs
            default=False
        ==Returns==
        auto_arima variable to use in other functions
        '''
        return auto_arima(df, seasonal=False,stationary=True,start_p=0, start_q=0, max_order=8, stepwise=False).order
    
    def ARIMA_predict(self, df, diff_type):
        '''
        ==Function== 
        Attain user inputs to decide ARIMA order 
        
        ==Parameters==
        |diff_type| : order of differencing applied to df
            'first' or 'second'
        
        ==Returns==
        res = .fit()
        atrain, atest = train and test set used for ARIMA
        arima_title = title to be used in a plot
        a_pred = predictions from ARIMA model
        order = order used in ARIMA
        '''
        if self.order_method.lower() == 'predetermined':
            if diff_type == 'first':
                order=(3,0,0)
            elif diff_type == 'second':
                order=(3,0,1)
        elif self.order_method.lower() == 'auto':
            order=self.auto_pdq(df)
        elif self.order_method.lower() == 'manual':
            print('CAUTION: MANUAL IS VERY COMPUTATIONALLY EXPENSIVE (~20 minutes) \nPlease enter "confirm" to proceed')
            confirmation = input()
            if confirmation.lower() =='confirm': 
                print('Please hold')
                order = self.best_order(df)
            else:
                print('Changing to Auto')
                order=self.auto_pdq(df)
        elif self.order_method.lower() == 'select':
            print('Please input each parameter')
            ord_p= int(input('p:'))
            ord_d= int(input('d:')) 
            ord_q=int(input('q:'))
            order = (ord_p,ord_d,ord_q)

        atrain, atest = self.train_test(df)
        atest_s, atest_e = atest.index.date[0], atest.index.date[-1]
        atrain_s, atrain_e = atrain.index.date[0], atrain.index.date[-1]
        res = ARIMA(df, order=order).fit()
        a_pred = res.predict(atest_s, atest_e)
        arima_title = f'ARIMA {order}         MSE={round(mean_squared_error(atest,a_pred),5)}'
        return res, atrain, atest, arima_title, a_pred, order    
    
    def all_models(self,df):
        '''
        ==Function==
        Combines all regression models and ARIMA to create 4 subplots depicting actual and predicted data
        
        '''
        first = df
        sec = df.diff().dropna()
        
        #first order
        y_preds_f, y_train_f, [train_s_f, train_e_f, pred_s_f, pred_e_f], model_type = self.regression(first)
        res_f, atrain_f, atest_f, arima_title_f, a_pred_f, order_f = self.ARIMA_predict(first,diff_type='first')
        idx_f = round(len(first)*.8)
        #second order
        y_preds_s, y_train_s, [train_s_s, train_e_s, pred_s_s, pred_e_s], model_type = self.regression(sec)
        res_s, atrain_s, atest_s, arima_title_s, a_pred_s, order_s = self.ARIMA_predict(sec, diff_type='second')
        idx_s = round(len(sec)*.8)
        
        fig, axs = plt.subplots(4,2, figsize= (30,20), constrained_layout=True)
        fig.suptitle('Trained on Data From {} - {} \n Forecast for {} - {}\n \n'.format(
    ' '.join([train_s_f.strftime("%b"), str(train_s_f.year)]),
    ' '.join([train_e_f.strftime("%b"), str(train_e_f.year)]),
    ' '.join([pred_s_f.strftime("%b"), str(pred_s_f.year)]),
    ' '.join([pred_e_f.strftime("%b"), str(pred_e_f.year)])),fontsize=30)
        fig.text(.18,.92, 'First Order Differencing',fontsize=25)
        fig.text(.68,.92, 'Second Order Differencing',fontsize=25)

        let_sets= ['f','s']
        for j in range(2):
            for i in range(1,4):
                #exec(f"axs[{i},{j}].plot(y_preds_{let_sets[j]}.{self.formastr('_'.join(('model_type',let_sets[j]))[i])}, label= '{'_'.join(('model_type',let_sets[j]))[i]}', linewidth=2)")
                
                exec(f"axs[{i},{j}].plot(y_preds_{let_sets[j]}.{self.formastr(model_type[i])}, label= '{model_type[i]}', linewidth=2)")
                exec(f"axs[{i},{j}].plot(y_preds_{let_sets[j]}.actual, label= 'Actual')")
                exec(f"axs[{i},{j}].plot(y_train_{let_sets[j]}[-30:], label='Train', color='gray')")
                exec(f"axs[{i},{j}].fill_between(y_preds_{let_sets[j]}.index, y_preds_{let_sets[j]}.{self.formastr(model_type[i])}, y_preds_{let_sets[j]}.actual, color='gray', alpha=.3)")
                exec(f"axs[{i},{j}].set_title('{model_type[i]}        MSE=%s' % round(mean_squared_error(y_preds_{let_sets[j]}.actual, y_preds_{let_sets[j]}.{self.formastr(model_type[i])}),5), fontsize=18)")
                exec(f"axs[{i},{j}].legend(loc='best')")
                exec(f"axs[{i},{j}].set_xlim(left=y_train_{let_sets[j]}.index.date[-31])")
            exec(f"axs[0,{j}].plot(a_pred_{let_sets[j]}, label='ARIMA Forecast')")
            exec(f"axs[0,{j}].plot(atest_{let_sets[j]}.index, atest_{let_sets[j]}, label='Actual')")
            exec(f"axs[0,{j}].plot(atrain_{let_sets[j]}.index[-30:], atrain_{let_sets[j]}[-30:], label='Train', color='gray')")
            exec(f"axs[0,{j}].fill_between(a_pred_{let_sets[j]}.index, atest_{let_sets[j]}.cost_per_watt.values, 0, color='gray', alpha=.3)")
            exec(f"axs[0,{j}].set_title(arima_title_{let_sets[j]}, fontsize=18)")
            exec(f"axs[0,{j}].legend(loc='best')")
            exec(f"axs[0,{j}].set_xlim(left=atrain_{let_sets[j]}.index.date[-31])")

        #plt.savefig('model_plots.png')
        plt.show()
    
    def residual_dist(self, df, order):
        '''
        ===Returns===
        a 2-tuple of the chi-squared statistic, and the associated p-value. if the p-value is very small, it means the residual is not a normal distribution
        '''
        arima_mod = ARIMA(df, order).fit(disp=False)
        resid = arima_mod.resid
        #print(normaltest(resid))
        fig = plt.figure()
        ax0 = fig.add_subplot(111)
        sns.distplot(resid ,fit = stats.norm, ax = ax0) 
        # Get the fitted parameters used by the function
        (mu, sigma) = stats.norm.fit(resid)
        plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
        plt.ylabel('Frequency')
        plt.title(f'Residual distribution \n {normaltest(resid)}')
        plt.show()
        
    def show_models(self,df):
        return self.all_models(df)

if __name__ == "__main__":
    Models().show_models(df)