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

class Models():
    
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
            - 'predetermined': uses predetermined (2,0,0) as ARIMA order
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
    +show_model
    
    
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
        
    
    def regressor_fits(self,df, lag_len=3):
        X_train, y_train, X_test, y_test = self.lag_train_test(df)
        rf= RandomForestRegressor(n_jobs=-1).fit(X_train,y_train)
        ols_lin = sm.OLS(y_train, X_train).fit()
        ols_train, ols_test= self.lag_train_test(df, lag_len=lag_len, Xy=False)
        ols_str = 'y ~ ' +" + ".join([f"Lag{i}" for i in range(1,lag_len+1)])
        ols = smf.ols(ols_str, data=ols_train).fit()
        return rf, ols_lin, ols
    
    def regressor_predicts(self,df):
        X_train, y_train, X_test, y_test = self.lag_train_test(df)
        ols_train, ols_test= self.lag_train_test(df, Xy=False)
        rff, ols_linf, olsf = self.regressor_fits(df)
        rf= rff.predict(X_test)
        ols_lin = ols_linf.predict(X_test)  
        ols = olsf.predict(ols_test)
        return rf, ols_lin, ols 
    
    def regressor_resids(self,df):
        X_train, y_train, X_test, y_test = self.lag_train_test(df)
        rf, ols_lin, ols_smf = self.regressor_fits(df)
        OLS_smf_resid = pd.DataFrame(ols_smf.resid)
        OLS_lin_resid = pd.DataFrame(ols_lin.resid)
        RandomForest_resid = (y_test.cost_per_watt - rf.predict(X_test)).rename(columns={'cost_per_watt':''})
        return RandomForest_resid, OLS_lin_resid, OLS_smf_resid
    
    def regres_dfs(self, df):
        y_preds = self.lag_train_test(df)[3]
        rf, ols_lin, ols_smf = self.regressor_predicts(df)
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
        
        ==Returns==
        auto_arima variable to use in other functions
        
        ==Set Parameters==
        out_of_sample_size=750
        stepwise=False
        n_jobs=-1
        start_q=0 
        stationary=True
        test='adf'
        seasonal=False 
        information_criterion='oob'
        
        '''
        #return auto_arima(df, seasonal=False,stationary=True,start_p=0, start_q=0, max_order=8, stepwise=False).order
        #return auto_arima(df, test='adf', out_of_sample_size=750,stationary=True,seasonal=False,n_jobs=-1, start_p=2,start_q=0,stepwise=False, max_q=0,max_order=2).order
    
        return auto_arima(df, out_of_sample_size=750,stepwise=False,n_jobs=-1,start_q=0, stationary=True, test='adf', seasonal=False, information_criterion='oob').order
    
    def ARIMA_predict(self, df):
        '''
        ==Function== 
        Attain user inputs to decide ARIMA order 
        
        ==Returns==
        res = .fit()
        atrain, atest = train and test set used for ARIMA
        arima_title = title to be used in a plot
        a_pred = predictions from ARIMA model
        order = order used in ARIMA
        '''
        if self.order_method.lower() == 'predetermined':
            order=(2,0,0)
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
        y_preds, y_train, [train_s, train_e, pred_s, pred_e], model_type = self.regression(df)
        res, atrain, atest, arima_title, a_pred, order = self.ARIMA_predict(df)        
        RandomForest_resid, OLS_lin_resid, OLS_smf_resid = self.regressor_resids(df)
        resid_list = [res.resid,RandomForest_resid, OLS_lin_resid, OLS_smf_resid]
        idx = round(len(df)*.8)
        fig, axs = plt.subplots(4,2, figsize= (30,20), constrained_layout=True)
        fig.suptitle('Trained on Data From {} - {} \n Forecast for {} - {}\n \n'.format(
    ' '.join([train_s.strftime("%b"), str(train_s.year)]),
    ' '.join([train_e.strftime("%b"), str(train_e.year)]),
    ' '.join([pred_s.strftime("%b"), str(pred_s.year)]),
    ' '.join([pred_e.strftime("%b"), str(pred_e.year)])),fontsize=30)
        for j in range(2):
            for i in range(1,4):
                exec(f"axs[{i},0].plot(y_preds.{self.formastr(model_type[i])}, label= '{model_type[i]}', linewidth=2)")
                exec(f"axs[{i},0].plot(y_preds.actual, label= 'Actual')")
                exec(f"axs[{i},0].plot(y_train[-30:], label='Train', color='gray')")
                exec(f"axs[{i},0].fill_between(y_preds.index, y_preds.{self.formastr(model_type[i])}, y_preds.actual, color='gray', alpha=.3)")
                exec(f"axs[{i},0].set_title('{model_type[i]}        MSE=%s' % round(mean_squared_error(y_preds.actual, y_preds.{self.formastr(model_type[i])}),5), fontsize=18)")
                exec(f"axs[{i},0].legend(loc='best')")
                exec(f"axs[{i},0].set_xlim(left=y_train.index.date[-31])")
                exec(f"axs[{i},1] = sns.distplot(resid_list[{i}], fit=stats.norm, ax=axs[{i},1])")
                exec(f"(mu,sigma)= stats.norm.fit(resid_list[{i}])")
                exec(f"axs[{i},1].legend(['Normal dist. ($\mu=$ round(mu,2) and $\sigma=$ round(sigma,2))'], loc='best')")
                exec(f"axs[{i},1].set_ylabel('Frequency')")
                exec(f"axs[{i},1].set_title('Residual Distribution for {model_type[i][:-11]}\n Normal Test Result | statistic={round(normaltest(resid_list[i])[0],4)} | pvalue={round(normaltest(resid_list[i])[1],5)}')")
        axs[0,0].plot(a_pred, label='ARIMA Forecast')
        axs[0,0].plot(atest.index, atest, label='Actual')
        axs[0,0].plot(atrain.index[-30:], atrain[-30:], label='Train', color='gray')
        axs[0,0].fill_between(a_pred.index, atest.cost_per_watt.values, 0, color='gray', alpha=.3)
        axs[0,0].set_title(arima_title, fontsize=18)
        axs[0,0].legend(loc='best')
        axs[0,0].set_xlim(left=atrain.index.date[-31])
        axs[0,1] = sns.distplot(resid_list[0], fit=stats.norm, ax=axs[0,1])
        (mu,sigma)= stats.norm.fit(resid_list[0])
        axs[0,1].legend(['Normal dist. ($\mu=$ rounded(mu,2) and $\sigma=$ rounded(sigma,2))'], loc='best')
        axs[0,1].set_ylabel('Frequency')
        axs[0,1].set_title(f'Residual Distribution for ARIMA \nNormal Test Result | statistic={round(normaltest(resid_list[0])[0],4)} | pvalue={round(normaltest(resid_list[0])[1],5)}')
        #plt.savefig('model_plots.png')
        plt.show()
    
#     def residual_dist(self, df, order):
#         '''
#         ===Returns===
#         a 2-tuple of the chi-squared statistic, and the associated p-value. if the p-value is very small, it means the residual is not a normal distribution
#         '''
#         arima_mod = ARIMA(df, (2,0,0)).fit()
#         resid = arima_mod.resid
#         fig = plt.figure(constrained_layout=True)
#         ax0 = fig.add_subplot(111)
#         sns.distplot(resid ,fit = stats.norm, ax = ax0) 
#         # Get the fitted parameters used by the function
#         (mu, sigma) = stats.norm.fit(resid)
#         plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
#         plt.ylabel('Frequency')
#         plt.suptitle('Residual Distribution', fontsize=(15))
#         plt.title(f'Normal Test Result | statistic={round(normaltest(resid)[0],4)} | pvalue={round(normaltest(resid)[1],5)}')
#         plt.show()
        
    def show_model(self,df):
        return self.all_models(df)

if __name__ == "__main__":
    Models().show_model(df)