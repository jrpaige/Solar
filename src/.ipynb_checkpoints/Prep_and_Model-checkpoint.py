import numpy as np
import pandas as pd
import cpi
import sys
import datetime
from datetime import datetime


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
# from sktime.forecasters import ARIMAForecaster
# from sktime.highlevel.tasks import ForecastingTask
# from sktime.highlevel.strategies import ForecastingStrategy
# from sktime.highlevel.strategies import Forecasting2TSRReductionStrategy
# from sktime.pipeline import Pipeline
# from sktime.transformers.compose import Tabulariser

#VISUALIZATION 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6
plt.style.use('ggplot')


# =============================================================================
# DF PREP
# =============================================================================
class Preps():
    '''
    ==Function==
    Returns stationary dataframe
    
    ==Columns==
    Installation Date, System Size, Total Installed Price, Customer Segment
    Lowercases and replaces all spaces with underscores
    
    ==Index==
    Cronologically Ordered by Installation Date
    
    ==Customer Segment==
    Residential is ~94% of the data 
    Other Sectors deviate largely from full data
    Focuses on Residential to eliminate any irregularity 

    ==Null Values==
    The number of null price cells[356171] / the number of non-null price cells[1543831] = > 20%
    Nulls were entered in as '-9999'
    Replaces -9999 with np.nan
    Replaces all null values with the median from that year 

    ==Inflation Adjustment==
    Adjusts all prices to reflect inflation of the current year(2019)
    The adjustment is made using data provided by The Bureau of Labor Statistics at the U.S. Department of Labor.
    
    ==Target Variable==
    create target variable, cost_per_watt 
    cost_per_watt = total installed price/system size/1000
    
    ==Resample into Weekly medians==
    Resamples into the weekly medians and continuous non-null df

    == Stationarity==
    Uses ADFuller method to test for stationarity.
    If data is not stationary, uses differencing to achieve stationarity.
    Completes ADF testing again

    '''
    
        
    def __init__(self, files):
        """
        ==Parameters==
        |files| : entered in as a list
        """
        self.self = self
        self.files = files
        cpi.update()
        
    def load(self):
        loaded_files = []
        count = 0
        print('PREPPING DATA'.center(76,' '))
        for i in range(1, len(self.files)+1):
            exec(f"df{i} = pd.read_csv(self.files[{count}],encoding='iso-8859-1',parse_dates=['Installation Date'], usecols=['Installation Date','System Size', 'Total Installed Price','Customer Segment'],na_values=(-9999, '-9999'))")
            count+=1
            exec(f"loaded_files.append(df{i})")
        if len(loaded_files) > 1:
            df = pd.concat([i for i in loaded_files],ignore_index=True)
        else: 
            df=loaded_files[0]

        return df

    def clean(self):
        df = self.load()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        return df

    def refine(self):
        df = self.clean()
        df = df.loc[df['customer_segment']=='RES']
        return df

    def date_index(self):
        df = self.refine()
        df.sort_values('installation_date', inplace=True)
        df.set_index('installation_date', drop=True, inplace=True)
        return df
    
    def null_handler(self):
        df = self.date_index()
        [df['total_installed_price'].replace(np.nan,round(df.loc[(df['total_installed_price'] != np.nan) & (df.index.year == i)]['total_installed_price'].median(),2),inplace=True) for i in range(1998,2019)] 
        return df

    def inflation_assist(self):
        df = self.null_handler()
        df['date'] = df.index.date
        df['adj_installed_price'] = round(df.apply(lambda x:cpi.inflate(x.total_installed_price, x.date), axis=1),2)
        return df

    def target_variable(self):
        df = self.inflation_assist()
        df['cost_per_watt'] = round(df['adj_installed_price']/ df['system_size']/1000,2)
        return df

    def resampler(self):
        df = self.target_variable()
        null_list = []
        for i in range(len(df.cost_per_watt.resample('W').median())):
            if df.cost_per_watt.resample('W').median()[i] >0:
                pass
            else:
                null_list.append(i)
        y = df.cost_per_watt.resample('W').median()[null_list[-1]+1:]
        return y

    def stationarity(self):
        y = self.resampler()
        if round(adfuller(y)[1],4) < 0.51:
            return y
        else:
            differences = y.diff(periods=1).dropna()
            if round(adfuller(differences)[1],4) < 0.51:
                return differences
            else:
                return differences
    
    def compile(self):
        df = self.stationarity()
        return pd.DataFrame(df)

file_path_1 = '/Users/jenniferpaige/code/DSI/getit/TTS_10-Dec-2019_p1.csv'
file_path_2 = '/Users/jenniferpaige/code/DSI/getit/TTS_10-Dec-2019_p2.csv'
files = [file_path_1, file_path_2]
if __name__ == "__main__":   
    df = Prep(files).compile()
    
#========RUN MODELS=========    
class Models():

    '''    
    ==RUN MODELS==
    Run ARIMA and Regression Models
    
    ==Parameters==
    
    | df | - pass in univariate time series dataframe for 'cost_per_watt'

    ===Returns===
    4 subplots 
        - ARIMA
        - Random Forest Regression
        - OLS Linear Regression
        - OLS smf Regression
    '''
    def __init__(self):
        self.self=self
        self.df = Preps(files).compile()
        
    def train_test(self):
        idx = round(len(self.df)*.8)
        train, test = self.df[:idx], self.df[idx:]
        return train, test
        
    def lag_train_test(self, Xy=True, lag_len=3):
        lag_df = (pd.concat([self.df.shift(i) for i in range(lag_len+1)], axis=1, keys=['y'] + ['Lag%s' % i for i in range(1, lag_len+1)])).dropna() 
        idx = round(len(lag_df)*.8)
        if Xy==False:
            train, test = lag_df[:idx], lag_df[idx:]
            return train, test
        elif Xy==True:
            lag_y, lag_X  = lag_df.y, lag_df.drop(columns='y')    
            X_train, y_train, X_test, y_test = lag_X[:idx], lag_y[:idx], lag_X[idx:], lag_y[idx:]
            return X_train, y_train, X_test, y_test
        
    def multiple_regressors(self, lag_len=3, print_mses=True):
        X_train, y_train, X_test, y_test = self.lag_train_test(lag_len=lag_len)
        rf= RandomForestRegressor(n_jobs=-1).fit(X_train,y_train).predict(X_test)
        ols_lin = sm.OLS(y_train, X_train).fit().predict(X_test)  
        ols_train, ols_test= self.lag_train_test(lag_len=lag_len, Xy=False)
        ols_str = 'y ~ ' +" + ".join([f"Lag{i}" for i in range(1,lag_len+1)])
        ols = smf.ols(ols_str, data=ols_train).fit().predict(ols_test)
        return rf, ols_lin, ols 
    
    def regres_dfs(self):
        y_preds = self.lag_train_test()[3]
        rf, ols_lin, ols_smf = self.multiple_regressors(print_mses=False)
        y_preds.rename(columns={'cost_per_watt':'actual'}, inplace=True)
        y_preds['randomforest'], y_preds['olslinear'],y_preds['olssmf'] = rf, ols_lin, ols_smf
        return y_preds
    
    def formastr(self,str):
        return str.replace(" ","").replace("Regression","").lower()
    
    def regressors(self):   
        y_preds = self.regres_dfs()
        y_train = self.lag_train_test(Xy=True)[1]
        pred_s, pred_e = y_preds.index.date[0], y_preds.index.date[-1]
        train_s, train_e = y_train.index.date[0], y_train.index.date[-1]
        model_type = ['ARIMA','Random Forest Regression', 'OLS Linear Regression', 'OLS smf Regression']
        return  y_preds, y_train, [train_s, train_e, pred_s, pred_e], model_type
    
    # === TEST VARIOUS PDQ'S MSE =========================================    
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


    # === FIND BEST PARAMETERS BY RUNNING THROUGH DIFFERENT PDQ'S =========================================   
    def best_order(self):
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
        df = self.df.dropna().values
        p_values = [0, 1, 2, 3, 4]
        d_values = range(0, 3)
        q_values = range(0, 3)
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        mse = self.evaluate_arima_model(order)
                        if mse < best_score:
                            best_score, best_cfg = mse, order
                        #print('ARIMA%s MSE=%.4f' % (order,mse))
                    except:
                        continue
        return best_cfg

    def auto_pdq(self):
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
        return auto_arima(self.df, seasonal=False,stationary=True,start_p=0, start_q=0, max_order=8, stepwise=False).order
    
    def ARIMA_predict(self):
        '''
        Would you like the model to look for an ARIMA order?: 
        if user enters Y:    
            model will look for best p,d,q order
            user must input Auto or Manual
            Auto uses auto_arima function
            Manual uses best_order function
            CAUTION: MANUAL IS VERY COMPUTATIONALLY EXPENSIVE (~20 minutes)
        if user enters N:
            user is asked if they would like to enter their own p,d,q
            if user enters Y:
                inputs for p,d, and q will follow
            if user enters N:
                model will use ARIMA p,d,q (4,1,1) as order                
        '''
        print('Would you like the model to look for an ARIMA order? (Y/N):')
        find_order = input()
        if find_order.lower()=='y':
            print('Auto or Manual?:')
            pdq_type= input()
            if pdq_type.lower()=='manual':
                print('CAUTION: MANUAL IS VERY COMPUTATIONALLY EXPENSIVE (~20 minutes) \nPlease enter "confirm" to proceed')
                confirm = input()
                if confirm.lower =='confirm':
                    print('Please hold')
                    order = self.best_order()
                elif confirm.lower !='confirm':
                    pdq_type == 'auto'
            elif pdq_type.lower()=='auto':
                order=self.auto_pdq()
        elif find_order.lower=='n':
            print('Would you like to use a specific order? (Y/N)')
            enter_order= input()
            if enter_order.lower()=='y':
                ord_p= int(input('p:'))
                ord_d= int(input('d:')) 
                ord_q=int(input('q:'))
                order = (ord_p,ord_d,ord_q)
            elif enter_order.lower()=='n':
                order=(4,1,1)
        atrain, atest = self.train_test()
        atest_s, atest_e = atest.index.date[0], atest.index.date[-1]
        atrain_s, atrain_e = atrain.index.date[0], atrain.index.date[-1]
        res = ARIMA(atrain, order=order).fit()
        a_pred = res.predict(atest_s, atest_e)
        arima_title = f'ARIMA {order}         MSE={round(mean_squared_error(atest,a_pred),5)}'
        return res, atrain, atest, arima_title, a_pred    
    
    def all_models(self):
        y_preds, y_train, [train_s, train_e, pred_s, pred_e], model_type = self.regressors()
        #ts_train, ts_test, ts_y_pred, skt_title = self.skt_ARIMA(df)
        res, atrain, atest, arima_title, a_pred = self.ARIMA_predict()
        idx = round(len(self.df)*.8)
        
        fig, axs = plt.subplots(4, figsize= (20,20))
        fig.tight_layout(h_pad=5)
        for i in range(1,4):
            exec(f"axs[{i}].plot(y_preds.{self.formastr(model_type[i])}, label= '{model_type[i]}', linewidth=2)")
            exec(f"axs[{i}].plot(y_preds.actual, label= 'Actual')")
            exec(f"axs[{i}].plot(y_train[-30:], label='Train', color='gray')")
            exec(f"axs[{i}].fill_between(y_preds.index, y_preds.{self.formastr(model_type[i])}, y_preds.actual, color='gray', alpha=.3)")
            exec(f"axs[{i}].set_title('{model_type[i]}        MSE=%s' % round(mean_squared_error(y_preds.actual, y_preds.{self.formastr(model_type[i])}),5), fontsize=18)")
            exec(f"axs[{i}].legend(loc='best')")
            exec(f"axs[{i}].set_xlim(left=y_train.index.date[-31])")
        axs[0].plot(a_pred, label='ARIMA Forecast')
        axs[0].plot(atest.index, atest, label='Actual')
        axs[0].plot(atrain.index[-30:], atrain[-30:], label='Train', color='gray')
        axs[0].fill_between(a_pred.index, atest.cost_per_watt.values, 0, color='gray', alpha=.3)
        axs[0].set_title(arima_title, fontsize=18)
        axs[0].legend(loc='best')
        axs[0].set_xlim(left=atrain.index.date[-31])
        fig.suptitle('Trained On Data From: \n[{}] to [{}]'.format(train_s, train_e), y=1.05 ,verticalalignment='top', fontsize=20)
        fig.suptitle('Forecast For Data from:     \n[{}] to [{}] \n'.format(pred_s, pred_e), y=1.05 ,verticalalignment='top', fontsize=20) 
        plt.savefig('model_plots.png')
        plt.show()
        
    def show_models(self):
        return self.all_models()
    
if __name__ == "__main__": 
    Models().show_models()
    #Models(order=(0,0,0)).show_models(df)