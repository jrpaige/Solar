import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.arima_model import *
from sklearn.metrics import r2_score, mean_squared_error, make_scorer, mean_absolute_error
import pyramid
from pmdarima.arima import auto_arima
import numpy as np
import pandas as pd
import sys
 
# === TEST VARIOUS PDQ'S MSE =========================================    
def evaluate_arima_model(X, arima_order):
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
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# === FIND BEST PARAMETERS BY RUNNING THROUGH DIFFERENT PDQ'S =========================================      
def best_order(df):
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
    p_values = [0, 1, 2,3, 4]
    d_values = range(0, 3)
    q_values = range(0, 3)
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(df, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    #print('ARIMA%s MSE=%.4f' % (order,mse))
                except:
                    continue
    #print('Best ARIMA %s MSE=%.4f' % (best_cfg, best_score))
    print(best_cfg)
    return best_cfg

        
# if __name__ == "__main__":   
#     ARIMA_pdq().best_order(df)



def auto_pdq(df,trace_list=False):
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
    #print('P, D, Q parameters to use in ARIMA model =', arima_pdq)