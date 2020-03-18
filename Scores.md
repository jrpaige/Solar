<b>random forest regressor</b> <br>
OOB Score: 0.9806250796856056 <br>
R-Squared on test: 0.9974039392381283<br>
MSE: 0.01903015436936935
   
   
<b>OLS Linear Regression on cost per watt with lags/b> <br>
R-Squared = 0.937<br>
coefs<br>
    - intercept = 0.0235
    - Lag1 =      0.3002
    - Lag2 =      0.2809
    - Lag3 =      0.4130

    
OLS Linear Regression on cost per watt<br>
R-squared = 0.937<br>
coef <br>
    - const = 12.0522
    - x1 =    -0.0102

    
OLS Linear Regression on weekly differences<br>
R-squared = 0.000<br>
coef 
    - const = -0.0164
    - x1 =    1.389e-05

ARIMA on weekly differences<br>
coef<br>
    - const         -0.0073  
    - ar.L1         -0.8021  
    - ar.L2         -0.9927  
    - a.L1          -0.1408      
    - ma.L2          0.2651      
    - ma.L3         -0.9064  
    - ma.L4         -0.0627
    - sigma2         1.6376
Mean Error:                       0.000835770942663377 
Mean Absolute Error :             0.18562102619417667 
Root Mean Squared Error :         0.3261506489169117 
Correlation between the 
Actual and the Forecast:          0.6710069749690127    
    
    
    
    
https://machinelearningmastery.com/time-series-forecasting-performance-measures-with-python/
 _____________________________       
    
<b>Forecast Error (or Residual Forecast Error) </b>
    forecast_error = expected_value - predicted_value

    expected = [0.0, 0.5, 0.0, 0.5, 0.0]
    predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
    forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
    print('Forecast Errors: %s' % forecast_errors)
    
_____________________________    
    
<b>Mean Forecast Error (or Forecast Bias)</b>
    mean_forecast_error = mean(forecast_error)

    expected = [0.0, 0.5, 0.0, 0.5, 0.0]
    predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
    forecast_errors = [expected[i]-predictions[i] for i in range(len(expected))]
    bias = sum(forecast_errors) * 1.0/len(expected)
    print('Bias: %f' % bias)
    
_____________________________        
    
<b>Mean Absolute Error</b>    
    mean_absolute_error = mean( abs(forecast_error) )
    
    from sklearn.metrics import mean_absolute_error
    expected = [0.0, 0.5, 0.0, 0.5, 0.0]
    predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
    mae = mean_absolute_error(expected, predictions)
    print('MAE: %f' % mae)
    
_____________________________        
    
    
<b> Mean Squared Error</b>
    mean_squared_error = mean(forecast_error^2)
       
    from sklearn.metrics import mean_squared_error
    expected = [0.0, 0.5, 0.0, 0.5, 0.0]
    predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
    mse = mean_squared_error(expected, predictions)
    print('MSE: %f' % mse)
    
_____________________________        
    
    
    
<b>Root Mean Squared Error</b>
    rmse = sqrt(mean_squared_error)
    
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    expected = [0.0, 0.5, 0.0, 0.5, 0.0]
    predictions = [0.2, 0.4, 0.1, 0.6, 0.2]
    mse = mean_squared_error(expected, predictions)
    rmse = sqrt(mse)
    print('RMSE: %f' % rmse)
    
_____________________________        
    