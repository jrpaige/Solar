# Time Series Forecasting of Residential Solar Panel Costs 

## <center>ABSTRACT</center>

The relationship between technology, climate change, and awareness relative to renewable energy creates interesting nuances for the future of the energy sector. Over 1.6 million observations collected from 1998 - 2018 of solar panel installations were used in the project, of which 95% consisted of residential installations. The project’s model employs univariate ARIMA time series analysis to generate a forecast for a cost_per_watt target variable. The exploratory aspect of the project provided an opportunity to understand time series analysis on a granular level.  

Tech Stack Used: 
Python, Numpy, Pandas, Scikit-Learn, Matplotlib, Math, SciPy, StatsModels, Pyramid,  CPI, Seaborn, sktime, fireTS

 
Fun Fact about:
The data includes over 60 features/columns related to price, taxes/rebate programs, technology, location, manufacturing, and installation. 

## <center>WORKFLOW</center>
- DATA
    - DATA ENGINEERING
    - EDA
- SCRIPTS
- TIME SERIES PREPARATION
  - SEASONALITY + TREND
  - STATIONARITY
- REGRESSION MODELS
  - MODEL BUILDING
- TIME SERIES MODELS
  - ARIMA
  - MODEL BUILDING
- PERFORMANCE
- RESULTS + SCORES
- INSIGHTS
- NEXT STEPS 


### <center><u>DATA</u></center>  
#### EDA

- 20 Years  
- 28 States
- 6 Customer Segments
- EDA NOTEBOOK TO BE ADDED 

#### DATA ENGINEERING
###### VARIABLE CREATION
- Total adjusted installed  cost = 
total installed cost with adjustments made for inflation
- <img src="https://latex.codecogs.com/gif.latex?cost\&space;per\&space;watt&space;=&space;\frac&space;{total\&space;adjusted\&space;installed\&space;cost}{system\&space;size}" title="cost\ per\ watt = \frac {total\ adjusted\ installed\ cost}{system\ size}" />

###### DATA TRANSFORMATION
- Resampled data into weekly medians

###### NULL HANDLING
- Nulls were replaced with median values from same year
  


### <center><u>SCRIPTS</u></center>
###### Prep.py
- Completes 11 steps to prep data for Time Series and Regression models

###### Regression_Helper_Funcs.py
- Model for Random Forest Regressor, OLS Linear , and OLS 
- Includes code for Robust Linear, GLS, Holt Linear, NARX, Direct Autoregressor, and Random Forest Grid Search

###### Time_Series_Helper_Funcs.py
###### ARIMA_Helper_Funcs.py

  
  
### <center><u>TIME SERIES PREPARATION</u></center>

#### SEASONALITY & TREND
Data did not show any signs of <u>seasonality</u> or <u>trends</u>. <br>
However, data was not initially <u>stationary</u>. 

 #### STATIONARITY
Ensuring a series is stationary makes the forecasts more reliable.
Within this data, <u>stationarity</u> had to be achieved by taking the change in cost_per_watt from one week to the next, aka <u>differencing</u> the data. <u>Differencing</u> can help stabilize the mean of a time series by removing changes in the level of a time series, and therefore eliminating (or reducing) any <u>trend</u> and <u>seasonality</u>.

Tests for stationarity: Adf, KPSS, PP

### <center><u>REGRESSION MODELS</u></center>
Regression was used as a means to reference how ARIMA was performing on the data when compared to basic regressors. 
- Random Forest Regressor
- OLS Linear 
- OLS 

### <u><center>TIME SERIES MODEL</center></u>

| ARIMA | Term |Parameter|Notes |Plot Used|
|:------:|:------:|:------:|:------|:------:|
|AR|AutoRegressive|[p]| Number of lags of Y to be used as predictors| [PACF] <br>Partial Autocorrelation |
|||||||||
|I|Integrated|[d]| Minimum number of differencing to make stationary number of lags|
|||||||||
|MA|Moving Average|[q] | Order of the moving average term <br> number of lagged forecast errors |[ACF] <br>Autocorrelation |

<br>
<br>
<br>

|Plot ||Math |Notes|
|:----------:|:-------------|:--------|:------|
|**PACF**| Partial Autocorrelation |<a href="https://www.codecogs.com/eqnedit.php?latex=Y_t&space;=&space;\alpha_0&space;&plus;&space;\alpha_1&space;Y_{t-1}&space;&plus;&space;\alpha_2&space;Y_{t-2}&space;&plus;&space;\alpha_3&space;Y_{t-3}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y_t&space;=&space;\alpha_0&space;&plus;&space;\alpha_1&space;Y_{t-1}&space;&plus;&space;\alpha_2&space;Y_{t-2}&space;&plus;&space;\alpha_3&space;Y_{t-3}" title="Y_t = \alpha_0 + \alpha_1 Y_{t-1} + \alpha_2 Y_{t-2} + \alpha_3 Y_{t-3}" /></a>|- Know if that lag is needed in the AR term <br> - Partial autocorrelation of lag (k) of a series is the coefficient of that lag in the autoregression equation of Y.<br> - <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;\small&space;Y_t" title="\small Y_t" />  = current series <br> -  $𝑌_{𝑡−1}$  = lag 1 of  𝑌 <br>- Partial AC of lag 3  <img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;\small&space;\left&space;[Y_{t-3}&space;\right&space;]" title="\small \left [Y_{t-3} \right ]" />  = the coefficient  𝛼_3  of   𝑌𝑡−3|
|**ACF**| Autocorrelation||- Conveys how many MA terms are required to remove any autocorrelation in the stationarized series.	
    
    
### <u>PERFORMANCE</u>
I chose to use Mean Squared Error as the metric to score both the regression and the ARIMA models. ACF was also taken into consideration in some initial time series dilligence and trials.

Given that the autoregressive and integrated lag aspect of an ARIMA model, it was no surprise that the ARIMA model and the OLS Linear model performed quite similarly. 


### <center><u>RESULTS + SCORES</u></center>



### <center><u>INSIGHTS</u></center>

Annomoloy Detection 
- There is a seemingly random jump in cost_per_watt around 2016. This can likely be attributed to the political lanscape, as politics have a major influence in the US on renewable energy progress. 
- Studies have shown that, historically, renewable energy progress slows when there is a larger Republican influence in both congress and the executive branch. [include citation]
- I hypothesize that residential customers were likely trying to take advantage of any renewable energy incentives programs available before they were gone. The increased demand likely drove the costs up. 

*** Beta coefficients for OLS and how the previous time frames affect future time frames ***

### <center><u>NEXT STEPS</u></center>

Use Reinforcement learning of a LSTM RNN Model to utilize multiple variables.<br>
Potential significant variables:
- income
- political landscape
- global energy trends
- temperature/climate change
- national generation and useage
- innovations

