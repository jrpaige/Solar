# Time Series Forecasting of Solar Panel Costs 
--- 
This model uses data collected through Berkeley Lab's [Tracking the Sun](https://emp.lbl.gov/tracking-the-sun/) initiative. 
The datasets includes over 1.6 million observations of solar panel installations collected over twenty years. 
The project’s model employs univariate ARIMA time series analysis and multiple regressors to generate a forecast for a cost_per_watt target variable. The exploratory aspect of the project provided an opportunity to understand time series analysis on a granular level.  

<img  align='center', src="https://www.ecmag.com/sites/default/files/solar-energy-city.jpg" width="700">

---
# Tech Stack

`Python`
`Numpy`
`Pandas`
`Scikit-Learn`
`Matplotlib`
`SciPy`
`StatsModels`
`PMDarima`
`CPI`
`Seaborn`
`sktime`
`fireTS`

---
# Scripts
`Prep_Class.py`
- This class incorporates 11 steps to prep the data for the Time Series and Regression models. 

`Regression_Helper_Funcs.py`
- Model for Random Forest Regressor, OLS Linear , and OLS 
- Includes code for Robust Linear, GLS, Holt Linear, NARX, Direct Autoregressor, and Random Forest Grid Search

`Time_Series_Helper_Funcs.py`
- code to assist in the decompostion of the data

`ARIMA_Helper_Funcs.py`
- code to employ the ARIMA model 

--- 
# Data 

### EDA
The data includes over 60 features/columns related to price, taxes/rebate programs, technology, location, manufacturing, and installation. 

### Data Transformation
- Resampled data into weekly medians

Per [Solar.com](https://www.solar.com/learn/solar-panel-cost/):
"Most [systems] cost between $3.00 and $4.00 per watt." "As a general rule of thumb, be skeptical of any solar quote[...]more than $5.00 per watt." It should be taken into consideration that these numbers are relative to today's prices. Given the increased technology and innovation, costs used to be much higher. In 1977, "the cost of a solar cell was about $330.28 per watt, in today's dollars." 

Within the data, 604,046 entries totaled more than $5.00 per watt and 59,342 entries totaled more than $10.00 per watt. The data's cost_per_watt variable accounts for the cost all associated costs including installation. Given inflation and decreasing costs over the time series's 20 year span, I excluded the 59342 entries which came in abve 

<img src="imgs/watt_outliers.png">


### Variable Creation
- Total adjusted installed  cost = 
total installed cost with adjustments made for inflation<br> 
- cost per watt<img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\fn_phv&space;\small&space;=\frac&space;{\textup{total&space;adjusted&space;installed&space;costs}}{\textup{system&space;size}}" />

### Null Handling
- Nulls were replaced with median values from same year
  
---
# Time Series 

#### DECOMPOSITION 
Data did not show any signs of seasonality, cyclicality, or trends. <br>
white noise<br>
However, data was not initially stationary. 

 #### STATIONARITY
- ACF and PACF plots 
- Rolling mean, median, and std
- Augmented Dickey Fuller Test


Within this data, stationarity had to be achieved by taking the change in cost_per_watt from one week to the next, aka differencing the data. 
Differencing can help stabilize the mean of a time series by removing changes in the level of a time series, and therefore eliminating (or reducing) any trend and seasonality.

A critical value of .05 or 5% was used to reject or fail to reject ADF's null hypothesis. After differencing, the P-value for the data reached less than 0.000000 and stationarity was decidedly achieved. 

---
# Models

#### TIME SERIES MODEL
##### ARIMA 
Used AutoArima to identify p,d,q parameters


#### REGRESSION MODELS
Regression was used as a means to reference how ARIMA was performing on the data when compared to basic regressors. 
##### Random Forest Regressor
##### OLS Linear 
##### OLS 
---
#  Performance 

#### EVALUTATION
I chose to use Mean Squared Error as the evaluation metric to score both the regression and the ARIMA models. ACF was also taken into consideration in some initial time series dilligence and trials.

#### RESULTS
Given that the autoregressive and integrated lag aspect of an ARIMA model, it was no surprise that the ARIMA model and the OLS Linear model performed quite similarly. 

Most of the models that employed the use of built out data outperformed traditional statistics for univariate time series.




---
#  Insights

#### Annomoloy Detection 
- There is a seemingly random jump in cost_per_watt around 2016. This can likely be attributed to the political lanscape, as politics have a major influence in the US on renewable energy progress. 
- Studies have shown that, historically, renewable energy progress slows when there is a larger Republican influence in both congress and the executive branch. [include citation]
- I hypothesize that residential customers were likely trying to take advantage of any renewable energy incentives programs available before they were gone. The increased demand likely drove the costs up. 

*** Beta coefficients for OLS and how the previous time frames affect future time frames ***

---
#  Next Steps

Use Reinforcement learning of a LSTM RNN Model to utilize multiple variables.<br>
Potential significant variables:
- income
- political landscape
- global energy trends
- temperature/climate change
- national generation and useage
- innovations

