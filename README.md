# Time Series Forecasting of Residential Solar Panel Costs 

## <center>ABSTRACT</center>

The relationship between technology, climate change, and awareness relative to renewable energy creates interesting nuances for the future of the energy sector. Over 1.6 million observations collected from 1998 - 2018 of solar panel installations were used in the project, of which 95% consisted of residential installations. The project’s model employs univariate ARIMA time series analysis to generate a forecast for a cost_per_watt target variable. The exploratory aspect of the project provided an opportunity to understand time series analysis on a granular level.  

Tech Stack Used: 
Python, Numpy, Pandas, Scikit-Learn, Matplotlib, Math, SciPy, StatsModels, Pyramid,  CPI, Seaborn,

 
Fun Fact about:
The data includes over 60 features/columns related to price, taxes/rebate programs, technology, location, manufacturing, and installation. 

## <center>WORKFLOW</center>
- DATA
    - DATA ENGINEERING
    - EDA
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
 - [1998-2018]
- 28 States
 - [AR,AZ,CA,CO,CT,DC,DE,FL,KS,MA,MD,MN,MO,NH,NJ,NM,NY,OH,OR,PA,RI,TX,UT,VT,WI]
- 6 Customer Segments
 - [Residential, Commercial, Non-Residential, Government, Non-Profit, School] 



<center> <b><u>AVERAGE COST PER WATT FOR RESIDENTIAL CUSTOMERS <br> GIVEN SIZE OF SYSTEM AND YEAR</u></b></center>

|   year |   (0.0, 2.5] |   (2.5, 5.0] |   (5.0, 7.5] |   (7.5, 10.0] |   (10.0, 12.5] |   (12.5, 17.5] |   (17.5, 42.5] |   (42.5, 18000.0] |
|-------:|-------------:|-------------:|-------------:|--------------:|---------------:|---------------:|---------------:|------------------:|
|   1998 |     40.6729  |      7.61333 |      5.11    |     nan       |      nan       |      nan       |      nan       |        nan        |
|   1999 |     14.5398  |      9.15847 |      7.86083 |       5.95923 |        5.845   |        5.2     |      nan       |        nan        |
|   2000 |     16.6478  |     10.1969  |      9.39538 |       5.54889 |        7.03    |      nan       |      nan       |        nan        |
|   2001 |     14.7452  |     11.2984  |     10.1574  |       6.9413  |        5.53    |        1.94    |        1.45    |        nan        |
|   2002 |     14.7635  |     11.3523  |      9.40317 |       5.7861  |        4.69968 |        5.67    |        1.17    |        nan        |
|   2003 |     13.3395  |     10.8438  |      9.86159 |       8.77286 |        3.34257 |        1.65444 |        1.22    |        nan        |
|   2004 |     12.8288  |     10.1207  |      9.42945 |       8.80782 |        5.52425 |        2.81833 |        1.06857 |          0.3925   |
|   2005 |     14.4218  |      9.54932 |      8.86249 |       8.92223 |        6.61143 |        2.55059 |        1.23909 |        nan        |
|   2006 |     11.7171  |      9.75206 |      9.07418 |       8.73899 |        6.22678 |        4.573   |        1.37273 |          0.29     |
|   2007 |     11.2895  |      9.83285 |      9.34322 |       9.09753 |        6.78621 |        2.11622 |        1.05231 |          0.318889 |
|   2008 |     11.3415  |      9.53851 |      8.99735 |       8.66592 |        7.46377 |        2.34    |        1.2388  |          0.37     |
|   2009 |     14.6225  |      9.2317  |      8.64841 |       8.166   |        7.17618 |        4.32281 |        1.40699 |          0.295    |
|   2010 |     15.6106  |      8.07208 |      7.42675 |       7.06973 |        6.73548 |        5.13536 |        1.4858  |          0.297647 |
|   2011 |      9.70845 |      7.47239 |      6.74946 |       6.20922 |        5.51875 |        4.10329 |        1.63742 |          0.285965 |
|   2012 |      8.72542 |      6.2594  |      5.68543 |       5.27859 |        4.86973 |        4.14255 |        1.6635  |          0.328537 |
|   2013 |      7.55915 |      5.44898 |      4.75281 |       4.38301 |        4.14832 |        3.78718 |        1.7666  |          0.224524 |
|   2014 |      7.80135 |      5.10343 |      3.9748  |       3.46374 |        3.15748 |        2.97587 |        1.69372 |          0.278475 |
|   2015 |      7.37186 |      4.95681 |      4.19053 |       3.76074 |        3.52809 |        3.25315 |        2.19498 |          0.226508 |
|   2016 |      6.63728 |      4.76325 |      4.30698 |       3.95878 |        3.70554 |        3.40448 |        2.53374 |          0.339263 |
|   2017 |      5.7474  |      4.41521 |      3.97419 |       3.59832 |        3.36909 |        3.06048 |        2.51272 |          0.474576 |
|   2018 |      5.69595 |      4.23095 |      3.86835 |       3.43824 |        3.2137  |        2.94957 |        2.42031 |          0.47925  |

#### DATA ENGINEERING
- Total adjusted installed  cost = 
total installed cost with adjustments made for inflation
- cost per watt =  $\frac {total\ adjusted\ installed\ cost}{system\ size} $
- Resampled data into weekly medians
- Nulls were replaced with median values from same year
- Created groupings for cost per watt, system size and total adjusted installed cost 
  
### <center><u>TIME SERIES PREPARATION</u></center>

Data is best when X variables(predictors) are independent and un-correlated.
#### SEASONALITY & TREND
Data did not show any signs of <u>seasonality</u> or <u>trends</u>. <br>
However, data was not initially <u>stationary</u>. 

 #### STATIONARITY
Ensuring a series is stationary makes the forecasts more reliable.
Within this data, <u>stationarity</u> had to be achieved by taking the difference in change from one week to the next, aka <u>differencing</u> the data. <u>Differencing</u> can help stabilize the mean of a time series by removing changes in the level of a time series, and therefore eliminating (or reducing) any <u>trend</u> and <u>seasonality</u>.

### <center><u>REGRESSION MODELS</u></center>
Regression was used as a means to reference how ARIMA was performing on the data when compared to basic regressors. Inititially, six types of regressors were used:
- Random Forest Regressor
- Bagging Regressor
- Linear Regression
- AdaBoost Regression 
- OLS Linear 
- OLS 

After multiple tests, Bagging, AdaBoost, and Linear Regressors did not perform.<br> <u>Random Forest</u>, <u>OLS Linear</u>, and <u>OLS</u> produced the most promising results and were kept in the model for comparison. 

### <u><center>TIME SERIES MODEL</center></u>
AR forecasts are essentially linear regression models which utilize lags much in the way OLS Linear Regression does. 
 
 #### ARIMA<br> 

|  ARIMA |     Meaning     |    Parameters [p,d,q]   |   Notes     |
|-------:|----------------:|---------------:|---------------:|
|   AR |      AutoRegressive  |      [p]| number of lags pf Y to be used as predictors|
|   I |      Integrated  |      [d]| minimum number of differencing to make stationary number of lags|
|   MA |      Moving Average  |      [q] | order of the moving average term <br> number of lagged forecast errors |
 
 Notes:<br>
 If time series is already stationary, d=0. <br>
 Used ARIMA given that the lags that would likely help 


### <center><u>PERFORMANCE</u></center>
Out of the eight or so applicable error metrics (Mean Absolute Percentage Error, Mean Error, Mean Absolute Error, Mean Percentage Error,Root Mean Squared Error, Autocorrelation of Error,Correlation between the Actual and the Forecast,Min-Max Error), I chose to use Mean Squared Error as the metric to score both the regression and the ARIMA models. ACF was also taken into consideration in some initial time series dilligence and trials.

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