Trend 
	• movement relatively higher/lower values over long periods of time 
	• Happens for some time and then disappears
	• No trend = horizontal or steady
	

Seasonality
	• Upward/downward swings
	• Repeating pattern within fixed time period 
	

Irregularity / Noise/ Residual
	• Erratic or unsystematic 
	•  short duration 
	• Non-repeating 
	• Random variation 

Cyclic
	• Repeating up and down movements over more than a year
	• No fixed pattern 
	• Can happen anytime 
	
Time series requires stationarity 

Stationarity
	• When statistical properties do not change over time
	• Series is stationary when three conditions are met:
		○ By Trend [constant mean]
			○  Variation of specific time frames
		○ By Seasonality [constant variance]
			○ Variation of specific time frames
			○ Distance from the mean should be equal  
		○ By covariance is independent of time
			○ Autocovariance does not depend on time  
Unit Root
	• Feature of some stochastic processes that can cause problems in statistical inference
	• A linear stochastic process has a unit root, if 1 is a root of the process's characteristic equation.
	• Such a process is non-stationary but does not always have a trend.
Dickey-Fuller Test
	• determines if a series is stationary or not.
	• H0 : a unit root is present in a time series sample
	• If unit root is present:
		○ p> 0
		○ process is not stationary
	• If unit root is not present:
		○ p=0
		○ reject null hyp
		○ process is stationary

Exponential smoothing
	• MA with decreasing weight for each observation
	• less importance to observations further back
	• smoothing factor = alpha
		○ 0 < value > 1
		○ determines how fast the weight decreases
		○ as smoothing factor approaches 0, the model turns into
		○ the smaller smoothing factor, the smaller the time series
Double
	• if a trend is present in the time series, use double exponential smoothing
	• smoothing factor = beta
		○ 0 < value > 1
Triple
	• if seasonality is present in the time series, use triple exponential smoothing
	• seasonal smoothing factor = gamma
	• length of the season = L


Autoregression  (ARIMA) [parameter P] 
	• Autoregressive lags
	• regression of the time series onto itself
	• assumption:
		○ current value depends on previous values with some lag.
	• param p = maximum lag after which most lags are not significant
		○ located on partial autocorrelation plot

Integration (ARIMA) [parameter d] 
	• param d = number of differences required to make the series stationary
	• Order of differentiation 

Moving Average (ARIMA) [parameter Q] 
	• param q = biggest lag after which other lags are not significant
		○ located on autocorrelation plot
 	• next observation is mean of all past observations
	• indentify interesting trends
	• The longer the window, smoother the trend

residual - difference between predicted data points and actual data points from
least squares - takes outliers into account which are weighted heavier


use lag of 3 previous weeks to predict the next week 
update amounts for inflation



ARIMA


I = Correlation between previous time period to the current 
If there is any correlation between current time period and a previous one, that is autoregressive


Time series requires stationarity 
Is stationary if three conditions are met
	- Constant mean
	- Constant variance
	- Autocovariance does not depend on time 
Check for stationarity
	- Rolling statistics
		○ Moving average/ moving variance
			§ See if it varies with time window
			§ Instance T will take average or variance of a time 
			§ Visual technique 
		○ Augmented Dickey Fuller Test 
			§ H0 = TS is non stationary
			§ Test results = 
				□ Test statistic 
				□ Critical values for different confidence levels 
					® If test statistic is less than critical value:
						◊ Reject H0 and say series is stationary 
ARIMA
	Figure out the noise 
		Average it out 
			Cross and drop set of _____ in the noise smoothens out 
				Can have average focused on noise 
				Take average now  
				
				
If trying to predict P:
	Use PS EF (partial autocorrelation graph 
If trying to predict Q:
	Use CF (autocorrelation plot)
If trying to predict d
	Order of differentiation defines the value d 
	

Don't use time series if values are constant 
Also if you have values in the form of functions


