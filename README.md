# Time Series Forecasting of Residential Solar Panel Costs 

## ABSTRACT

The relationship between technology, climate change, and awareness relative to renewable energy creates interesting nuances for the future of the energy sector. Over 1.6 million observations collected from 1998 - 2018 of solar panel installations were used in the project, of which 95% consisted of residential installations. The project’s model employs univariate ARIMA time series analysis to generate a forecast for a cost_per_watt target variable. The exploratory aspect of the project provided an opportunity to understand time series analysis on a granular level.  

Tech Stack Used: 
Python, Numpy, Pandas, Scikit-Learn, Matplotlib, Math, SciPy, StatsModels, Pyramid,  CPI

 
Fun Fact about you or your project:
The data includes over 60 features/columns related to price, taxes/rebate programs, technology, location, manufacturing, install specs etc.

## RESULTS

## WORKFLOW:
- DATA + EDA
- DATA ENGINEERING
- TIME SERIES PREPARATION
  - STATIONARITY
- MODELS
- PERFORMANCE
- RESULTS + SCORES
- NEXT STEPS



### DATA + EDA

#### <u>AVERAGE COST PER WATT FOR RESIDENTIAL CUSTOMERS GIVEN SIZE OF SYSTEM AND YEAR</u>

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


### DATA ENGINEERING
  inflation
  cost per watt
  dealing with NaNs
### TIME SERIES PREPARATION

  - STATIONARITY
    moved to differences as data set 

### MODELS
  ### ARIMA
  ### OLS
  ### RANDOM FOREST REGRESSOR
  ### LINEAR REGRESSION 

### PERFORMANCE


### RESULTS + SCORES


### NEXT STEPS
