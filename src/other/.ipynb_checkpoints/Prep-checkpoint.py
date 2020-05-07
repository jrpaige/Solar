import numpy as np
import pandas as pd
import cpi
import sys
import matplotlib.pyplot as plt
from pytictoc import TicToc
from statsmodels.tsa.stattools import adfuller
from src.Time_Series_Helper_Funcs import rolling_plot
# t = TicToc()

# t.tic()
# t.toc()

file_path_1 = '/Users/jenniferpaige/Desktop/TTS_10-Dec-2019_p1.csv'
file_path_2 = '/Users/jenniferpaige/Desktop/TTS_10-Dec-2019_p2.csv'

# =============================================================================
# DF PREP
# =============================================================================
def prep(eda=False, show_rolling_plot=False):
    
    '''
    ==Function==
    Returns orginal dataframe and stationary dataframe
    
    ==Columns==
    Installation Date, System Size, Total Installed Price, Customer Segment,State
    Lowercases and replaces all spaces with underscores
    
    ==Index==
    Cronologically Ordered by Installation Date
    
    ==Customer Segment==
    Residential is 95% of the data 
    Non-Residential pricing could vary wildly due to government, incentives, etc...
    Focuses on Residential to eliminate any irregularity 

    ==Null Values==
    The num. of null price cells[356171] / the num. of price cells[1543831] = > 20%
    Nulls were entered in as '-9999'
    Replaces -9999 with np.nan
    Replaces all null values with the median from that year 

    ==Inflation Adjustment==
    Adjusts all prices to reflect inflation of the current year(2019)
    The adjustment is made using data provided by The Bureau of Labor Statistics at the U.S. Department of Labor.
    
    ==Target Variable==
    create target variable, cost_per_watt 
    cost_per_watt = total installed price/system size/1000
    
    ==Outliers==
    Removes > 1600 outlier observations which reflect >$25/watt
    
    ==Resample into Weekly medians==
    Resamples data into the weekly medians into continuous non-null df

    == Stationarity==
    Uses ADFuller method to test for stationarity.
    Prints ADF results. 
    If data is not stationary, uses differencing to achieve stationarity.
    Completes ADF testing again
    
    ==Parameters==
    |eda| :
        if eda == True:
            will print table of Max, Min, Mean, Mode, Median Prices within data.
        default = False
        
    |show_rolling_plot| 
        if show_rolling_plot==True:
            will show plot of weekly rolling mean, median, and std. dev with 3 windows 
        default = False
     
    RETIRED-----
    |stationary_only|:     
        if stationary_only == True:
            will only return df that is stationary
        else: 
            will return both final-non-stationary and final-stationary pandas dfs
            default = False
            
    '''
    print('PREP'.center(76,'-'))
    
    print(" 1 of 12 |    Reading in data \n         |    Filtering to 5 features:\n         |       Date, System Size, Total Cost, Customer Segment, State \n         |    Changing -9999 values to null")
    dfMod1 = pd.read_csv(file_path_1,
                    encoding='iso-8859-1',
                    parse_dates=['Installation Date'], 
                    usecols=['Installation Date', 'System Size','Total Installed Price', 'State',
                    'Customer Segment'], na_values=(-9999, '-9999'))
    dfMod2 = pd.read_csv(file_path_2,
                    encoding='iso-8859-1',
                    parse_dates=['Installation Date'], 
                    usecols=['Installation Date', 'System Size', 'Total Installed Price' , 'State',
                    'Customer Segment'],na_values=(-9999, '-9999'))
    df = pd.concat([dfMod1,dfMod2], ignore_index=True)

    
    print(' 2 of 12 |    Cleaning up column names')
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    
    
    print(' 3 of 12 |    Refining to only RES Customer Segment')
    df = df.loc[df['customer_segment']=='RES']
    

    print(' 4 of 12 |    Sorting values by installation_date\n         |    Assigning installation_date as index')
    df.sort_values('installation_date', inplace=True)
    df.set_index('installation_date', drop=True, inplace=True)
    
    
    print(' 5 of 12 |    Replacing all null values with median values from same year')
    [df['total_installed_price'].replace(np.nan,round(df.loc[(df['total_installed_price'] != np.nan) & (df.index.year == i)]['total_installed_price'].median(),2),inplace=True) for i in range(1998,2019)] 
    
    
    print(' 6 of 12 |    Adusting prices for inflation')
    df['date'] = df.index.date
    df['adj_installed_price'] = round(df.apply(lambda x:cpi.inflate(x.total_installed_price, x.date), axis=1),2)

    
    print(' 7 of 12 |    Creating target variable: cost_per_watt')
    df['cost_per_watt'] = round(df['adj_installed_price']/ df['system_size']/1000,2)
    

    print(' 8 of 12 |    Removing outliers above $25 per watt') 
    df = df.loc[df.cost_per_watt < 25]
    

    print(' 9 of 12 |    Resampling data into weekly medians\n         |    Cropping dataframe to keep only continuous non-null data')
    null_list = []
    for i in range(len(df.cost_per_watt.resample('W').median())):
        if df.cost_per_watt.resample('W').median()[i] >0:
            pass
        else:
            null_list.append(i)
    y = df.cost_per_watt.resample('W').median()[null_list[-1]+1:]
    
    print('10 of 12 |    Testing for stationarity')
    if round(adfuller(y)[1],4) < 0.51:
        print("         |       ADF P-value: {} \n         |       Time Series achieved stationarity. \n         |       Reject ADF".format(round(adfuller(y)[1],4)))
        print('prep complete'.upper().center(76,'-'))
#         if eda==True:
#             print('EDA '.upper().center(76,' '))
#             print(eda_price(df))        
        return df, y

    else:
        print('         |       ADF P-value: {} \n         |       Time Series is not stationary.   \n         |       Fail to reject ADF H0'.format(round(adfuller(y)[1],4)))
        print('11 of 12 |    Creating differenced data to achieve stationarity')

        weekly_differences = y.diff(periods=1).dropna()
        
        print('12 of 12 |    Testing for stationarity on differenced data') 
        
        if round(adfuller(weekly_differences)[1],4) < 0.51: 
            print('         |       ADF P-value: {} \n         |       Differenced data achieved stationarity. \n         |       Reject ADF H0'.format(round(adfuller(weekly_differences)[1],4)))
        print('prep complete'.upper().center(76,'-'))
        if show_rolling_plot==True:
            print(rolling_plot(weekly_differences))
            print("".center(76, '-'))
        if eda==True:
            print('EDA '.upper().center(76,' '))
            print(eda_price(df)) 
        return pd.DataFrame(df), pd.DataFrame(weekly_differences)
    
    
    
# =============================================================================
# MAX, MEAN, MEDIAN, MODE, & MIN EDA ON PRICE 
# =============================================================================    


def eda_price(df):
    price = pd.DataFrame()
    price['Max'] = "${:,.2f}".format(df['adj_installed_price'].max()), "${:,.2f}".format(df['cost_per_watt'].max())
    price['Mean']= "${:,.2f}".format(df['adj_installed_price'].mean()) , "${:,.2f}".format(df['cost_per_watt'].mean())
    price['Median']= "${:,.2f}".format(df['adj_installed_price'].median()), "${:,.2f}".format(df['cost_per_watt'].median())
    price['Mode']= "${:,.2f}".format(df['adj_installed_price'].mode()[0]), "${:,.2f}".format(df['cost_per_watt'].mode()[0])
    price['Min']= "${:,.2f}".format(df['adj_installed_price'].min()), "${:,.2f}".format(df['cost_per_watt'].min())
    price.index = 'Total Cost', 'Cost Per Watt'
    return price
