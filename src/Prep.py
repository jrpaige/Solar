import numpy as np
import pandas as pd
import cpi
import sys
import matplotlib.pyplot as plt
from pytictoc import TicToc
from statsmodels.tsa.stattools import adfuller
# t = TicToc()

# t.tic()
# t.toc()

file_path_1 = '/Users/jenniferpaige/Desktop/TTS_10-Dec-2019_p1.csv'
file_path_2 = '/Users/jenniferpaige/Desktop/TTS_10-Dec-2019_p2.csv'

# =============================================================================
# DF PREP
# =============================================================================
def prep(stationary_only= True):
    
    '''
    ==Columns==
    Installation Date, System Size, Total Installed Price, Customer Segment
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
    
    == Parameters==
    |stationary_only|:     
        if stationary_only == True:
            will only return df that is stationary
        else: 
            will return both final-non-stationary and final-stationary dfs
            
    '''
    
    print("1 of 14 |    Reading in first dataset. \n             Using 4/60 features/columns: 'Installation Date', 'System Size', 'Total Installed Price' , 'Customer Segment' \n             Changing -9999 values to null")
    dfMod1 = pd.read_csv(file_path_1,
                    encoding='iso-8859-1',
                    parse_dates=['Installation Date'], 
                    usecols=['Installation Date', 'System Size','Total Installed Price' , 'Customer Segment'], 
                    na_values=(-9999, '-9999'))
   

    print("2 of 14 |    Reading in second dataset. \n             Using 4/60 features/columns: 'Installation Date', 'System Size', 'Total Installed Price' , 'Customer Segment' \n             Changing -9999 values to null")
    
    dfMod2 = pd.read_csv(file_path_2,
                    encoding='iso-8859-1',
                    parse_dates=['Installation Date'], 
                    usecols=['Installation Date', 'System Size', 'Total Installed Price' , 'Customer Segment'], 
                    na_values=(-9999, '-9999'))
    print('3 of 14 |    Concatenating datasets together')  
    dfMod = pd.concat([dfMod1,dfMod2], ignore_index=True)
    df = dfMod.copy()

    print('4 of 14 |    Refining to only RES Customer Segment')
 
    df = df.loc[df['Customer Segment']=='RES']
    
    print('5 of 14 |    Cleaning up column names')
    
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    
    print('6 of 14 |    Sorting values by installation_date')
    df.sort_values('installation_date', inplace=True)
    
    print('7 of 14 |    Assigning installation_date as index')
    df.set_index('installation_date', drop=True, inplace=True)
    

    print('8 of 14 |    Replacing all null values with median values from same year')
    [df['total_installed_price'].replace(np.nan, 
                                         round(df.loc[(df['total_installed_price'] != np.nan) & 
                                        (df.index.year == i)]['total_installed_price'].median(),2), 
                                        inplace=True) for i in range(1998,2019)] 
    
    print('9 of 14 |    Adusting prices for inflation')
    df['date'] = df.index.date
    df['adj_installed_price'] = round(df.apply(lambda x: cpi.inflate(x.total_installed_price, x.date), axis=1),2)

    print('10 of 14|    Creating target variable: cost_per_watt')
    df['cost_per_watt'] = round(df['adj_installed_price']/ df['system_size']/1000,2)
    
    
    print("11 of 14|    Removing outliers above $25 per watt") 
    df = df.loc[df.cost_per_watt < 25]

    print("12 of 14|    Resampling data into weekly medians and cropping dataframe so only continuous non-null data remains")
    null_list = []
    for i in range(len(df.cost_per_watt.resample('W').median())):
        if df.cost_per_watt.resample('W').median()[i] >0:
            pass
        else:
            null_list.append(i)
    df = df.cost_per_watt.resample('W').median()[null_list[-1]+1:]
    
    print("13 of 14|    Testing for stationarity") 
    if round(adfuller(df)[1],4) < 0.51:
        print("ADF P-value: {} \n Time Series achieved stationarity! Reject ADF H0.".format(round(adfuller(df)[1],4)))
        print("14 of 14|    Prep complete \n -----------------------------------------------")
        if stationary_only == True:
            return df
        elif stationary_only==False:
            return df, df
    else:
        print('             ADF P-value: {} \n Time Series is not stationary. \n              Fail to reject ADF H0'.format(round(adfuller(df)[1],4)))
        print("14 of 14|    Creating differenced data to achieve stationarity") 
        weekly_differences = df.diff(periods=1).dropna()
        print("             Testing for stationarity on differenced data.")
        if round(adfuller(weekly_differences)[1],4) < 0.51: 
            print("             ADF P-value: {} \n             Differenced data achieved stationarity! Reject ADF H0.".format(round(adfuller(weekly_differences)[1],4)))
        if stationary_only== True:
              return weekly_differences
        elif stationary_only== False:
              return df, weekly_differences
            
    print('Prep complete \n ------------------------------------------------------------')

    
    
    
# =============================================================================
# MAX, MEAN, MEDIAN, MODE, & MIN EDA ON PRICE 
# =============================================================================    
def price_eda(df):    
    print('MAX PRICE \n total            $',  df['adj_installed_price'].max(), '\n cost per watt    $',  df['cost_per_watt'].max())
    print('MEAN PRICE \n total            $', round( df['adj_installed_price'].mean(),2), '\n cost per watt    $', round( df['cost_per_watt'].mean(),2))
    print('MEDIAN PRICE \n total            $',  df['adj_installed_price'].median(), '\n cost per watt    $',  df['cost_per_watt'].median())
    print('MODE PRICE \n total            $',  df['adj_installed_price'].mode()[0], '\n cost per watt    $',  df['cost_per_watt'].mode()[0])
    print('MIN PRICE \n total            $',  df['adj_installed_price'].min(), '\n cost per watt    $',  df['cost_per_watt'].min())
    df.cost_per_watt.plot(title=('Cost per Watt'))
    plt.show()
    df.adj_installed_price.plot(title=('Total Installed Price (Adjusted for Inflation)'))



