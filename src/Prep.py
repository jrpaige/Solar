import numpy as np
import pandas as pd
import cpi
import sys
import datetime
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import normaltest

cpi.update()

# =============================================================================
# DF PREP
# =============================================================================
class Prep():
    '''
    ==Function==
    Prepare data for time series analysis 
    
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
    
    compile
    ==Returns==
    Stationary dataframe 
    
    ==Prints==
    3 sets of 3 plots
    data, PACF, ACF
    '''

        
    def __init__(self):
        self.self = self
        self.files = ['/Users/jenniferpaige/code/DSI/getit/TTS_10-Dec-2019_p1.csv','/Users/jenniferpaige/code/DSI/getit/TTS_10-Dec-2019_p2.csv']
        
        cpi.update()
        
    def load(self):
        loaded_files = []
        count = 0
        print('PREP'.center(76,'-'))
        print(" 1 of 11 |    Reading in data \n         |    Filtering to 4 features:\n         |       Date, System Size, Total Cost, Customer Segment \n         |    Changing -9999 values to null")
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
        print(' 2 of 11 |    Cleaning up column names')
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        return df

    def refine(self):
        df = self.clean()
        print(' 3 of 11 |    Refining to only RES Customer Segment')
        df = df.loc[df['customer_segment']=='RES']
        return df

    def date_index(self):
        df = self.refine()
        print(' 4 of 11 |    Sorting values by installation_date\n         |    Assigning installation_date as index')
        df.sort_values('installation_date', inplace=True)
        df.set_index('installation_date', drop=True, inplace=True)
        return df
    
    def null_handler(self):
        df = self.date_index()
        print(' 5 of 11 |    Replacing all null values with median values from same year')
        [df['total_installed_price'].replace(np.nan,round(df.loc[(df['total_installed_price'] != np.nan) & (df.index.year == i)]['total_installed_price'].median(),2),inplace=True) for i in range(1998,2019)] 
        return df

    def inflation_assist(self):
        df = self.null_handler()
        print(' 6 of 11 |    Adusting prices for inflation')
        df['date'] = df.index.date
        df['adj_installed_price'] = round(df.apply(lambda x:cpi.inflate(x.total_installed_price, x.date), axis=1),2)
        return df

    def target_variable(self):
        df = self.inflation_assist()
        print(' 7 of 11 |    Creating target variable: cost_per_watt')
        df['cost_per_watt'] = round(df['adj_installed_price']/ df['system_size']/1000,2)
        return df

    def resampler(self):
        df = self.target_variable()
        print(' 8 of 11 |    Resampling data into weekly medians\n         |    Cropping dataframe to keep only continuous non-null data')
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
        print(' 9 of 11 |    Testing for stationarity')
        if round(adfuller(y)[1],4) < 0.51:
            print("         |       ADF P-value: {} \n         |       Time Series achieved stationarity. \n         |       Reject ADF H0".format(round(adfuller(y)[1],4)))
            print('prep complete'.upper().center(76,'-'))
            return y, self.differ_plots(y), self.rolling_plots(y)
        else:
            print('         |       ADF P-value: {} \n         |       Time Series is not stationary.   \n         |       Fail to reject ADF H0'.format(round(adfuller(y)[1],4)))
            print('10 of 11 |    Creating differenced data to achieve stationarity')
            first_ord = y.diff().dropna()
            print('11 of 11 |    Testing for stationarity on differenced data')
            if round(adfuller(first_ord)[1],4) < 0.51:
                print('         |       ADF P-value: {} \n         |       Differenced data achieved stationarity. \n         |       Reject ADF H0'.format(round(adfuller(first_ord)[1],4)))
                print('prep complete'.upper().center(76,'-'))
                return first_ord, self.differ_plots(y)#, self.rolling_plots(differences)
            else:
                print('After differencing, data is still not stationary. \
                Consider applying other methods.')
                print('prep complete'.upper().center(76,'-'))
                return first_ord,self.differ_plots(y), #self.rolling_plots(differences)
            
        
    def differ_plots(self, y):
        # Original Series
        fig, axes = plt.subplots(3, 3, constrained_layout=True, figsize=(15,10))
        axes[0, 0].plot(y); axes[0, 0].set_title('Original Series')
        plot_acf(y, ax=axes[0, 1])
        plot_pacf(y, ax=axes[0, 2])
        # 1st Differencing
        axes[1, 0].plot(y.diff()); axes[1, 0].set_title('1st Order Differencing')
        plot_acf(y.diff().dropna(), ax=axes[1, 1])
        plot_pacf(y.diff().dropna(),ax=axes[1, 2])
        # 2nd Differencing
        axes[2, 0].plot(y.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
        plot_acf(y.diff().diff().dropna(), ax=axes[2, 1])
        plot_pacf(y.diff().diff().dropna(),ax=axes[2, 2])
        plt.show()
    
    def compile(self):
        first_ord = self.stationarity()[0]
        return pd.DataFrame(first_ord)

if __name__ == "__main__":   
    df = Prep().compile()
    