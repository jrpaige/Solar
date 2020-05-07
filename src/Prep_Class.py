import numpy as np
import pandas as pd
import cpi
import sys
from statsmodels.tsa.stattools import adfuller

# =============================================================================
# DF PREP
# =============================================================================
class Prep():
    '''
    ==Function==
    Returns stationary dataframe
    
    ==Columns==
    Installation Date, System Size, Total Installed Price, Customer Segment,State
    Lowercases and replaces all spaces with underscores
    
    ==Index==
    Cronologically Ordered by Installation Date
    
    ==Customer Segment==
    Residential is 95% of the data 
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
    
    ==Outliers==
    Removes outlier observations which reflect >$25/watt
    
    ==Resample into Weekly medians==
    Resamples into the weekly medians and continuous non-null df

    == Stationarity==
    Uses ADFuller method to test for stationarity.
    If data is not stationary, uses differencing to achieve stationarity.
    Completes ADF testing again
    
    compile
    '''
    
    def __init__(self, files):
        """
        ==Parameters==
        |files| : entered in as a list
        """
        self.self = self
        self.files = files
        #self.df = pd.DataFrame()
#         self.y = pd.DataFrame()
#         self.differences = pd.DataFrame()

    def load(self):
        """ 
        ==Function==
        takes in as many csvs as needed, filters to 5 columns and concatenates together
        """
        loaded_files = []
        count = 0
        print('PREP'.center(76,'-'))
        print(" 1 of 12 |    Reading in data \n         |    Filtering to 5 features:\n         |       Date, System Size, Total Cost, Customer Segment, State \n         |    Changing -9999 values to null")
        for i in range(1, len(self.files)+1):
            exec(f"df{i} = pd.read_csv(self.files[{count}],encoding='iso-8859-1',parse_dates=['Installation Date'], usecols=['Installation Date','System Size', 'Total Installed Price','Customer Segment', 'State'],na_values=(-9999, '-9999'))")
            count+=1
            exec(f"loaded_files.append(df{i})")
        if len(loaded_files) > 1:
            df = pd.concat([i for i in loaded_files],ignore_index=True)
        else: 
            df=loaded_files[0]

        return df

    def clean(self):
        """ 
        ==Function==
        Cleans up column names to 
            - all lower
            - replaces ' ' with '_'
        """
        df = self.load()
        print(' 2 of 12 |    Cleaning up column names')
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        return df

    def refine(self):
        """ 
        ==Function==
        Refines to only Residential customer segment    
        """
        df = self.clean()
        print(' 3 of 12 |    Refining to only RES Customer Segment')
        df = df.loc[df['customer_segment']=='RES']
        return df

    def date_index(self):
        """ 
        ==Function==
        Sorts values by installtion date
        Assigns installation date as index
        """
        df = self.refine()
        print(' 4 of 12 |    Sorting values by installation_date\n         |    Assigning installation_date as index')
        df.sort_values('installation_date', inplace=True)
        df.set_index('installation_date', drop=True, inplace=True)
        return df
    
    def null_handler(self):
        """ 
        ==Function==
        Replaces all null values with median values from same year
        """
        df = self.date_index()
        print(' 5 of 12 |    Replacing all null values with median values from same year')
        [df['total_installed_price'].replace(np.nan,round(df.loc[(df['total_installed_price'] != np.nan) & (df.index.year == i)]['total_installed_price'].median(),2),inplace=True) for i in range(1998,2019)] 
        return df

    def inflation_assist(self):
        """ 
        ==Function==
        Adjusts prices for inflation
        """
        
        df = self.null_handler()
        print(' 6 of 12 |    Adusting prices for inflation')
        df['date'] = df.index.date
        df['adj_installed_price'] = round(df.apply(lambda x:cpi.inflate(x.total_installed_price, x.date), axis=1),2)
        return df

    def target_variable(self):
        """
        ===Function===
        Creates target variable `cost_per_watt`
            = adjusted installed price / system size
        """
        
        df = self.inflation_assist()
        print(' 7 of 12 |    Creating target variable: cost_per_watt')
        df['cost_per_watt'] = round(df['adj_installed_price']/ df['system_size']/1000,2)
        return df

    def outlier_removal(self):
        """
        ===Function===
        Removes outliers about $25 per watt
        """
        
        df = self.target_variable()
        print(' 8 of 12 |    Removing outliers above $25 per watt') 
        df = df.loc[df.cost_per_watt < 25]
        return df

    def resampler(self):
        """
        ===Function===
        Resamples data into weekly medians
        Crops dataframe to keep only continuous non-null data

        ===Returns===
        y 
        """
        
        df = self.outlier_removal()
        print(' 9 of 12 |    Resampling data into weekly medians\n         |    Cropping dataframe to keep only continuous non-null data')
        null_list = []
        for i in range(len(df.cost_per_watt.resample('W').median())):
            if df.cost_per_watt.resample('W').median()[i] >0:
                pass
            else:
                null_list.append(i)
        y = df.cost_per_watt.resample('W').median()[null_list[-1]+1:]
        return y

    def stationarity(self):
        """
        ===Function===
        Tests for stationarity using ADF 
        If data is not stationary:
            uses differencing  and retests
        """
        
        y = self.resampler()
        print('10 of 12 |    Testing for stationarity')
        if round(adfuller(y)[1],4) < 0.51:
            print("         |       ADF P-value: {} \n         |       Time Series achieved stationarity. \n         |       Reject ADF H0".format(round(adfuller(y)[1],4)))
            print('prep complete'.upper().center(76,'-'))
            return y
        else:
            print('         |       ADF P-value: {} \n         |       Time Series is not stationary.   \n         |       Fail to reject ADF H0'.format(round(adfuller(y)[1],4)))
            print('11 of 12 |    Creating differenced data to achieve stationarity')
            differences = y.diff(periods=1).dropna()
            print('12 of 12 |    Testing for stationarity on differenced data') 
            if round(adfuller(differences)[1],4) < 0.51:
                print('         |       ADF P-value: {} \n         |       Differenced data achieved stationarity. \n         |       Reject ADF H0'.format(round(adfuller(differences)[1],4)))
                print('prep complete'.upper().center(76,'-'))
                return differences
            else:
                print('After differencing, data is still not stationary. \
                Consider applying other methods.')
                print('prep complete'.upper().center(76,'-'))
                return differences
    def compile(self):
        df = self.stationarity()
        return pd.DataFrame(df)
    
file_path_1 = '/Users/jenniferpaige/Desktop/TTS_10-Dec-2019_p1.csv'
file_path_2 = '/Users/jenniferpaige/Desktop/TTS_10-Dec-2019_p2.csv'
files = [file_path_1, file_path_2]
if __name__ == "__main__":   
    df = Prep(files).compile()