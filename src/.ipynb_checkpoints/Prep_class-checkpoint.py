import numpy as np
import pandas as pd
import cpi
import sys
# import matplotlib.pyplot as plt
# from pytictoc import TicToc
# from statsmodels.tsa.stattools import adfuller
# t = TicToc()

# t.tic()
# t.toc()

file_path_1 = '/Users/jenniferpaige/Desktop/TTS_10-Dec-2019_p1.csv'
file_path_2 = '/Users/jenniferpaige/Desktop/TTS_10-Dec-2019_p2.csv'
files = [file_path_1, file_path_2]

# =============================================================================
# DF PREP
# =============================================================================
class Prep():
    def __init__(self, files=files):
        """
        ==Parameters==
        |files| : entered in as a list
        """
        self.self = self
        self.files = files
        self.loaded_files = []
        
    def load_full(self):
        print("Reading in dataset(s). \n             Using 4/60 features/columns:'Installation Date', 'System Size', 'Total Installed Price' , 'Customer Segment'\n             Changing -9999 values to null")
        count = 0
        self.loaded_files = []
        for i in range(1, len(self.files)+1):
            exec("df{} = pd.read_csv(files[{}],encoding='iso-8859-1',  parse_dates=['Installation Date'], usecols=['Installation Date','System Size','Total Installed Price','Customer Segment'],             na_values=(-9999, '-9999'))".format(i, count))
            count+=1
            [exec("self.loaded_files.append(df{})".format(i)) for i in range(1, len(self.files)+1)]
               
        if len(self.loaded_files) > 1:
            print('Concatenating datasets together')  
            df = pd.concat([self.loaded_files],ignore_index=True)
            return df
        else: 
            df=loaded_files[0]
            return df

# df= Prep(files)
# df.load_full()
    
    
    