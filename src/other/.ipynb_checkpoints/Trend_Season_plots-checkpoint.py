import numpy as np
import pandas as pd
import cpi
import sys
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
import plotly.express as px
plt.style.use('ggplot')

def trend_seasonality_plots(df):
    plotdf = pd.DataFrame(df.cost_per_watt)
    print('Check for any apparent signs of trends') 
    #by Month
    plotdf.groupby([plotdf.index.year, plotdf.index.month]).median().plot()
    plt.title('Median Cost per Watt per Month')
    plt.ylabel('Cost ($)')

    #by Week
    plotdf.groupby([plotdf.index.year, plotdf.index.weekofyear]).median().plot()
    plt.title('Median Cofst per Watt per Week')
    plt.ylabel('Cost ($)')

    #by Quarter
    plotdf.groupby([plotdf.index.year, plotdf.index.quarter]).median().plot()
    plt.title('Median Cost per Watt per quarter')
    plt.ylabel('Cost ($)')
    
    print('Check for any apparent signs of seasonality')
    #Check for any apparent signs of seasonality by Quarter
    plotdf.groupby([plotdf.index.quarter]).median().plot()
    plt.title('Median Cost Per Watt Throughout 4 Quarters in a Year')
    plt.ylabel('Cost ($)')

    #Check for any apparent signs of seasonality by Week
    plotdf.groupby([plotdf.index.weekofyear]).median().plot()
    plt.title('Median Cost Per Watt Throughout 52 Weeks in a Year')
    plt.ylabel('Cost ($)')

    #Check for any apparent signs of seasonality by Month
    plotdf.groupby([plotdf.index.month]).median().plot()
    plt.title('Median Cost Per Watt Throughout 12 Months in a Year')
    plt.ylabel('Cost ($)')
    