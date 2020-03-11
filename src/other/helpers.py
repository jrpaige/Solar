import pandas as pd
import numpy as np


#def df_year_split(df, year, drop_columns):   
#     return df[df['Year']==year].drop(columns=drop_columns)\
#                             .reset_index().drop(columns='index')


def time_series_me(df1,df2,df3,df_for_y):
  
    time_series_df = df1.copy()
    
    time_series_df = df1.set_index('installation_date')\
                        .join(df2.set_index('installation_date'),lsuffix='_1', rsuffix='_2',how='outer')\
                        .join(df3.set_index('installation_date'),lsuffix='_2', rsuffix='_3',how='outer')\
                        .join(df_for_y.set_index('installation_date'),lsuffix='_3', rsuffix='_4',how='outer')\
                        .reset_index()
 
 
    X = time_series_df.drop(columns='price_per_system_w', inplace=True)
    y = time_series_df['price_per_system_w']
    
    return time_series_df,X,y


def residual_plot(ax, x, y, y_hat, n_bins=50):
    residuals = y_hat - y
    ax.axhline(0, color="black", linestyle="--")
    ax.scatter(x, residuals, color="grey", alpha=0.5)
    ax.set_ylabel("Residuals ($\hat y - y$)")


def plot_many_residuals(df, var_names, y_hat, n_bins=50):
    fig, axs = plt.subplots(len(var_names), figsize=(12, 3*len(var_names)))
    for ax, name in zip(axs, var_names):
        x = df[name]
        residual_plot(ax, x, df['price_per_system_w'], y_hat)
        ax.set_xlabel(name)
        ax.set_title("Model Residuals by {}".format(name))
    return fig, axs