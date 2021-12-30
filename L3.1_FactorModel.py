#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 21:25:09 2021

This code performs the following tasks:
    1. load or download equity data from Yahoo and factor return data from Ken French Website
    2. run regression models of the equity returns vs. the factor returns, from 1 factor to 4 factors
    3. Export the model coefficient to an excel file

@author: fanyang
"""

#from pandas_datareader.famafrench import get_available_datasets
#dsa = get_available_datasets()

import pandas_datareader.data as web
import pandas as pd
#pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()
import datetime

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import matplotlib.pyplot as plt
import numpy as np

#function to get the price data from yahoo finance
def getDataBatch(tickers, startdate, enddate):
    def getData(ticker):
        return(pdr.get_data_yahoo(ticker, startdate, enddate))
    datas = map(getData, tickers)
    return(pd.concat(datas, keys=tickers, names=['tickers','Date']))

#------------------------------------------------------------------------
flag_downloadData = False # If True download data from web. Otherwise read in from the excel.

if flag_downloadData:
    # define the time period
    start_dt = datetime.datetime(2000, 1, 1)
    end_dt = datetime.datetime(2018,12,31)
    
    # get fama french data using pandas data reader
    # fama french 3 factor model
    ds = web.DataReader('F-F_Research_Data_Factors_daily','famafrench',
                        start = start_dt, end=end_dt)
    df_FamaFrench = ds[0]
    df_FamaFrench.columns = ['MKT', 'SMB', 'HML', 'RF']
    df_FamaFrench['RF'] = 100*df_FamaFrench['RF']
    # momentum factor
    ds2 = web.DataReader('F-F_Momentum_Factor_daily','fammafrench',
                         start=start_dt,end=end_dt)
    df2 = ds2[0]
    df2.columns = ['MoM']
    # combine the factors together
    df_FamaFrench = pd.concat([df_FamaFrench,df2],axis = 1)
    
    #---------------------------------------------------
    # get stock price and find returns
    # for one stock case,
    stocks = pdr.get_data_yahoo('^GSPC', start=start_dt, end=end_dt)
    # calculate returns
    ret_stocks = stocks['Adj Close'].pct_change().dropna()
    
    # for multiple stocks cases,
    tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG', '^GSPC', 'C', 'GE', 'PG', 'CELG', 'DIOD', 'FCNTX']
    all_data = getDataBatch(tickers, start_dt, end_dt)
    # Isolalte the 'Adj Close' values and transform the DataFrame
    daily_close_px = all_data[['Adj Close']].reset_index().pivot('Date','Ticker','Adj Close')
    # calculate the daily percentage change for 'daily_close_px'
    daily_pct_change = daily_close_px.pct_change().dropna()
    daily_pct_change.cloumns = ['AAPL', 'MSFT', 'IBM', 'GOOG', '^GSPC', 'C', 'GE', 'PG', 'CELG', 'DIOD', 'FCNTX']
    
    # merge the return with famafrench
    df_all = pd.concat([daily_pct_change*100,df_FamaFrench],axis = 1).dropna()
    
    # subtract risk free rate from factors
    # risk free rate needs to be divided by 250 because the data is daily
    df_all_lessRF = df_all.copy()
    # df_all['MKT'] = df_all['MKT']-1/250*df_all['RF']
    for dfnames in df_all_lessRF.columns:
        if dfnames not in ['MoM','SMB', 'HML', 'RF']:
            df_all_lessRF[dfnames]=df_all_lessRF[dfnames]-1/250*df_all['RF']
            
    # save data to excel
    writer = pd.ExcelWriter('InputforRegression.xlsx',engine='xlsxwriter')
    daily_pct_change.to_excel(writer,'StockReturn')
    df_FamaFrench.to_excel(writer,'FactorReturn')
    df_all.to_excel(writer,'AllReturn')
    df_all_lessRF.to_excel(writer,'AllReturnLessRF')
    writer.save()
    
else:
    daily_pct_change = pd.read_excel('InputforRegression.xlsx',sheet_name = 'StockReturn')
    df_FamaFrench = pd.read_excel('InputforRegression.xlsx',sheet_name='FactorReturn')
    df_all = pd.read_excel('InputforRegression.xlsx',sheet_name='AllReturn')
    df_all_lessRF = pd.read_excel('InputforRegression.xlsx',sheet_name='AllReturnLessRF')
    
    
#--------------------------------------------
# Linear Regression, CAPM model, using SP as market proxy


#----------------APPL-----------------------------
aapl_1f = smf.ols(formula='AAPL~MKT',data = df_all_lessRF).fit()
print(aapl_1f.summary())
aapl_1f_params = aapl_1f.params
aapl_1f_standardError = aapl_1f.bse
aapl_1f_PredValue = aapl_1f.predict()
aapl_1f_FittedValue = aapl_1f.fittedvalues
aapl_1f_Residual = aapl_1f.resid
aapl_1f_SumSqaureResidual = aapl_1f.ssr

#plot the data together with the linear regression fit

ret_MKT_sorted = np.sort(df_all_lessRF['MKT'])
#calculate the model 'predicted returns'for Apple given the market factor returns
ret_AAPL_sorted_calc = aapl_1f_params[0] + aapl_1f_params[1]*ret_MKT_sorted

plt.figure()
plt.plot(df_all_lessRF['MKT'],df_all_lessRF['AAPL'],'o',label='raw data')
plt.plot(ret_MKT_sorted, ret_AAPL_sorted_calc,'-r',label='CAPM model fit')
plt.xlabel('Market Returns', fontsize=12)
plt.ylabel('Apple Returns', fontsize=12)
plt.legend()

aapl_3f = smf.ols(formula = 'AAPL ~ MKT+SMB+HML',data = df_all_lessRF).fit()
print( aapl_3f.summary())

aapl_4f = smf.ols(formula = 'AAPL ~ MKT+SMB+HML+MoM',data = df_all_lessRF).fit()
print( aapl_4f.summary())

#--------------value stock-------------------------
pg_3f = smf.ols(formula='PG ~ MKT+SMB+HML', data=df_all_lessRF).fit()
print(pg_3f.summary())

pg_4f = smf.ols(formula='PG ~ MKT+SMB+HML+MoM', data=df_all_lessRF).fit()
print(pg_4f.summary())

#--------------Fidelity Contra Fund----------------
contra_3f = smf.ols(formula='Contra~MKT+SMB+HML', data=df_all_lessRF).fit()
print(contra_3f.summary())

contra_4f = smf.ols(formula = 'Contra~MKT+SMB+HML+MoM', data=df_all_lessRF).fit()
print(contra_4f.summary())


# Save data to a dataframe to be exported to excel files
Name_factors_save = ['Intercept','MKT','SMB','HML','MoM']
Name_stocks_save = ['AAPL','PG','Contra']
Parameters_save = np.vstack((aapl_4f.params.transpose(),pg_4f.params.transpose(),contra_4f.params.transpose()))
df_save = pd.DataFrame(Parameters_save,columns=Name_factors_save,index=Name_stocks_save)

# Write out results
writer=pd.ExcelWriter('Output_Coefficients.xlsx', engine='xlsxwriter')
df_save.to_excel(writer,sheet_name='FactorModel',
                 startrow=1, startcol=0, header=True, index=True)

# get the xlsxwriter object from the dataframe writer object
workbook = writer.book

# define desired formats for excel cells
cell_format_bold = workbook.add_format({'bold':True,'italic':True,'font_color':'red'})
cell_format1 = workbook.add_format({'font_color':'black','num_format':'0.00'})


worksheet1 = writer.sheets['FactorModel']
worksheet1.write('A1','Historical Beta Coefficients',cell_format_bold)
worksheet1.conditional_format('B3:F4',{'type':'cell',
                                    'criteria':'>',
                                       'value':-10000,
                                      'format':cell_format1})


        
    
    
    
    
    
    
    
    