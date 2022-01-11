#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:30:52 2021

@author: fanyang
"""

import pandas_datareader.data as web
import pandas as pd
#pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
import datetime

import warnings
warnings.filterwarnings("ignore")

#function to get the price data from yahoo finance
def getDataBatch(tickers, startdate, enddate):
    def getData(ticker):
        return(pdr.get_data_yahoo(ticker, start = startdate, end = enddate))
    datas = map(getData, tickers)
    return(pd.concat(datas, keys = tickers, names = ['Ticker','Date']))

# define the time period
start_dt = datetime.datetime(2000, 1, 1)
end_dt = datetime.datetime(2020, 12, 31)

#
# Get the Famama French data using pandas data reader
# Famma French 3 factor model
ds = web.DataReader('F-F_Research_Data_Factors_Daily','famafrench',
                    start = start_dt, end = end_dt)
df_FamaFrench = ds[0]
df_FamaFrench.columns = ['MKT','SMB','HML','RF']
df_FamaFrench['RF'] = 100 * df_FamaFrench['RF'] #scale riskfree rates to be in % as other factors
# moment factor
ds2 = web.DataReader('F-F_Researh_Data_Factors_Daily','famafrench',
                     start = start_dt, end = end_dt)
df2 = ds2[0]
df2.columns = ['MoM']
#combine the factors together
df_FamaFrench = pd.concat([df_FamaFrench,df2],axis=1)

#------------------------------------------------------
# Get stock price from yahoo finance
#------------------------------------------------------
# for one stock/index case
# ^GSPC is the ticker for S&P500
SP500 = pdr.get_data_yahoo('^GSPC', start = start_dt, end = end_dt)
ret_Stock = SP500['Adj Close'].pct_change().dropna()

# for multiple stock cases
tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG', '^GSPC', 'C', 'GE', 'PG', 'CELG', 'DIOD', 'FCNTX']
stock_data = getDataBatch(tickers, start_dt, end_dt)
#isolate the'Adj close' values and transform the DataFrame
daily_close_px = stock_data[['Adj Close']].reset_index().pivot(index = 'Date', columns = 'Ticker', values = 'Adj Close')
#calculate the daily percentage change for 'daily_close_px'
daily_pct_change = daily_close_px.pct_change().dropna()
daily_pct_change.columns = ['AAPL', 'MSFT', 'IBM', 'GOOG', 'SP500', 'C', 'GE', 'PG', 'CELG', 'DIOD', 'Contra']

# # merge the return with FamaFrench
# df_all = pd.concat([df_FamaFrench],daily_pct_change*100),axis = 1).dropna()
    
#------------------------------------------------------
# Get data from Fred
#------------------------------------------------------
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2020, 12, 31)

cpi = web.DataReader("CPIAUCSL", "fred", start, end)
Baa_Yeild = web.DataReader("BAA10Y","fred",start, end)

#------------------------------------------------------
# Get China stock data from Baostock.com
#------------------------------------------------------
import baostock as bs
import pandas as pd

# 登陆系统
lg = bs.login(user_id = "anonymous", password = "123456")
# 显示登录返回信息
print(lg.error_code)
print(lg.error_msg)
# 显示详细指标参数
rs = bs.query_history_k_data("sh.601398",
                             "date,code,open,high,low,close,volume,amount,adjustflag",
                             start_date='2021-11-01',end_date='2021-11-25',
                             frequency="d",adjustflag="3")
print(rs.error_code)
print(rs.error_msg)
# 获取具体的信息
result_list = []
while (rs.error_code == "0") & rs.next():
    # 分页查询，将每页信息合在一起
    result_list.append(rs.get_row_data())
result = pd.DataFrame(result_list, columns=rs.fields)
result.to_csv("history_k_data_fanyang.csv",encoding="gbk",index = False)
print(result)
# 登出系统
bs.logout()





