#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 21:50:10 2021

@author: fanyang
"""
import pandas as pd
#from pandas import ExcelWriter
#from pandas import ExcelFile
import numpy as np
import datetime

import warnings
warnings.filterwarnings("ignore")


# This function does two tasks:
# 1. Given input dates with the format of YYYYMM, identify the month end date for each of the dates
# 2. Converts the given dates into YYYYMMDD with the month end days added
def addMthEndDaytoYearMonth(x):
    import calendar
    #year=int(x[0:4])
    year = int(x/100)
    month = int(x - year*100)
    # find what is the last day of the month for the given year and month
    day = calendar.monthrange(year, month)[1]
    # Create new dates by adding the month end days to the known year and month
    DateWithMthEndDate = datetime.date(year,month,day)
    return DateWithMthEndDate

# read in data from an excel xlsx file
df_Factor = pd. read_excel('FamaFrenchFactorReturns.xlsx', sheet_name = 'FamaFrench4FactorHistData_Month',
                           header = 3, index_col=0)
print(df_Factor.columns)

df_Regime = pd.read_excel('InputDataRegimeFlag.xlsx',sheet_name = 'Flag',
                          header = 0, index_col = 0)
print(df_Regime.columns)

FactorDate = pd.DataFrame(df_Factor.index)
FactorDateWDay = FactorDate.apply(addMthEndDaytoYearMonth,axis=1)
df_Factor.index = FactorDateWDay
# Change the format of the index from datetime to dateindex so that it can be matched with the Regime dataframe
df_Factor.index = pd.to_datetime(df_Factor.index)
Name_Factors = df_Factor.columns

# merging/joining dataframes
frames = [df_Regime,df_Factor]
# merge the two frames by simply combining them
df_merged = pd.concat(frames) #default axis is 0, which means merge the columns, keep all the row
df_merged_C = pd.concat(frames, axis = 1, join = 'inner' ) # merge only the common rows
df_merged_C1 = pd.concat(frames, axis = 1, join = 'outer')

# Calculate conditional expected return and risk
df_merged_C_RiskOn = df_merged_C[df_merged_C['RiskOn Flag']]
del df_merged_C_RiskOn['RiskOn Flag']
mean_ret_RiskOn = df_merged_C_RiskOn.mean()
cov_ret_RiskOn = df_merged_C_RiskOn.cov()
df_merged_C_RiskOff = df_merged_C[~df_merged_C['RiskOn Flag']]
del df_merged_C_RiskOff['RiskOn Flag']
mean_ret_RiskOff = df_merged_C_RiskOff.mean()
cov_ret_RiskOff = df_merged_C_RiskOff.cov()

# Create dataframes to be exported to excel
mean_data = np.vstack((mean_ret_RiskOn,mean_ret_RiskOff))
df_mean = pd.DataFrame(mean_data, columns = Name_Factors, index=np.transpose(['RiskOn', 'RiskOff']))
df_cov_RiskOn = pd.DataFrame(cov_ret_RiskOn, columns = Name_Factors, index = np.transpose(Name_Factors))
df_cov_RiskOff = pd.DataFrame(cov_ret_RiskOff, columns = Name_Factors, index = np.transpose(Name_Factors))

# Create a pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('Output_RegimeBasedRetNRisk.xlsx',engine = 'xlsxwriter')

# Write each dataframe to a different worksheet

# cell_format1.set_num_format('0.000')
# cell_format05.set_num_format('mm/dd/yyyy')
# cell_format03.set_num_format('#,##0.00')

df_mean.to_excel(writer, sheet_name='Returns',
                 startrow=1, startcol=0, header=True, index=True)
df_cov_RiskOn.to_excel(writer, sheet_name='CovRiskOn',
                       startrow=1, startcol=0, header=True, index=True)
df_cov_RiskOff.to_excel(writer, sheet_name='CovRiskOff',
                        startrow=1, startcol=0, header=True, index=True)

# Get the xlsxwriter objects from the dataframe writer ojbect.
workbook = writer.book

# define desired formats for excel cells.
cell_format_bold = workbook.add_format({'bold':True, 'italic':True, 'font_color':'red'})
cell_format1 = workbook.add_format({'font_color':'black','num_format':'0.00'})

# Saving mean returns to the first sheet
worksheet1 = writer.sheets['Returns']
worksheet1.write('A1','Mean Monthly Regime Returns',cell_format_bold)
worksheet1.conditional_format('B3:F4',{'type':'3_color_scale','format':cell_format1})
worksheet1.conditional_format('B3:F4',{'type':'cell',
                                       'criteria':'>',
                                       'value':-10000,
                                       'format':cell_format1
                                       })

# saving covariance matrix to the second sheet
worksheet2 = writer.sheets['CovRiskOn']
worksheet2.write(0,0,'RiskOn Monthly Covariance Matrix',cell_format_bold)
worksheet2.conditional_format('B3:F10',{'type':'3_color_scale','format':cell_format1})
worksheet2.conditional_format('B3:F10',{'type':'cell',
                                        'creteria':'>',
                                        'value':-10000,
                                        'format':cell_format1})

writer.save()



