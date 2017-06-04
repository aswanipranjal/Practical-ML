import pandas as pd
import quandl
import math

quandl.ApiConfig.api_key = 'VDxfKkzAm8MFL1fZsSat'
df = quandl.get_table('WIKI/PRICES')
# print(df.head())
df = df[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
df['hl_pct'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100
df['pct_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100
# The ones below are features
df = df[['adj_close', 'hl_pct', 'pct_change', 'adj_volume']]
# print(df.head())

forecast_col = 'adj_close'
# fills specified value wherever 'fill' is not available
# In machine learning, we don't want to have null data. We could remove the 
# entire column, but that would lead to 'wastage' of data in the other attributes
# An outlandish value in place of the unavailable data is more or less regarded as an 
# outlying value and will be 'ignored'
# this is better than sacrificing data
df.fillna(-99999, inplace=True)

# here we are trying to predict out 10% of the dataframe
forecast_out = int(math.ceil(0.01*len(df)))

# we are basically creating a space ten% days out into the future, thus the negative shift
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)
print(df.head())