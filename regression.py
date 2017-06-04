import pandas as pd
import quandl

quandl.ApiConfig.api_key = 'VDxfKkzAm8MFL1fZsSat'
df = quandl.get_table('WIKI/PRICES')
# print(df.head())
df = df[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
df['hl_pct'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100
df['pct_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100
df = df[['adj_close', 'hl_pct', 'pct_change', 'adj_volume']]
print(df.head())