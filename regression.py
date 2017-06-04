import pandas as pd
import quandl

quandl.ApiConfig.api_key = 'VDxfKkzAm8MFL1fZsSat'
df = quandl.get_table('WIKI/PRICES')

print(df.head())