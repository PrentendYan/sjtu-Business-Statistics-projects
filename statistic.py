import pandas as pd
import numpy as np
import os
from pprint import pprint

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.width', 180)

data_per_month = pd.read_csv(os.path.join('statistic','电子电器股价（月）.csv'))
data_per_year = pd.read_csv(os.path.join('statistic','电子电器股价（年）.csv'))
fiscal_index = pd.read_csv(os.path.join('statistic','after_cleaning','Fiscal_Index.csv'))
GDP = pd.read_csv(os.path.join('statistic','after_cleaning','GDP.csv'))
IFO = pd.read_csv(os.path.join('statistic','after_cleaning','IFO.csv'))
Inflation = pd.read_csv(os.path.join('statistic','after_cleaning','Inflation.csv'))
Rates = pd.read_csv(os.path.join('statistic','after_cleaning','Rates.csv'))

index_month = data_per_month.keys()[0]
index_year = data_per_year.keys()[0]

def sum_data(data,index):
    data = data.sort_values(by = index , ascending = False)
    data = data[0:10]

    data.loc['sum'] = data.apply(lambda x: x.sum())

    dataset = dict(data.loc['sum'])
    return dataset

stock_dataset_per_month = sum_data(data_per_month,index_month)
stock_dataset_per_year = sum_data(data_per_year,index_year)

def turn_into_seasons(data):
    cnt = 0
    season = []
    item_sum = 0
    for key,item in data.items():
        cnt += 1
        item_sum += item
        if cnt%3 == 0:
            season.append(item_sum/3)
            item_sum = 0
    return season

def divide_into_seasons(data):
    season = []
    for key,item in data.items():
        for _ in range(4):
            season.append(item)
    season.append(0)
    season.append(0)
    return season

date = ['2022y2s','2022y1s','2021y4s','2021y3s','2021y2s','2021y1s','2020y4s','2020y3s','2020y2s','2020y1s','2019y4s','2019y3s','2019y2s','2019y1s','2018y4s','2018y3s','2018y2s','2018y1s']

income_of_centre_goverment = divide_into_seasons(fiscal_index['income_of_centre_goverment'])
income_of_local_goverment = divide_into_seasons(fiscal_index['income_of_local_goverment'])
debt_of_local_goverment = divide_into_seasons(fiscal_index['debt_of_local_goverment'])
GDP = GDP['GDP']
PMI = turn_into_seasons(IFO['PMI'])
Consumer_Confidence_Index = turn_into_seasons(IFO['Consumer_Confidence_Index'])
CPI = turn_into_seasons(Inflation['CPI'])
PPI = turn_into_seasons(Inflation['PPI'])
SHIBOR = turn_into_seasons(Rates['SHIBOR'])
Treasury_Bill_Rate = turn_into_seasons(Rates['Treasury_Bill_Rate'])
exchange_rate = turn_into_seasons(Rates['exchange_rate'])
stock_dataset_per_season = turn_into_seasons(stock_dataset_per_month)
add_all = sum(stock_dataset_per_season)
for i in range(len(stock_dataset_per_season)):
    stock_dataset_per_season[i] = stock_dataset_per_season[i]/add_all

dataset = pd.DataFrame()
columns = ['date','income_of_centre_goverment','income_of_local_goverment','debt_of_local_goverment','GDP','PMI','Consumer_Confidence_Index','CPI','PPI','SHIBOR','Treasury_Bill_Rate','exchange_rate','stock_dataset_per_season']

for i in columns:
    dataset[i] = eval(i)

