from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd


# second, prepare text samples and their labels
# Import CSV
df_raw_1 = pd.read_csv('./output/GOOGL_data_with_news_3.csv')
df_raw_1['tagged_news'] = df_raw_1['tagged_news'].replace(np.nan, '', regex=True)
df_with_targets_1 = df_raw_1.copy()
predict_num_days_ahead = 2
df_with_targets_1['target_reg'] = df_with_targets_1['open'].shift(-predict_num_days_ahead)
df_with_targets_1 = df_with_targets_1.iloc[:-1]
df_with_targets_1['target_class'] = df_with_targets_1['target_reg'] > df_with_targets_1['close']
df_1 = df_with_targets_1.copy()
# df_1 = df_1.dropna()
print(df_1[['main_news', 'tagged_news']])

df_raw_2 = pd.read_csv('./output/Facebook_with_news.csv')
df_raw_2['tagged_news'] = df_raw_2['tagged_news'].replace(np.nan, '', regex=True)
df_with_targets_2 = df_raw_2.copy()
df_with_targets_2['target_reg'] = df_with_targets_2['open'].shift(-predict_num_days_ahead)
df_with_targets_2 = df_with_targets_2.iloc[:-1]
df_with_targets_2['target_class'] = df_with_targets_2['target_reg'] > df_with_targets_2['close']
df_2 = df_with_targets_2.copy()
# df_2 = df_2.dropna()

df_raw_3 = pd.read_csv('./output/Microsoft_with_news.csv')
df_raw_3['tagged_news'] = df_raw_3['tagged_news'].replace(np.nan, '', regex=True)
df_with_targets_3 = df_raw_3.copy()
df_with_targets_3['target_reg'] = df_with_targets_3['open'].shift(-predict_num_days_ahead)
df_with_targets_3 = df_with_targets_3.iloc[:-1]
df_with_targets_3['target_class'] = df_with_targets_3['target_reg'] > df_with_targets_3['close']
df_3 = df_with_targets_3.copy()
# df_3 = df_3.dropna()

df_combined = pd.concat([df_1, df_2, df_3])
df_combined.to_csv('combined_news.csv')
