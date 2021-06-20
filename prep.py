#!/usr/bin/env python
# coding: utf-8

# # Part 1, perfect features


import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)

from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from xgboost import plot_importance


def plot_features(booster, figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


import time
import sys
import gc
import pickle
import random

random.seed(0)
np.random.seed(0)

start_time = time.time()


items = pd.read_csv("./input/items.csv")
shops = pd.read_csv("./input/shops.csv")
cats = pd.read_csv("./input/item_categories.csv")
train = pd.read_csv("./input/sales_train.csv")
# set index to ID to avoid droping it later
test = pd.read_csv("./input/test.csv").set_index("ID")


# ## Outliers

# There are items with strange prices and sales. After detailed exploration I decided to remove items with price > 100000 and sales > 1001 (1000 is ok).

train = train[train.item_price < 100000]
train = train[train.item_cnt_day < 1001]


# There is one item with price below zero. Fill it with median.

median = train[
    (train.shop_id == 32)
    & (train.item_id == 2973)
    & (train.date_block_num == 4)
    & (train.item_price > 0)
].item_price.median()
train.loc[train.item_price < 0, "item_price"] = median


# Several shops are duplicates of each other (according to its name). Fix train and test set.


# shops[(shops.shop_id==10) | (shops.shop_id==11)]


# Якутск Орджоникидзе, 56
train.loc[train.shop_id == 0, "shop_id"] = 57
test.loc[test.shop_id == 0, "shop_id"] = 57
# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, "shop_id"] = 58
test.loc[test.shop_id == 1, "shop_id"] = 58
# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, "shop_id"] = 11
test.loc[test.shop_id == 10, "shop_id"] = 11


# ## Shops/Cats/Items preprocessing
# Observations:
# * Each shop_name starts with the city name.
# * Each category contains type and subtype in its name.


shops.loc[
    shops.shop_name == 'Сергиев Посад ТЦ "7Я"', "shop_name"
] = 'СергиевПосад ТЦ "7Я"'
shops["city"] = shops["shop_name"].str.split(" ").map(lambda x: x[0])
shops.loc[shops.city == "!Якутск", "city"] = "Якутск"
shops["city_code"] = LabelEncoder().fit_transform(shops["city"])
shops = shops[["shop_id", "city_code"]]

cats["split"] = cats["item_category_name"].str.split("-")
cats["type"] = cats["split"].map(lambda x: x[0].strip())
cats["type_code"] = LabelEncoder().fit_transform(cats["type"])
# if subtype is nan then type
cats["subtype"] = cats["split"].map(
    lambda x: x[1].strip() if len(x) > 1 else x[0].strip()
)
cats["subtype_code"] = LabelEncoder().fit_transform(cats["subtype"])
cats = cats[["item_category_id", "type_code", "subtype_code"]]

items.drop(["item_name"], axis=1, inplace=True)


# ## Monthly sales
# Test set is a product of some shops and some items within 34 month. There are 5100 items * 42 shops = 214200 pairs. 363 items are new compared to the train. Hence, for the most of the items in the test set target value should be zero.
# In the other hand train set contains only pairs which were sold or returned in the past. Tha main idea is to calculate monthly sales and <b>extend it with zero sales</b> for each unique pair within the month. This way train data will be similar to test data.

ts = time.time()
matrix = []
cols = ["date_block_num", "shop_id", "item_id"]
for i in range(34):
    sales = train[train.date_block_num == i]
    matrix.append(
        np.array(
            list(product([i], sales.shop_id.unique(), sales.item_id.unique())),
            dtype="int16",
        )
    )

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)
matrix["shop_id"] = matrix["shop_id"].astype(np.int8)
matrix["item_id"] = matrix["item_id"].astype(np.int16)
matrix.sort_values(cols, inplace=True)
print("time cost:", time.time() - ts)


# Aggregate train set by shop/item pairs to calculate target aggreagates, then <b>clip(0,20)</b> target value. This way train target will be similar to the test predictions.
#
# <i>I use floats instead of ints for item_cnt_month to avoid downcasting it after concatination with the test set later. If it would be int16, after concatination with NaN values it becomes int64, but foat16 becomes float16 even with NaNs.</i>


train["revenue"] = train["item_price"] * train["item_cnt_day"]


ts = time.time()
group = train.groupby(["date_block_num", "shop_id", "item_id"]).agg(
    {"item_cnt_day": ["sum"]}
)
group.columns = ["item_cnt_month"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=cols, how="left")
matrix["item_cnt_month"] = (
    matrix["item_cnt_month"]
    .fillna(0)
    .clip(0, 20)  # NB clip target here
    .astype(np.float16)
)
print("time cost:", time.time() - ts)


# ## Test set
# To use time tricks append test pairs to the matrix.

test["date_block_num"] = 34
test["date_block_num"] = test["date_block_num"].astype(np.int8)
test["shop_id"] = test["shop_id"].astype(np.int8)
test["item_id"] = test["item_id"].astype(np.int16)


ts = time.time()
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True)  # 34 month
print("time cost:", time.time() - ts)


# ## Shops/Items/Cats features


ts = time.time()
matrix = pd.merge(matrix, shops, on=["shop_id"], how="left")
matrix = pd.merge(matrix, items, on=["item_id"], how="left")
matrix = pd.merge(matrix, cats, on=["item_category_id"], how="left")
matrix["city_code"] = matrix["city_code"].astype(np.int8)
matrix["item_category_id"] = matrix["item_category_id"].astype(np.int8)
matrix["type_code"] = matrix["type_code"].astype(np.int8)
matrix["subtype_code"] = matrix["subtype_code"].astype(np.int8)
print("time cost:", time.time() - ts)


# ## Traget lags
#
# 即使資料去除前12個月，還是會有 NaN 出現，
# 因為 lag_feature 是依照 "date_block_num", "shop_id", "item_id" 去合併的。


def lag_feature(df, lags, col):
    """
    df(DataFrame)
    lags(list)
    col(string)
    """
    tmp = df[["date_block_num", "shop_id", "item_id", col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = [
            "date_block_num",
            "shop_id",
            "item_id",
            col + "_lag_" + str(i),
        ]
        shifted["date_block_num"] += i
        df = pd.merge(
            df, shifted, on=["date_block_num", "shop_id", "item_id"], how="left"
        )
    return df


ts = time.time()
matrix = lag_feature(matrix, [1, 2, 3], "item_cnt_month")
print("time cost:", time.time() - ts)


# ## Mean encoded features

# date_avg_item_cnt: 對每個月的物品銷量做平均


ts = time.time()
group = matrix.groupby(["date_block_num"]).agg({"item_cnt_month": ["mean"]})
group.columns = ["date_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num"], how="left")
matrix["date_avg_item_cnt"] = matrix["date_avg_item_cnt"].astype(np.float16)
matrix = lag_feature(matrix, [1], "date_avg_item_cnt")
matrix.drop(["date_avg_item_cnt"], axis=1, inplace=True)
print("time cost:", time.time() - ts)


# date_item_avg_item_cnt: 對每個月的每個商品的銷量做平均


ts = time.time()
group = matrix.groupby(["date_block_num", "item_id"]).agg({"item_cnt_month": ["mean"]})
group.columns = ["date_item_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "item_id"], how="left")
matrix["date_item_avg_item_cnt"] = matrix["date_item_avg_item_cnt"].astype(np.float16)
matrix = lag_feature(matrix, [1, 2, 3], "date_item_avg_item_cnt")
matrix.drop(["date_item_avg_item_cnt"], axis=1, inplace=True)
print("time cost:", time.time() - ts)


# date_shop_avg_item_cnt: 對每個月的每個商店的銷量做平均


ts = time.time()
group = matrix.groupby(["date_block_num", "shop_id"]).agg({"item_cnt_month": ["mean"]})
group.columns = ["date_shop_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "shop_id"], how="left")
matrix["date_shop_avg_item_cnt"] = matrix["date_shop_avg_item_cnt"].astype(np.float16)
matrix = lag_feature(matrix, [1, 2, 3], "date_shop_avg_item_cnt")
matrix.drop(["date_shop_avg_item_cnt"], axis=1, inplace=True)
print("time cost:", time.time() - ts)


# date_cat_avg_item_cnt: 對每個月的每個商品類別的銷量做平均


ts = time.time()
group = matrix.groupby(["date_block_num", "item_category_id"]).agg(
    {"item_cnt_month": ["mean"]}
)
group.columns = ["date_cat_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "item_category_id"], how="left")
matrix["date_cat_avg_item_cnt"] = matrix["date_cat_avg_item_cnt"].astype(np.float16)
matrix = lag_feature(matrix, [1], "date_cat_avg_item_cnt")
matrix.drop(["date_cat_avg_item_cnt"], axis=1, inplace=True)
print("time cost:", time.time() - ts)


# date_shop_cat_avg_item_cnt: 對每個月的每個商店的每個物品類別id的銷量做平均


ts = time.time()
group = matrix.groupby(["date_block_num", "shop_id", "item_category_id"]).agg(
    {"item_cnt_month": ["mean"]}
)
group.columns = ["date_shop_cat_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(
    matrix, group, on=["date_block_num", "shop_id", "item_category_id"], how="left"
)
matrix["date_shop_cat_avg_item_cnt"] = matrix["date_shop_cat_avg_item_cnt"].astype(
    np.float16
)
matrix = lag_feature(matrix, [1], "date_shop_cat_avg_item_cnt")
matrix.drop(["date_shop_cat_avg_item_cnt"], axis=1, inplace=True)
print("time cost:", time.time() - ts)


# date_shop_type_avg_item_cnt: 對每個月的每個商店的每個類別做平均


ts = time.time()
group = matrix.groupby(["date_block_num", "shop_id", "type_code"]).agg(
    {"item_cnt_month": ["mean"]}
)
group.columns = ["date_shop_type_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(
    matrix, group, on=["date_block_num", "shop_id", "type_code"], how="left"
)
matrix["date_shop_type_avg_item_cnt"] = matrix["date_shop_type_avg_item_cnt"].astype(
    np.float16
)
matrix = lag_feature(matrix, [1], "date_shop_type_avg_item_cnt")
matrix.drop(["date_shop_type_avg_item_cnt"], axis=1, inplace=True)
print("time cost:", time.time() - ts)


# date_shop_subtype_avg_item_cnt: 對每個月的每個商店的每個子類別做平均


ts = time.time()
group = matrix.groupby(["date_block_num", "shop_id", "subtype_code"]).agg(
    {"item_cnt_month": ["mean"]}
)
group.columns = ["date_shop_subtype_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(
    matrix, group, on=["date_block_num", "shop_id", "subtype_code"], how="left"
)
matrix["date_shop_subtype_avg_item_cnt"] = matrix[
    "date_shop_subtype_avg_item_cnt"
].astype(np.float16)
matrix = lag_feature(matrix, [1], "date_shop_subtype_avg_item_cnt")
matrix.drop(["date_shop_subtype_avg_item_cnt"], axis=1, inplace=True)
print("time cost:", time.time() - ts)


# date_city_avg_item_cnt: 對每個月的每個城市的銷量做平均


ts = time.time()
group = matrix.groupby(["date_block_num", "city_code"]).agg(
    {"item_cnt_month": ["mean"]}
)
group.columns = ["date_city_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "city_code"], how="left")
matrix["date_city_avg_item_cnt"] = matrix["date_city_avg_item_cnt"].astype(np.float16)
matrix = lag_feature(matrix, [1], "date_city_avg_item_cnt")
matrix.drop(["date_city_avg_item_cnt"], axis=1, inplace=True)
print("time cost:", time.time() - ts)


# date_item_city_avg_item_cnt: 對每個月每個物品在每個城市的銷量做平均


ts = time.time()
group = matrix.groupby(["date_block_num", "item_id", "city_code"]).agg(
    {"item_cnt_month": ["mean"]}
)
group.columns = ["date_item_city_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(
    matrix, group, on=["date_block_num", "item_id", "city_code"], how="left"
)
matrix["date_item_city_avg_item_cnt"] = matrix["date_item_city_avg_item_cnt"].astype(
    np.float16
)
matrix = lag_feature(matrix, [1], "date_item_city_avg_item_cnt")
matrix.drop(["date_item_city_avg_item_cnt"], axis=1, inplace=True)
print("time cost:", time.time() - ts)


# date_type_avg_item_cnt: 對每個月每個類別的銷量做平均


ts = time.time()
group = matrix.groupby(["date_block_num", "type_code"]).agg(
    {"item_cnt_month": ["mean"]}
)
group.columns = ["date_type_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "type_code"], how="left")
matrix["date_type_avg_item_cnt"] = matrix["date_type_avg_item_cnt"].astype(np.float16)
matrix = lag_feature(matrix, [1], "date_type_avg_item_cnt")
matrix.drop(["date_type_avg_item_cnt"], axis=1, inplace=True)
print("time cost:", time.time() - ts)


# date_subtype_avg_item_cnt: 對每個月每個子類別的銷量做平均


ts = time.time()
group = matrix.groupby(["date_block_num", "subtype_code"]).agg(
    {"item_cnt_month": ["mean"]}
)
group.columns = ["date_subtype_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "subtype_code"], how="left")
matrix["date_subtype_avg_item_cnt"] = matrix["date_subtype_avg_item_cnt"].astype(
    np.float16
)
matrix = lag_feature(matrix, [1], "date_subtype_avg_item_cnt")
matrix.drop(["date_subtype_avg_item_cnt"], axis=1, inplace=True)
print("time cost:", time.time() - ts)


# ## Trend features

# Price trend for the last six months.
#
# delta_price_lag: 跟過去六個月跟整體平均價是上升還是下降(但第一個月若是下降就直接回傳下降，即使2~6都是上升)


ts = time.time()
group = train.groupby(["item_id"]).agg({"item_price": ["mean"]})  # 每個物品的平均價格
group.columns = ["item_avg_item_price"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["item_id"], how="left")
matrix["item_avg_item_price"] = matrix["item_avg_item_price"].astype(np.float16)

group = train.groupby(["date_block_num", "item_id"]).agg(
    {"item_price": ["mean"]}
)  # 每個月每個物品的平均價格
group.columns = ["date_item_avg_item_price"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "item_id"], how="left")
matrix["date_item_avg_item_price"] = matrix["date_item_avg_item_price"].astype(
    np.float16
)

lags = [1, 2, 3]
matrix = lag_feature(matrix, lags, "date_item_avg_item_price")  # 過去每個月每個物品的平均價格

for i in lags:  # 總體平均價格 與 上n月的平均價格 的比率
    matrix["delta_price_lag_" + str(i)] = (
        matrix["date_item_avg_item_price_lag_" + str(i)] - matrix["item_avg_item_price"]
    ) / matrix["item_avg_item_price"]


def select_trend(row):
    for i in lags:
        if row["delta_price_lag_" + str(i)]:
            return row["delta_price_lag_" + str(i)]
    return 0


matrix["delta_price_lag"] = matrix.apply(select_trend, axis=1)
matrix["delta_price_lag"] = matrix["delta_price_lag"].astype(np.float16)
matrix["delta_price_lag"].fillna(0, inplace=True)

# https://stackoverflow.com/questions/31828240/first-non-null-value-per-row-from-a-list-of-pandas-columns/31828559
# matrix['price_trend'] = matrix[['delta_price_lag_1','delta_price_lag_2','delta_price_lag_3']].bfill(axis=1).iloc[:, 0]
# Invalid dtype for backfill_2d [float16]

fetures_to_drop = ["item_avg_item_price", "date_item_avg_item_price"]
for i in lags:
    fetures_to_drop += ["date_item_avg_item_price_lag_" + str(i)]
    fetures_to_drop += ["delta_price_lag_" + str(i)]

matrix.drop(fetures_to_drop, axis=1, inplace=True)

print("time cost:", time.time() - ts)


# Last month shop revenue trend


ts = time.time()
group = train.groupby(["date_block_num", "shop_id"]).agg(
    {"revenue": ["sum"]}
)  # 每個月每個商店的銷售額
group.columns = ["date_shop_revenue"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "shop_id"], how="left")
matrix["date_shop_revenue"] = matrix["date_shop_revenue"].astype(np.float32)

group = group.groupby(["shop_id"]).agg({"date_shop_revenue": ["mean"]})  # 每個商店的平均銷售額
group.columns = ["shop_avg_revenue"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["shop_id"], how="left")
matrix["shop_avg_revenue"] = matrix["shop_avg_revenue"].astype(np.float32)

matrix["delta_revenue"] = (  # 商店的平均銷售額 與 每個月每個商店的銷售額 的比率
    matrix["date_shop_revenue"] - matrix["shop_avg_revenue"]
) / matrix["shop_avg_revenue"]
matrix["delta_revenue"] = matrix["delta_revenue"].astype(np.float16)

matrix = lag_feature(matrix, [1], "delta_revenue")  # 上個月 商店的平均銷售額 與 每個月每個商店的銷售額 的比率

matrix.drop(
    ["date_shop_revenue", "shop_avg_revenue", "delta_revenue"], axis=1, inplace=True
)
print("time cost:", time.time() - ts)


# ## Special features


matrix["month"] = matrix["date_block_num"] % 12


# Number of days in a month. There are no leap years.


days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
matrix["days"] = matrix["month"].map(days).astype(np.int8)


# Months since the last sale for each shop/item pair and for item only. I use programing approach.
#
# Create HashTable with key equals to {shop_id,item_id} and value equals to date_block_num.
# Iterate data from the top.
# Foreach row if {row.shop_id,row.item_id} is not present in the table,then add it to the table and set its value to row.date_block_num.
# if HashTable contains key, then calculate the difference beteween cached value and row.date_block_num.

# item_shop_last_sale: 距離這個商品在這個商店上次賣出過了多久


ts = time.time()
cache = {}
matrix["item_shop_last_sale"] = -1
matrix["item_shop_last_sale"] = matrix["item_shop_last_sale"].astype(np.int8)
for idx, row in matrix.iterrows():
    key = str(row.item_id) + " " + str(row.shop_id)
    if key not in cache:
        if row.item_cnt_month != 0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        matrix.at[idx, "item_shop_last_sale"] = row.date_block_num - last_date_block_num
        cache[key] = row.date_block_num
print("time cost:", time.time() - ts)


# item_last_sale: 距離這個商品上次賣出過了多久


ts = time.time()
cache = {}
matrix["item_last_sale"] = -1
matrix["item_last_sale"] = matrix["item_last_sale"].astype(np.int8)
for idx, row in matrix.iterrows():
    key = row.item_id
    if key not in cache:
        if row.item_cnt_month != 0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        if row.date_block_num > last_date_block_num:
            matrix.at[idx, "item_last_sale"] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num
print("time cost:", time.time() - ts)


# Months since the first sale for each shop/item pair and for item only.


ts = time.time()
matrix["item_shop_first_sale"] = matrix["date_block_num"] - matrix.groupby(
    ["item_id", "shop_id"]
)["date_block_num"].transform("min")
matrix["item_first_sale"] = matrix["date_block_num"] - matrix.groupby("item_id")[
    "date_block_num"
].transform("min")
print("time cost:", time.time() - ts)


# ## Final preparations
# Because of the using 12 as lag value drop first 12 months. Also drop all the columns with this month calculated values (other words which can not be calcucated for the test set).


ts = time.time()
matrix = matrix[matrix.date_block_num > 2]
print("time cost:", time.time() - ts)


# Producing lags brings a lot of nulls.


ts = time.time()


def fill_na(df):
    for col in df.columns:
        if ("_lag_" in col) & (df[col].isnull().any()):
            if "item_cnt" in col:
                df[col].fillna(0, inplace=True)
    return df


matrix = fill_na(matrix)
print("time cost:", time.time() - ts)

matrix.to_pickle("./data_3month.pkl")

print("--- {:6.2f} minutes ---".format((time.time() - start_time) / 60))
