#!/usr/bin/env python
# coding: utf-8

# 
# This notebook is simpified version of the final project in the [How to Win a Data Science Competition: Learn from Top Kagglers](https://www.coursera.org/learn/competitive-data-science) course. Simplified means without ensembling.
# 
# #### Pipline
# * load data
# * heal data and remove outliers
# * work with shops/items/cats objects and features
# * create matrix as product of item/shop pairs within each month in the train set
# * get monthly sales for each item/shop pair in the train set and merge it to the matrix
# * clip item_cnt_month by (0,20)
# * append test to the matrix, fill 34 month nans with zeros
# * merge shops/items/cats to the matrix
# * add target lag features
# * add mean encoded features
# * add price trend features
# * add month
# * add days
# * add months since last sale/months since first sale features
# * cut first year and drop columns which can not be calculated for the test set
# * select best features
# * set validation strategy 34 test, 33 validation, less than 33 train
# * fit the model, predict and clip targets for the test set

# # Part 1, perfect features

# In[1]:


import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)

from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

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

sys.version_info


# In[2]:


items = pd.read_csv("./data/items.csv")
shops = pd.read_csv("./data/shops.csv")
cats = pd.read_csv("./data/item_categories.csv")
train = pd.read_csv("./data/sales_train.csv")
# set index to ID to avoid droping it later
test = pd.read_csv("./data/test.csv").set_index("ID")


# ## Outliers

# There are items with strange prices and sales. After detailed exploration I decided to remove items with price > 100000 and sales > 1001 (1000 is ok).

# In[5]:


plt.figure(figsize=(10, 4))
plt.xlim(-100, 3000)
sns.boxplot(x=train.item_cnt_day)

plt.figure(figsize=(10, 4))
plt.xlim(train.item_price.min(), train.item_price.max() * 1.1)
sns.boxplot(x=train.item_price)
plt.show()


# In[6]:


train = train[train.item_price < 100000]
train = train[train.item_cnt_day < 1001]


# There is one item with price below zero. Fill it with median.

# In[7]:


train[train["item_price"] <= 0]


# In[8]:


median = train[
    (train.shop_id == 32)
    & (train.item_id == 2973)
    & (train.date_block_num == 4)
    & (train.item_price > 0)
].item_price.median()
train.loc[train.item_price < 0, "item_price"] = median


# Several shops are duplicates of each other (according to its name). Fix train and test set.

# In[9]:


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

# In[11]:


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

# In[15]:


len(test.shop_id.unique())


# In[16]:


len(set(test.shop_id).intersection(set(train.shop_id)))


# In[17]:


len(test.item_id.unique())


# In[18]:


len(set(test.item_id).intersection(set(train.item_id)))


# In[19]:


len(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id)))), len(
    list(set(test.item_id))
), len(test)


# In[21]:


train[train.date_block_num == 0].head()


# In[22]:


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
time.time() - ts


# In[23]:


matrix.head()


# Aggregate train set by shop/item pairs to calculate target aggreagates, then <b>clip(0,20)</b> target value. This way train target will be similar to the test predictions.
# 
# <i>I use floats instead of ints for item_cnt_month to avoid downcasting it after concatination with the test set later. If it would be int16, after concatination with NaN values it becomes int64, but foat16 becomes float16 even with NaNs.</i>

# In[24]:


train["revenue"] = train["item_price"] * train["item_cnt_day"]


# In[45]:


train[
    (train["date_block_num"] == 0) & (train["shop_id"] == 2) & (train["item_id"] == 19)
]


# In[25]:


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
time.time() - ts


# In[29]:


matrix.head()


# ## Test set
# To use time tricks append test pairs to the matrix.

# In[30]:


test["date_block_num"] = 34
test["date_block_num"] = test["date_block_num"].astype(np.int8)
test["shop_id"] = test["shop_id"].astype(np.int8)
test["item_id"] = test["item_id"].astype(np.int16)


# In[31]:


ts = time.time()
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True)  # 34 month
time.time() - ts


# In[33]:


matrix.tail()


# ## Shops/Items/Cats features

# In[34]:


ts = time.time()
matrix = pd.merge(matrix, shops, on=["shop_id"], how="left")
matrix = pd.merge(matrix, items, on=["item_id"], how="left")
matrix = pd.merge(matrix, cats, on=["item_category_id"], how="left")
matrix["city_code"] = matrix["city_code"].astype(np.int8)
matrix["item_category_id"] = matrix["item_category_id"].astype(np.int8)
matrix["type_code"] = matrix["type_code"].astype(np.int8)
matrix["subtype_code"] = matrix["subtype_code"].astype(np.int8)
time.time() - ts


# In[35]:


matrix.head()


# In[36]:


matrix.tail()


# ## Traget lags

# In[46]:


def lag_feature(df, lags, col):
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


# In[47]:


ts = time.time()
matrix = lag_feature(matrix, [1, 2, 3, 6, 12], "item_cnt_month")
time.time() - ts


# ## Mean encoded features

# In[48]:


ts = time.time()
group = matrix.groupby(["date_block_num"]).agg({"item_cnt_month": ["mean"]})
group.columns = ["date_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num"], how="left")
matrix["date_avg_item_cnt"] = matrix["date_avg_item_cnt"].astype(np.float16)
matrix = lag_feature(matrix, [1], "date_avg_item_cnt")
matrix.drop(["date_avg_item_cnt"], axis=1, inplace=True)
time.time() - ts


# In[49]:


matrix[(matrix["date_block_num"] == 0) & (matrix["item_id"] == 19)].head()


# In[50]:


ts = time.time()
group = matrix.groupby(["date_block_num", "item_id"]).agg({"item_cnt_month": ["mean"]})
group.columns = ["date_item_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "item_id"], how="left")
matrix["date_item_avg_item_cnt"] = matrix["date_item_avg_item_cnt"].astype(np.float16)
matrix = lag_feature(matrix, [1, 2, 3, 6, 12], "date_item_avg_item_cnt")
matrix.drop(["date_item_avg_item_cnt"], axis=1, inplace=True)
time.time() - ts


# In[51]:


ts = time.time()
group = matrix.groupby(["date_block_num", "shop_id"]).agg({"item_cnt_month": ["mean"]})
group.columns = ["date_shop_avg_item_cnt"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "shop_id"], how="left")
matrix["date_shop_avg_item_cnt"] = matrix["date_shop_avg_item_cnt"].astype(np.float16)
matrix = lag_feature(matrix, [1, 2, 3, 6, 12], "date_shop_avg_item_cnt")
matrix.drop(["date_shop_avg_item_cnt"], axis=1, inplace=True)
time.time() - ts


# In[52]:


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
time.time() - ts


# In[53]:


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
time.time() - ts


# In[54]:


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
time.time() - ts


# In[55]:


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
time.time() - ts


# In[56]:


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
time.time() - ts


# In[57]:


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
time.time() - ts


# In[58]:


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
time.time() - ts


# In[59]:


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
time.time() - ts


# ## Trend features

# Price trend for the last six months.

# In[60]:


ts = time.time()
group = train.groupby(["item_id"]).agg({"item_price": ["mean"]})
group.columns = ["item_avg_item_price"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["item_id"], how="left")
matrix["item_avg_item_price"] = matrix["item_avg_item_price"].astype(np.float16)

group = train.groupby(["date_block_num", "item_id"]).agg({"item_price": ["mean"]})
group.columns = ["date_item_avg_item_price"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "item_id"], how="left")
matrix["date_item_avg_item_price"] = matrix["date_item_avg_item_price"].astype(
    np.float16
)

lags = [1, 2, 3, 4, 5, 6]
matrix = lag_feature(matrix, lags, "date_item_avg_item_price")

for i in lags:
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

time.time() - ts


# Last month shop revenue trend

# In[61]:


ts = time.time()
group = train.groupby(["date_block_num", "shop_id"]).agg({"revenue": ["sum"]})
group.columns = ["date_shop_revenue"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["date_block_num", "shop_id"], how="left")
matrix["date_shop_revenue"] = matrix["date_shop_revenue"].astype(np.float32)

group = group.groupby(["shop_id"]).agg({"date_shop_revenue": ["mean"]})
group.columns = ["shop_avg_revenue"]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=["shop_id"], how="left")
matrix["shop_avg_revenue"] = matrix["shop_avg_revenue"].astype(np.float32)

matrix["delta_revenue"] = (
    matrix["date_shop_revenue"] - matrix["shop_avg_revenue"]
) / matrix["shop_avg_revenue"]
matrix["delta_revenue"] = matrix["delta_revenue"].astype(np.float16)

matrix = lag_feature(matrix, [1], "delta_revenue")

matrix.drop(
    ["date_shop_revenue", "shop_avg_revenue", "delta_revenue"], axis=1, inplace=True
)
time.time() - ts


# ## Special features

# In[62]:


matrix["month"] = matrix["date_block_num"] % 12


# Number of days in a month. There are no leap years.

# In[63]:


days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
matrix["days"] = matrix["month"].map(days).astype(np.int8)


# Months since the last sale for each shop/item pair and for item only. I use programing approach.
# 
# <i>Create HashTable with key equals to {shop_id,item_id} and value equals to date_block_num. Iterate data from the top. Foreach row if {row.shop_id,row.item_id} is not present in the table, then add it to the table and set its value to row.date_block_num. if HashTable contains key, then calculate the difference beteween cached value and row.date_block_num.</i>

# In[64]:


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
time.time() - ts


# In[65]:


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
time.time() - ts


# Months since the first sale for each shop/item pair and for item only.

# In[66]:


ts = time.time()
matrix["item_shop_first_sale"] = matrix["date_block_num"] - matrix.groupby(
    ["item_id", "shop_id"]
)["date_block_num"].transform("min")
matrix["item_first_sale"] = matrix["date_block_num"] - matrix.groupby("item_id")[
    "date_block_num"
].transform("min")
time.time() - ts


# ## Final preparations
# Because of the using 12 as lag value drop first 12 months. Also drop all the columns with this month calculated values (other words which can not be calcucated for the test set).

# In[67]:


ts = time.time()
matrix = matrix[matrix.date_block_num > 11]
time.time() - ts


# Producing lags brings a lot of nulls.

# In[68]:


ts = time.time()


def fill_na(df):
    for col in df.columns:
        if ("_lag_" in col) & (df[col].isnull().any()):
            if "item_cnt" in col:
                df[col].fillna(0, inplace=True)
    return df


matrix = fill_na(matrix)
time.time() - ts


# In[69]:


matrix.columns


# In[70]:


matrix.info()


# In[71]:


matrix.to_pickle("data.pkl")
del matrix
del cache
del group
del items
del shops
del cats
del train
# leave test for submission
gc.collect()

