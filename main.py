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

sys.version_info

# items = pd.read_csv("../input/items.csv")
# shops = pd.read_csv("../input/shops.csv")
# cats = pd.read_csv("../input/item_categories.csv")
# train = pd.read_csv("../input/sales_train.csv")
# # set index to ID to avoid droping it later
test = pd.read_csv("../input/test.csv").set_index("ID")


# # Part 2, xgboost


# data = pd.read_pickle("../data.pkl")
data = pd.read_pickle("./data_3month.pkl")

# Select perfect features


# data = data[
#     [
#         "date_block_num",
#         "shop_id",
#         "item_id",
#         "item_cnt_month",
#         "city_code",
#         "item_category_id",
#         "type_code",
#         "subtype_code",
#         "item_cnt_month_lag_1",
#         "item_cnt_month_lag_2",
#         "item_cnt_month_lag_3",
#         "item_cnt_month_lag_6",
#         "item_cnt_month_lag_12",
#         "date_avg_item_cnt_lag_1",
#         "date_item_avg_item_cnt_lag_1",
#         "date_item_avg_item_cnt_lag_2",
#         "date_item_avg_item_cnt_lag_3",
#         "date_item_avg_item_cnt_lag_6",
#         "date_item_avg_item_cnt_lag_12",
#         "date_shop_avg_item_cnt_lag_1",
#         "date_shop_avg_item_cnt_lag_2",
#         "date_shop_avg_item_cnt_lag_3",
#         "date_shop_avg_item_cnt_lag_6",
#         "date_shop_avg_item_cnt_lag_12",
#         "date_cat_avg_item_cnt_lag_1",
#         "date_shop_cat_avg_item_cnt_lag_1",
#         #'date_shop_type_avg_item_cnt_lag_1',
#         #'date_shop_subtype_avg_item_cnt_lag_1',
#         "date_city_avg_item_cnt_lag_1",
#         "date_item_city_avg_item_cnt_lag_1",
#         #'date_type_avg_item_cnt_lag_1',
#         #'date_subtype_avg_item_cnt_lag_1',
#         "delta_price_lag",
#         "month",
#         "days",
#         "item_shop_last_sale",
#         "item_last_sale",
#         "item_shop_first_sale",
#         "item_first_sale",
#     ]
# ]


# ## remove shops that do not appear in test


print("test shops:", len(data[data.date_block_num == 34].shop_id.unique()))
a = data[data.date_block_num == 34].shop_id.unique()

print("data shops:", len(data[data.date_block_num != 34].shop_id.unique()))
b = data[data.date_block_num != 34].shop_id.unique()

bad_shop = []
for i in b:
    if i not in a:
        bad_shop.append(i)
bad_shop.sort()

data = data[
    (data.shop_id != 9)
    & (data.shop_id != 13)
    & (data.shop_id != 17)
    & (data.shop_id != 20)
    & (data.shop_id != 27)
    & (data.shop_id != 29)
    & (data.shop_id != 30)
    & (data.shop_id != 33)
    & (data.shop_id != 40)
    & (data.shop_id != 43)
    & (data.shop_id != 51)
    & (data.shop_id != 54)
]

# Validation strategy is 34 month for the test set, 33 month for the validation set and 13-33 months for the train.

X_train = data[data.date_block_num < 33].drop(["item_cnt_month"], axis=1)
Y_train = data[data.date_block_num < 33]["item_cnt_month"]
X_valid = data[data.date_block_num == 33].drop(["item_cnt_month"], axis=1)
Y_valid = data[data.date_block_num == 33]["item_cnt_month"]
X_test = data[data.date_block_num == 34].drop(["item_cnt_month"], axis=1)

del data
gc.collect()


# ## Blending

split_num = int(len(X_train) * 0.9)
print("training data:", split_num)
print("hold out:", len(X_train) - split_num)


# shuffle the train data


X_train_s = X_train.sample(frac=1, random_state=0)

X_train_l1 = X_train_s[:split_num]
Y_train_l1 = Y_train.loc[X_train_l1.index]

X_train_l2 = X_train_s[split_num:]
Y_train_l2 = Y_train.loc[X_train_l2.index]

# # leavel 1

# xgboost
ts = time.time()

model_1 = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.05,
    seed=0,
    n_jobs=-1,
)

model_1.fit(
    X_train_l1,
    Y_train_l1,
    eval_metric="rmse",
    eval_set=[(X_train_l1, Y_train_l1), (X_valid, Y_valid)],
    verbose=True,
    early_stopping_rounds=20,
)

print("time cost:", time.time() - ts)

# lightgbm

import lightgbm as lgb

feature_name = X_train_l1.columns.tolist()

params = {
    "objective": "mse",
    "metric": "rmse",
    "num_leaves": 2 ** 8 - 1,
    "learning_rate": 0.005,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.75,
    "bagging_freq": 5,
    "seed": 1,
    "verbose": 1,
    "n_jobs": -1,  # <--modify
}
feature_name_indexes = [
    "shop_id",
    "item_id",
    "city_code",
    "item_category_id",
    "type_code",
    "subtype_code",
]

lgb_train = lgb.Dataset(X_train_l1[feature_name], Y_train_l1)
lgb_eval = lgb.Dataset(X_valid[feature_name], Y_valid, reference=lgb_train)

evals_result = {}
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=3000,
    valid_sets=(lgb_train, lgb_eval),
    feature_name=feature_name,
    categorical_feature=feature_name_indexes,
    verbose_eval=50,
    evals_result=evals_result,
    early_stopping_rounds=30,
)


# catboost


from catboost import CatBoostRegressor

catboost_model = CatBoostRegressor(
    iterations=1000,
    max_ctr_complexity=10,
    random_seed=0,
    od_type="Iter",
    od_wait=30,
    verbose=50,
    depth=8,
    thread_count=-1,
)

feature_name_indexes = [1, 2, 3, 4, 5, 6]

catboost_model.fit(
    X_train_l1,
    Y_train_l1.astype(float),
    cat_features=feature_name_indexes,
    eval_set=(X_valid, Y_valid.astype(float)),
)


Y_pred_1 = model_1.predict(X_train_l2)  # .clip(0, 20)
Y_pred_2 = gbm.predict(X_train_l2)  # .clip(0, 20)
Y_pred_3 = catboost_model.predict(X_train_l2)  # .clip(0, 20)

Y_valid_1 = model_1.predict(X_valid)  # .clip(0, 20)
Y_valid_2 = gbm.predict(X_valid)  # .clip(0, 20)
Y_valid_3 = catboost_model.predict(X_valid)  # .clip(0, 20)

Y_test_1 = model_1.predict(X_test)  # .clip(0, 20)
Y_test_2 = gbm.predict(X_test)  # .clip(0, 20)
Y_test_3 = catboost_model.predict(X_test)  # .clip(0, 20)

# # leavel 2

# ## data


# X_train_l2_fea = np.stack((Y_pred_1, Y_pred_2, Y_pred_3), axis=-1)
# X_val_l2_fea = np.stack((Y_valid_1, Y_valid_2, Y_valid_3), axis=-1)
# X_test_l2_fea = np.stack((Y_test_1, Y_test_2, Y_test_3), axis=-1)

X_train_l2_fea = np.stack((Y_pred_1, Y_pred_2), axis=-1)
X_val_l2_fea = np.stack((Y_valid_1, Y_valid_2), axis=-1)
X_test_l2_fea = np.stack((Y_test_1, Y_test_2), axis=-1)

print(X_train_l2_fea.shape)
print(X_val_l2_fea.shape)
print(X_test_l2_fea.shape)

# # xgb

import xgboost as xgb

ts = time.time()


# def learning_rate_decay(boosting_round, num_boost_round):
#     learning_rate_start = 0.1
#     learning_rate_min = 0.0009
#     lr_decay = 0.96
#     lr = learning_rate_start * np.power(lr_decay, boosting_round)
#     return max(learning_rate_min, lr)


model_l2 = XGBRegressor(
    max_depth=10,
    n_estimators=1200,
    min_child_weight=0.3,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.002,
    seed=47,
    n_jobs=24,
)

model_l2.fit(
    X_train_l2_fea,
    Y_train_l2,
    eval_metric="rmse",
    eval_set=[(X_train_l2_fea, Y_train_l2), (X_val_l2_fea, Y_valid)],
    verbose=50,
    early_stopping_rounds=30,
    # callbacks=[xgb.callback.reset_learning_rate(learning_rate_decay)],
)

print("time cost:", time.time() - ts)


plot_features(model_l2, (10, 14))


Y_test_l2_fea = model_l2.predict(X_test_l2_fea).clip(0, 20)
submission = pd.DataFrame({"ID": test.index, "item_cnt_month": Y_test_l2_fea})
submission.to_csv("./submission.csv", index=False)
submission.head(20)
