import numpy as np
import pandas as pd
from itertools import product
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
import time
import sys
import gc
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

from utils import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
plt.rcParams["figure.figsize"] = (12, 6)

# load data
test = pd.read_csv("./data/test.csv").set_index("ID")
data = pd.read_pickle("./data.pkl").drop(
    columns=[
        "date_shop_type_avg_item_cnt_lag_1",
        "date_shop_subtype_avg_item_cnt_lag_1",
        "date_type_avg_item_cnt_lag_1",
        "date_subtype_avg_item_cnt_lag_1",
    ]
)

# train / val / test data
X_train = data[data.date_block_num < 33].drop(["item_cnt_month"], axis=1)
Y_train = data[data.date_block_num < 33]["item_cnt_month"]
X_valid = data[data.date_block_num == 33].drop(["item_cnt_month"], axis=1)
Y_valid = data[data.date_block_num == 33]["item_cnt_month"]
X_test = data[data.date_block_num == 34].drop(["item_cnt_month"], axis=1)
del data
gc.collect()

# ensemble model data
split_num = int(len(X_train) * 0.8)
X_train_s = X_train.sample(frac=1, random_state=0)
X_train_l1 = X_train_s[:split_num]
Y_train_l1 = Y_train.loc[X_train_l1.index]
X_train_l2 = X_train_s[split_num:]
Y_train_l2 = Y_train.loc[X_train_l2.index]
print("level 1 training data:", split_num)
print("level 2 training data:", len(X_train) - split_num)
print()

# l1m1
ts = time.time()
model_1 = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42,
)
model_1.fit(
    X_train_l1,
    Y_train_l1,
    eval_metric="rmse",
    eval_set=[(X_train_l1, Y_train_l1), (X_valid, Y_valid)],
    verbose=True,
    early_stopping_rounds=10,
)
print("\nl1m1 finished:", time.time() - ts)
print()

# l1m2
ts = time.time()
model_2 = XGBRegressor(
    max_depth=6,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=43,
)
model_2.fit(
    X_train_l1,
    Y_train_l1,
    eval_metric="rmse",
    eval_set=[(X_train_l1, Y_train_l1), (X_valid, Y_valid)],
    verbose=True,
    early_stopping_rounds=10,
)
print("\nl1m2 finished:", time.time() - ts)
print()

# l1m3
ts = time.time()
model_3 = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=44,
)
model_3.fit(
    X_train_l1,
    Y_train_l1,
    eval_metric="rmse",
    eval_set=[(X_train_l1, Y_train_l1), (X_valid, Y_valid)],
    verbose=True,
    early_stopping_rounds=10,
)
print("\nl1m3 finished:", time.time() - ts)
print()

# l1m4
ts = time.time()
model_4 = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=45,
)
model_4.fit(
    X_train_l1,
    Y_train_l1,
    eval_metric="rmse",
    eval_set=[(X_train_l1, Y_train_l1), (X_valid, Y_valid)],
    verbose=True,
    early_stopping_rounds=10,
)
print("\nl1m4 finished:", time.time() - ts)
print()

# l1m5
ts = time.time()
model_5 = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=46,
)
model_5.fit(
    X_train_l1,
    Y_train_l1,
    eval_metric="rmse",
    eval_set=[(X_train_l1, Y_train_l1), (X_valid, Y_valid)],
    verbose=True,
    early_stopping_rounds=10,
)
print("\nl1m5 finished:", time.time() - ts)
print()

# level 2 feature
Y_pred_1 = model_1.predict(X_train_l2)  # .clip(0, 20)
Y_pred_2 = model_2.predict(X_train_l2)  # .clip(0, 20)
Y_pred_3 = model_3.predict(X_train_l2)  # .clip(0, 20)
Y_pred_4 = model_4.predict(X_train_l2)  # .clip(0, 20)

Y_pred_5 = model_5.predict(X_train_l2)  # .clip(0, 20)
Y_valid_1 = model_1.predict(X_valid)  # .clip(0, 20)
Y_valid_2 = model_2.predict(X_valid)  # .clip(0, 20)
Y_valid_3 = model_3.predict(X_valid)  # .clip(0, 20)
Y_valid_4 = model_4.predict(X_valid)  # .clip(0, 20)
Y_valid_5 = model_5.predict(X_valid)  # .clip(0, 20)

Y_test_1 = model_1.predict(X_test)  # .clip(0, 20)
Y_test_2 = model_2.predict(X_test)  # .clip(0, 20)
Y_test_3 = model_3.predict(X_test)  # .clip(0, 20)
Y_test_4 = model_4.predict(X_test)  # .clip(0, 20)
Y_test_5 = model_5.predict(X_test)  # .clip(0, 20)

X_train_l2_fea = np.stack((Y_pred_1, Y_pred_2, Y_pred_3, Y_pred_4, Y_pred_5), axis=-1)
X_val_l2_fea = np.stack(
    (Y_valid_1, Y_valid_2, Y_valid_3, Y_valid_4, Y_valid_5), axis=-1
)
X_test_l2_fea = np.stack((Y_test_1, Y_test_2, Y_test_3, Y_test_4, Y_test_5), axis=-1)

# torch device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU is avalible")
    print("Working on ", torch.cuda.get_device_name())
else:
    device = torch.device("cpu")
    print("GPU is not avalible")

# dataset
trainset = Data(X_train_l2_fea, Y_train_l2.values)
validset = Data(X_val_l2_fea, Y_valid.values)

# model
model = NN(in_features=5, out_features=1)
model.to(device)
print(model)

# hypperparameter
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.AdamW(model.parameters(), lr=0.005)
bs = 4096
train_loader = DataLoader(dataset=trainset, batch_size=bs, shuffle=True)
val_loader = DataLoader(dataset=validset, batch_size=bs, shuffle=False)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="min", factor=0.5, patience=2
)
num_epochs = 10

# training
history = training(
    model, num_epochs, train_loader, val_loader, scheduler, optimiser, loss_fn, device
)

# learning curve
plt.plot(history[0], label="Train")
plt.plot(history[1], label="valid")
plt.grid()
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("loss v epochs")
plt.show()
plt.savefig("./learning_curve.png", bbox_inches="tight")

# predict
testset = TestData(X_test_l2_fea)
test_loader = DataLoader(dataset=testset, batch_size=bs, shuffle=False)

model.eval()
pred = []
with torch.no_grad():
    for x in test_loader:
        x = x.to(device)
        ypred = model(x)
        pred.extend(ypred.tolist())
pred = np.array(pred).clip(0, 20)

submission = pd.DataFrame({"ID": test.index, "item_cnt_month": pred})
submission.to_csv("./submission.csv", index=False)
