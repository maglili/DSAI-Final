# DSAI - HW4

## Report

PowerPoint: [link](https://1drv.ms/p/s!AlsGDFPDXfPAhjXEEaLUd2lhqQA7)

## Intro

本題目為 [Kaggle - Predict Future Sales](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)。
執行前請先將比賽的 data 放到 input/ 資料夾中。

## Usage

### Step 1: Data preprocessing

```bash
python prep.py
```

資料前處理，產生特徵。執行完後，會在工作目錄下生成 data.pkl

### Step 2: Stacking XGBoost

```bash
python main.py
```

Level 1 model: XGBoost, LightGBM

Level 2 model: XGBoost
