from ast import Str
import copy
from pathlib import Path
from typing import List
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

import yfinance as yf


def get_crypto_df(crypto_names: list, price_period: str="6mo", price_interval: str="1d"):
    for i, name in enumerate(crypto_names):
        ticker=f"{name}-USD"

        dataframe_tmp = yf.download(ticker, period=price_period, interval=price_interval)
        dataframe_tmp = dataframe_tmp[['Open','High','Low','Close','Volume']]
        dataframe_tmp["time_idx"] = [i for i, _ in enumerate(dataframe_tmp.index)]
        dataframe_tmp["month"] = [str(i.month) for i in dataframe_tmp.index]
        dataframe_tmp["crypto"] = name

        if i==0:
            dataframe = dataframe_tmp
        else:
            dataframe = pd.concat([dataframe, dataframe_tmp], ignore_index=True)
    return dataframe

dataframe = get_crypto_df(['BTC', 'ETH', 'ADA', 'AVAX'], "6mo", "1d")
#print(dataframe)

max_prediction_length = 6
max_encoder_length = 60

#training_cutoff = int(dataframe["time_idx"].max() * 0.8)
#train_df = dataframe[lambda x: x.time_idx <= training_cutoff]
#valid_df = dataframe[lambda x: x.time_idx > training_cutoff]
#print(train_df)
#print(valid_df)

training_cutoff = dataframe["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    dataframe[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="Close",
    group_ids=["crypto"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["crypto"],
    static_reals=[],
    time_varying_known_categoricals=["month"],
    time_varying_known_reals=[],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["Open","High","Low","Close","Volume"],
    #add_relative_time_idx=True,
    #add_target_scales=True,
    #add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, dataframe, predict=True, stop_randomization=True)

print(training.get_parameters())
print(validation.get_parameters())

batch_size = 2  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
valid_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)



""" # and load the first batch
x, y = next(iter(train_dataloader))
print("x =", x)
print("\ny =", y)
print("\nsizes of x =")
for key, value in x.items():
    print(f"\t{key} = {value.size()}") """