# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load raw data and generate time series dataset."""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


DATA_DIR = 'gs://time_series_datasets'
LOCAL_CACHE_DIR = 'dataset/'


class TSFDataLoader:
  """Generate data loader from raw data."""

  def __init__(
    self, train_path, val_path, test_path, batch_size, seq_len, pred_len, target, date_col,
  ):
    self.batch_size = batch_size
    self.seq_len = seq_len
    self.pred_len = pred_len
    self.target = target
    self.date_col = date_col
    self.target_slice = slice(0, None)

    self._read_data(train_path, val_path, test_path)

  def _load_csv(self, path: str) -> pd.DataFrame:
    """Load CSV, removes column with date and returns DataFrame."""
    df = pd.read_csv(path)
    if self.date_col in df.columns:
      df = df.drop(columns=[self.date_col])
    df = df.select_dtypes(include=[np.number])
    return df

  def _read_data(self, train_path, val_path, test_path):
    """Load and scales data."""
    train_df = self._load_csv(train_path)
    val_df = self._load_csv(val_path)
    test_df = self._load_csv(test_path)

    if self.target in train_df.columns:
        target_idx = train_df.columns.get_loc(self.target)
        self.target_slice = slice(target_idx, target_idx + 1)
    else:
        raise ValueError(
            f"Cieľový stĺpec '{self.target}' nebol nájdený v dátach. "
            f"Dostupné stĺpce: {list(train_df.columns)}"
        )

    self.n_feature = train_df.shape[1]

    # standardize by training set
    self.scaler = StandardScaler()
    self.scaler.fit(train_df.values)

    def scale_df(df, scaler):
      data = scaler.transform(df.values)
      return pd.DataFrame(data, index=df.index, columns=df.columns)

    self.train_df = scale_df(train_df, self.scaler)
    self.val_df = scale_df(val_df, self.scaler)
    self.test_df = scale_df(test_df, self.scaler)
    # self.n_feature = self.train_df.shape[-1] //

  def _split_window(self, data):
    inputs = data[:, : self.seq_len, :]
    labels = data[:, self.seq_len :, self.target_slice]
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.seq_len, None])
    labels.set_shape([None, self.pred_len, None])
    return inputs, labels

  def _make_dataset(self, data, shuffle=True):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=(self.seq_len + self.pred_len),
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=self.batch_size,
    )
    ds = ds.map(self._split_window)
    return ds

  def inverse_transform(self, data):
    return self.scaler.inverse_transform(data)

  def get_train(self, shuffle=True):
    return self._make_dataset(self.train_df, shuffle=shuffle)

  def get_val(self):
    return self._make_dataset(self.val_df, shuffle=False)

  def get_test(self, shuffle = False):
    return self._make_dataset(self.test_df, shuffle=shuffle)
