# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A Data Provder that reads parallel (aligned) data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import data_provider
import random

class TrainValidSplitDataProvider(data_provider.DataProvider):
  """Creates a ParallelDataProvider. This data provider reads two datasets
  in parallel, keeping them aligned.

  Args:
    dataset1: The first dataset. An instance of the Dataset class.
    dataset2: The second dataset. An instance of the Dataset class.
      Can be None. If None, only `dataset1` is read.
    shuffle: Whether to shuffle the data sources and common queue when
      reading.
    num_epochs: The number of times each data source is read. If left as None,
      the data will be cycled through indefinitely.
    common_queue_capacity: The capacity of the common queue.
    common_queue_min: The minimum number of elements in the common queue after
      a dequeue.
    seed: The seed to use if shuffling.
  """

  def __init__(self,
               data,
               decoder1,
               decoder2,
               shuffle=True,
               num_epochs=None,
               common_queue_capacity=4096,
               common_queue_min=1024,
               seed=None,
               num_samples=None,
               batch_reader=None):

    if seed is None:
      seed = np.random.randint(10e8)

    # Optionally shuffle the data
    if shuffle:
      shuffle_queue = tf.RandomShuffleQueue(
          capacity=common_queue_capacity,
          min_after_dequeue=common_queue_min,
          dtypes=[tf.string, tf.string],
          seed=seed)
      enqueue_ops = []
      enqueue_ops.append(shuffle_queue.enqueue([data_batch, data_batch]))
      tf.train.add_queue_runner(
          tf.train.QueueRunner(shuffle_queue, enqueue_ops))
      data_source, data_target = shuffle_queue.dequeue()

    # Decode source items
    items = decoder1.list_items()
    tensors = decoder1.decode(data, items, batch_reader)

    # Decode target items
    items2 = decoder2.list_items()
    tensors2 = decoder2.decode(data, items2, batch_reader)

    # Merge items and results
    items = items + items2
    tensors = tensors + tensors2

    super(TrainValidSplitDataProvider, self).__init__(
      items_to_tensors=dict(zip(items, tensors)),
      num_samples=num_samples)

