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
"""A decoder that splits a string into tokens and returns the
individual tokens and the length.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import data_decoder

import random


class SplitGraphemesPhonemesDecoder(data_decoder.DataDecoder):
  """A DataProvider that splits a string tensor into individual tokens and
  returns the tokens and the length.
  Optionally prepends or appends special tokens.

  Args:
    delimiter: Delimiter to split on. Must be a single character.
    tokens_feature_name: A descriptive feature name for the token values
    length_feature_name: A descriptive feature name for the length value
  """

  def __init__(self,
               delimiter=" ",
               feature_name="source_tokens",
               length_feature_name="source_len",
               prepend_token=None,
               append_token=None):
    self.delimiter = delimiter
    self.feature_name = feature_name
    self.length_feature_name = length_feature_name
    self.prepend_token = prepend_token
    self.append_token = append_token

  def decode(self, data, items):
    decoded_items = {}

    data_line = random.choice(data)

    # Split lines with source and target sequences
    source_target_lines = tf.string_split([data_line], delimiter=self.delimiter)

    if self.length_feature_name == "source_tokens":
      results = tf.slice(source_target_lines.values, [0], [1])
      # Split graphemes
      results = tf.string_split(results, delimiter='')
    else:
      results = tf.slice(source_target_lines.values, [1], [-1])

    # Optionally prepend a special token
    if self.prepend_token is not None:
      results = tf.concat([[self.prepend_token], results], 0)

    # Optionally append a special token
    if self.append_token is not None:
      results = tf.concat([results, [self.append_token]], 0)

    decoded_items[self.length_feature_name] = tf.size(results)
    decoded_items[self.feature_name] = results
    return [decoded_items[_] for _ in items]

  def list_items(self):
    return [self.feature_name, self.length_feature_name]

