# Copyright 2016 AC Technologies LLC. All Rights Reserved.
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
# ==============================================================================

"""Utilities for tokenizing, creation vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
from tensorflow.python.platform import gfile


def create_vocabulary(data_path, save_dir):
  """Create vocabulary from input data.
  Input data is assumed to contain one word per line.

  Args:
    data: word list that will be used to create vocabulary.

  Rerurn:
    vocab: vocabulary dictionary. In this dictionary keys are symbols
           and values are their indexes.
  """
  data_lines = codecs.open(data_path, "r", "utf-8").readlines()
  vocab_gr, vocab_ph = {}, {}
  for line in data_lines:
    line_split = line.replace('\t', ' ').replace('\n', '')
    line_split = line_split.split(' ')
    graphemes, phonemes = line_split[0], line_split[1:]
    for item in graphemes:
      if item not in vocab_gr:
        vocab_gr[item] = 1
    for item in phonemes:
      if item not in vocab_ph:
        vocab_ph[item] = 1

  #vocab = dict([(x, y) for (y, x) in enumerate(vocab_list)])
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)
  vocab_gr_path, vocab_ph_path = os.path.join(save_dir, "vocab.grapheme"),\
    os.path.join(save_dir, "vocab.phoneme")
  save_vocabulary(vocab_gr, vocab_gr_path)
  save_vocabulary(vocab_ph, vocab_ph_path)



def save_vocabulary(vocab, vocabulary_path):
  """Save vocabulary file in vocabulary_path.
  We write vocabulary to vocabulary_path in a one-token-per-line format,
  so that later token in the first line gets id=0, second line gets id=1,
  and so on.

  Args:
    vocab: vocabulary dictionary.
    vocabulary_path: path where the vocabulary will be created.

  """
  print("Creating vocabulary %s" % (vocabulary_path))
  with codecs.open(vocabulary_path, "w", "utf-8") as vocab_file:
    for symbol in sorted(vocab):
      vocab_file.write(symbol + '\n')


def collect_pronunciations(dic_lines):
  '''Create dictionary mapping word to its different pronounciations.
  '''
  dic = {}
  for line in dic_lines:
    lst = line.strip().split()
    if len(lst) > 1:
      if lst[0] not in dic:
        dic[lst[0]] = [" ".join(lst[1:])]
      else:
        dic[lst[0]].append(" ".join(lst[1:]))
    elif len(lst) == 1:
      print("WARNING: No phonemes for word '%s' line ignored" % (lst[0]))
  return dic


def unify(dic_lines):
  dic = []
  for line in dic_lines:
    lst = line.strip().split()
    dic.append(" ".join(lst))
  return dic


def split_dictionaries(train_path=None, valid_path=None, test_path=None):
  """Split source dictionary to train, validation and test sets.
  """
  if train_path:
    source_dic = codecs.open(train_path, "r", "utf-8").readlines()
    train_dic, valid_dic, test_dic = [], [], []
    if valid_path:
      valid_dic = codecs.open(valid_path, "r", "utf-8").readlines()
      valid_dic = unify(valid_dic)
    if test_path:
      test_dic = codecs.open(test_path, "r", "utf-8").readlines()
      test_dic = unify(test_dic)

    dic = collect_pronunciations(source_dic)

    # Split dictionary to train, validation and test (if not assigned).
    for i, word in enumerate(dic):
      for pronunciations in dic[word]:
        if i % 20 == 0 and not valid_path:
          valid_dic.append(word + ' ' + pronunciations)
        elif (i % 20 == 1 or i % 20 == 2) and not test_path:
          test_dic.append(word + ' ' + pronunciations)
        else:
          train_dic.append(word + ' ' + pronunciations)
    return train_dic, valid_dic, test_dic
  else:
    if not test_path:
      raise RuntimeError("Path to the decode file not specified.")
    test_dic = codecs.open(test_path, "r", "utf-8").readlines()
    test_dic = unify(test_dic)
    return None, None, test_dic
