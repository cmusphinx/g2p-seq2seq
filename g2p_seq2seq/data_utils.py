# Copyright 2016 AC Technology LLC. All Rights Reserved.
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

import gzip
import os
import re
import tarfile
import codecs

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def create_vocabulary(vocabulary_path, data):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one word per line. Each word is
  tokenized and digits.
  Vocabulary contains the most-frequent tokens.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data: data file that will be used to create vocabulary.

  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s" % (vocabulary_path))
    vocab = {}
    for i, line in enumerate(data):
      if i > 0 and i % 100000 == 0:
        print("  processing line %d" % i)
      for item in line:
        if item in vocab:
          vocab[item] += 1
        else:
          vocab[item] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    with codecs.open(vocabulary_path, "w", "utf-8") as vocab_file:
      for w in vocab_list:
        vocab_file.write( w + '\n')


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    d
    c
  will result in a vocabulary {"d": 0, "c": 1}, and this function will
  also return the reversed-vocabulary ["d", "c"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with codecs.open(vocabulary_path, "r", "utf-8") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def data_to_token_ids(data, vocab):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data: input data in one-word-per-line format.
    vocabulary: vocabulary.
  """
  tokens_dic =[]
  for i, line in enumerate(data):
    token_ids = [vocab.get(s, UNK_ID) for s in line]
    tokens_dic.append(token_ids)
  return tokens_dic


def split_to_grapheme_phoneme(inp_dictionary):
  """Split input dictionary into two separate lists with graphemes and phonemes.

  Args:
    inp_dictionary: input dictionary.
  """

  graphemes, phonemes = [], []
  for l in inp_dictionary:
    line = l.strip().split()
    if len(line)>1:
      graphemes.append(list(line[0]))
      phonemes.append(line[1:])
  return graphemes, phonemes


def prepare_g2p_data(model_dir, train_gr, train_ph, valid_gr, valid_ph):
  """Create vocabularies into model_dir, create ids data lists.

  Args:
    model_dir: directory in which the data sets will be stored.

  Returns:
    A tuple of 6 elements:
      (1) Token-ids for Grapheme training data-set,
      (2) Token-ids for Phoneme training data-set,
      (3) Token-ids for Grapheme development data-set,
      (4) Token-ids for Phoneme development data-set,
      (5) Grapheme vocabulary file,
      (6) Phoneme vocabulary file.
  """
  # Create vocabularies of the appropriate sizes.
  ph_vocab_path = os.path.join(model_dir, "vocab.phoneme")
  gr_vocab_path = os.path.join(model_dir, "vocab.grapheme")
  print("Creating vocabularies in %s" %model_dir)
  create_vocabulary(ph_vocab_path, train_ph)
  create_vocabulary(gr_vocab_path, train_gr)

  # Initialize vocabularies.
  ph_vocab, _ = initialize_vocabulary(ph_vocab_path)
  gr_vocab, _ = initialize_vocabulary(gr_vocab_path)

  # Create ids for the training data.
  train_ph_ids = data_to_token_ids(train_ph, ph_vocab)
  train_gr_ids = data_to_token_ids(train_gr, gr_vocab)

  # Create ids for the development data.
  valid_ph_ids = data_to_token_ids(valid_ph, ph_vocab)
  valid_gr_ids = data_to_token_ids(valid_gr, gr_vocab)

  return (train_gr_ids, train_ph_ids,
          valid_gr_ids, valid_ph_ids,
          gr_vocab, ph_vocab)
