# Copyright 2015 Google Inc. All Rights Reserved.
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

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
#_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data, tokenizer=None):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one word per line. Each word is
  tokenized and digits.
  Vocabulary contains the most-frequent tokens.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    tokenizer: a function to use to tokenize each data word;
      if None, basic_tokenizer will be used.

  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s" % (vocabulary_path))
    vocab = {}
    for i, line in enumerate(data):
      if i % 100000 == 0:
        print("  processing line %d" % i)
      tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
      for item in tokens:
        if item in vocab:
          vocab[item] += 1
        else:
          vocab[item] = 1
    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
      for w in vocab_list:
        vocab_file.write(w + "\n")



def get_vocab_size(vocabulary_path):
  """Return size of the vocabulary.

  Args:
    vocabulary_path: path to the vocabulary file.
  """
  counter = 0
  if gfile.Exists(vocabulary_path):
    with gfile.GFile(vocabulary_path, mode="r") as f:
      for line in f:
        counter += 1
  return counter


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
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "M A N Y" may become tokenized into
  ["M", "A", "N", "Y"] and with vocabulary {"M": 1, "A": 2,
  "N": 4, "Y": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data, vocabulary_path,
                      tokenizer=None):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
  """
  vocab, _ = initialize_vocabulary(vocabulary_path)
  tokens_dic =[]
  for i, line in enumerate(data):
    if i > 0 and i % 100000 == 0:
      print("  tokenizing line %d" % i)
    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
    tokens_dic.append(token_ids)
  return tokens_dic


def split_to_grapheme_phoneme(inp_dictionary):
  """Split input dictionary into two separate files (which ended with .grapheme and .phoneme) in path directory.

  Args:
    inp_dictionary: input dictionary.
    path: path where we will create two files with separate graphemes and phonemes.
  """
  # Create vocabularies of the appropriate sizes.

  lst = []
  for line in inp_dictionary:
    lst.append(line.split())

  graphemes, phonemes = [], []
  for line in lst:
    if len(line)>1:
      graphemes.append(' '.join(list(line[0])) + '\n')
      phonemes.append(' '.join(line[1:]) + '\n')

  return graphemes, phonemes



def prepare_g2p_data(model_dir, train_gr, train_ph, valid_gr, valid_ph):
  """Get G2P data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.

  Returns:
    A tuple of 8 elements:
      (1) path to the token-ids for Grapheme training data-set,
      (2) path to the token-ids for Phoneme training data-set,
      (3) path to the token-ids for Grapheme development data-set,
      (4) path to the token-ids for Phoneme development data-set,
      (5) path to the Grapheme vocabulary file,
      (6) path to the Phoneme vocabulary file,
      (7) Grapheme vocabulary size,
      (8) Phoneme vocabulary size.
  """
  # Create vocabularies of the appropriate sizes.
  ph_vocab_path = os.path.join(model_dir, "vocab.phoneme")
  gr_vocab_path = os.path.join(model_dir, "vocab.grapheme")
  create_vocabulary(ph_vocab_path, train_ph)
  create_vocabulary(gr_vocab_path, train_gr)
  gr_vocab_size = get_vocab_size(gr_vocab_path)
  ph_vocab_size = get_vocab_size(ph_vocab_path)

  # Create token ids for the training data.
  print("Tokenizing data train phonemes")
  train_ph_ids = data_to_token_ids(train_ph, ph_vocab_path)
  print("Tokenizing data train graphemes")
  train_gr_ids = data_to_token_ids(train_gr, gr_vocab_path)

  # Create token ids for the development data.
  print("Tokenizing data valid phonemes")
  valid_ph_ids = data_to_token_ids(valid_ph, ph_vocab_path)
  print("Tokenizing data valid graphemes")
  valid_gr_ids = data_to_token_ids(valid_gr, gr_vocab_path)

  return (train_gr_ids, train_ph_ids,
          valid_gr_ids, valid_ph_ids,
          gr_vocab_path, ph_vocab_path,
          gr_vocab_size, ph_vocab_size)
