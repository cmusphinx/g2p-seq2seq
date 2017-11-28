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

"""Binary for training translation models and decoding from them.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import six
from tensor2tensor.data_generators import text_encoder

from IPython.core.debugger import Tracer

class GraphemePhonemeEncoder(text_encoder.TextEncoder):
  """Encodes each grapheme or phoneme to an id. For 8-bit strings only."""

  def __init__(self,
               vocab_filepath=None,
               vocab_list=None,
               separator="",
               num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS):
    """Initialize from a file or list, one token per line.

    Handling of reserved tokens works as follows:
    - When initializing from a list, we add reserved tokens to the vocab.
    - When initializing from a file, we do not add reserved tokens to the vocab.
    - When saving vocab files, we save reserved tokens to the file.

    Args:
      vocab_filename: If not None, the full filename to read vocab from. If this
         is not None, then vocab_list should be None.
      vocab_list: If not None, a list of elements of the vocabulary. If this is
         not None, then vocab_filename should be None.
      num_reserved_ids: Number of IDs to save for reserved tokens like <EOS>.
    """
    super(GraphemePhonemeEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
    if vocab_filepath and os.path.exists(vocab_filepath):
      self._init_vocab_from_file(vocab_filepath)
    else:
      assert vocab_list is not None
      self._init_vocab_from_list(vocab_list)
    self._separator = separator

  def encode(self, symbols_line):
    if isinstance(symbols_line, unicode):
      symbols_line = symbols_line.encode("utf-8")
    if self._separator:
      symbols_list = symbols_line.strip().split(self._separator)
    else:
      symbols_list = list(symbols_line.strip())
    return [self._sym_to_id[sym] for sym in symbols_list]

  def decode(self, ids):
    return " ".join(self.decode_list(ids))

  def decode_list(self, ids):
    return [self._id_to_sym[id_] for id_ in ids]

  @property
  def vocab_size(self):
    return len(self._id_to_sym)

  def _init_vocab_from_file(self, filename):
    """Load vocab from a file.

    Args:
      filename: The file to load vocabulary from.
    """
    def sym_gen():
      with tf.gfile.Open(filename) as f:
        for line in f:
          sym = line.strip()
          yield sym

    self._init_vocab(sym_gen(), add_reserved_symbols=False)

  def _init_vocab_from_list(self, vocab_list):
    """Initialize symbols from a list of symbols.

    It is ok if reserved symbols appear in the vocab list. They will be
    removed. The set of symbols in vocab_list should be unique.

    Args:
      vocab_list: A list of symbols.
    """
    def sym_gen():
      for sym in vocab_list:
        if sym not in text_encoder.RESERVED_TOKENS:
          yield sym

    self._init_vocab(sym_gen())

  def _init_vocab(self, sym_generator, add_reserved_symbols=True):
    """Initialize vocabulary with sym from sym_generator."""

    self._id_to_sym = {}
    non_reserved_start_index = 0

    if add_reserved_symbols:
      self._id_to_sym.update(enumerate(text_encoder.RESERVED_TOKENS))
      non_reserved_start_index = len(text_encoder.RESERVED_TOKENS)

    self._id_to_sym.update(
        enumerate(sym_generator, start=non_reserved_start_index))

    # _sym_to_id is the reverse of _id_to_sym
    self._sym_to_id = dict((v, k)
      for k, v in six.iteritems(self._id_to_sym))

  def store_to_file(self, filename):
    """Write vocab file to disk.

    Vocab files have one symbol per line. The file ends in a newline. Reserved
    symbols are written to the vocab file as well.

    Args:
      filename: Full path of the file to store the vocab to.
    """
    with tf.gfile.Open(filename, "w") as f:
      for i in xrange(len(self._id_to_sym)):
        f.write(self._id_to_sym[i] + "\n")


def tabbed_parsing_character_generator(data_dir, train, model_dir):
  """Generate source and target data from a single file."""
  #character_vocab = text_encoder.ByteTextEncoder()
  filename = "cmudict.dic.{0}".format("train" if train else "dev")
  pair_filepath = os.path.join(data_dir, filename)
  src_vocab_path = os.path.join(model_dir, "vocab.gr")
  tgt_vocab_path = os.path.join(model_dir, "vocab.ph")
  if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
    source_vocab = GraphemePhonemeEncoder(src_vocab_path)
    target_vocab = GraphemePhonemeEncoder(tgt_vocab_path, separator=" ")
    return tabbed_generator(pair_filepath, source_vocab, target_vocab, EOS)
  elif train:
    graphemes, phonemes = {}, {}
    data_filepath = os.path.join(data_dir, "cmudict.dic.train")
    with tf.gfile.GFile(data_filepath, mode="r") as data_file:
      for line in data_file:
        line_split = line.strip().split("\t")
        line_grs, line_phs = list(line_split[0]), line_split[1].split(" ")
        graphemes = update_vocab_symbols(graphemes, line_grs)
        phonemes = update_vocab_symbols(phonemes, line_phs)
    graphemes, phonemes = sorted(graphemes.keys()), sorted(phonemes.keys())
    source_vocab = GraphemePhonemeEncoder(vocab_filepath=src_vocab_path,
      vocab_list=graphemes)
    target_vocab = GraphemePhonemeEncoder(vocab_filepath=tgt_vocab_path,
      vocab_list=phonemes, separator=" ")
    source_vocab.store_to_file(src_vocab_path)
    target_vocab.store_to_file(tgt_vocab_path)
    return tabbed_generator(pair_filepath, source_vocab, target_vocab, EOS)
  else:
    raise IOError("Vocabulary files {} and {} not found.".format(src_vocab_path,
      tgt_vocab_path))


def update_vocab_symbols(init_vocab, update_syms):
  updated_vocab = init_vocab
  for sym in update_syms:
    updated_vocab.update({sym : 1})
  return updated_vocab


def tabbed_generator(source_path, source_vocab, target_vocab, eos=None):
  r"""Generator for sequence-to-sequence tasks using tabbed files.

  Tokens are derived from text files where each line contains both
  a source and a target string. The two strings are separated by a tab
  character ('\t'). It yields dictionaries of "inputs" and "targets" where
  inputs are characters from the source lines converted to integers, and
  targets are characters from the target lines, also converted to integers.

  Args:
    source_path: path to the file with source and target sentences.
    source_vocab: a SubwordTextEncoder to encode the source string.
    target_vocab: a SubwordTextEncoder to encode the target string.
    eos: integer to append at the end of each sequence (default: None).
  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from characters in the file lines.
  """
  eos_list = [] if eos is None else [eos]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    for line in source_file:
      if line and "\t" in line:
        parts = line.split("\t", 1)
        source, target = parts[0].strip(), parts[1].strip()
        source_ints = source_vocab.encode(source) + eos_list
        target_ints = target_vocab.encode(target) + eos_list
        yield {"inputs": source_ints, "targets": target_ints}

