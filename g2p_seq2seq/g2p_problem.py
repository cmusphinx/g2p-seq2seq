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

"""Module for registering custom Grapheme-to-Phoneme problem in tensor2tensor.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from g2p_seq2seq import g2p_encoder
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import generator_utils

from IPython.core.debugger import Tracer

EOS = text_encoder.EOS_ID

@registry.register_problem
class GraphemeToPhonemeProblem(problem.Text2TextProblem):
  """Problem spec for cmudict PRONALSYL Grapheme-to-Phoneme translation."""

  def __init__(self, model_dir):
    """Create a Problem.

    Args:
      was_reversed: bool, whether to reverse inputs and targets.
      was_copy: bool, whether to copy inputs to targets. Can be composed with
        was_reversed so that if both are true, the targets become the inputs,
        which are then copied to targets so that the task is targets->targets.
    """
    super(GraphemeToPhonemeProblem, self).__init__()
    self._encoders = None
    self._hparams = None
    self._feature_info = None
    self._model_dir = model_dir
    self._target_vocab_size = None

  def generator(self, data_path, source_vocab, target_vocab):
    """Generator for the training and evaluation data.
    Generate source and target data from a single file.

    Args:
      data_path: The path to data file.
      source_vocab: the object of GraphemePhonemeEncoder class with encode and
        decode functions for symbols from source file.
      target_vocab: the object of GraphemePhonemeEncoder class with encode and
        decode functions for symbols from target file.

    Yields:
      dicts with keys "inputs" and "targets", with values being lists of token
      ids.
    """
    return tabbed_generator(data_path, source_vocab, target_vocab, EOS)

  def filepattern(self, data_dir, dataset_split, shard=None):
    return os.path.join(data_dir, dataset_split)

  @property
  def input_space_id(self):
    return 0

  @property
  def target_space_id(self):
    return 0

  @property
  def num_shards(self):
    return 1

  @property
  def use_subword_tokenizer(self):
    return False

  @property
  def is_character_level(self):
    return False

  @property
  def targeted_vocab_size(self):
    return None

  @property
  def vocab_name(self):
    return None

  def generate_data(self, file_path, source_vocab, target_vocab):
    preprocess_file_path = os.path.join(
        self._model_dir,
        os.path.basename(file_path) + ".preprocessed")
    generate_files(
        self.generator(file_path, source_vocab, target_vocab),
        preprocess_file_path)
    return preprocess_file_path

  def get_feature_encoders(self, data_dir=None):
    if self._encoders is None:
      self._encoders = self.feature_encoders()
    return self._encoders

  def feature_encoders(self):
    targets_encoder = g2p_encoder.GraphemePhonemeEncoder(
        vocab_path=os.path.join(self._model_dir, "vocab.ph"), separator=" ")
    if self.has_inputs:
      inputs_encoder = g2p_encoder.GraphemePhonemeEncoder(
          vocab_path=os.path.join(self._model_dir, "vocab.gr"))
      return {"inputs": inputs_encoder, "targets": targets_encoder}
    return {"targets": targets_encoder}


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


def generate_files(generator, output_filename):
  """Generate cases from a generator and save as TFRecord files.

  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

  Args:
    generator: a generator yielding (string -> int/float/str list) dictionaries.
    output_filenames: List of output file paths.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  counter = 0
  for case in generator:
    if counter > 0 and counter % 100000 == 0:
      tf.logging.info("Generating case %d." % counter)
    counter += 1
    sequence_example = generator_utils.to_example(case)
    writer.write(sequence_example.SerializeToString())

  writer.close()
