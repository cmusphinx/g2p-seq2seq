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
import random
import tensorflow as tf

from tensorflow.python.data.ops import dataset_ops as dataset_ops
from g2p_seq2seq import g2p_encoder
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import generator_utils
from tensorflow.python.data.ops import dataset_ops as dataset_ops

EOS = text_encoder.EOS_ID


@registry.register_problem
class GraphemeToPhonemeProblem(problem.Text2TextProblem):
  """Problem spec for cmudict PRONALSYL Grapheme-to-Phoneme translation."""

  def __init__(self, model_dir, file_path, is_training):
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
    self.file_path = file_path
    if is_training:
      self.source_vocab, self.target_vocab = g2p_encoder.load_create_vocabs(
          self._model_dir, data_path=file_path)
    else:
      self.source_vocab, self.target_vocab = g2p_encoder.load_create_vocabs(
          self._model_dir, data_path=None)

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
    return self.tabbed_generator(data_path, source_vocab, target_vocab, EOS)

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

  def generate_data(self, file_path):
    """Generate cases from a generator and save as TFRecord files.

    Generated cases are transformed to tf.Example protos and saved as TFRecords
    in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

    Args:
      generator: a generator yielding (string -> int/float/str list)
                 dictionaries.
    """
    preprocess_file_path = os.path.join(
        self._model_dir,
        os.path.basename(file_path) + ".preprocessed")
    generate_files(
        self.generator(file_path, self.source_vocab, self.target_vocab),
        preprocess_file_path)
    return preprocess_file_path

  def get_feature_encoders(self, data_dir=None):
    if self._encoders is None:
      self._encoders = self.feature_encoders()
    return self._encoders

  def feature_encoders(self):
    targets_encoder = self.target_vocab
    if self.has_inputs:
      inputs_encoder = self.source_vocab
      return {"inputs": inputs_encoder, "targets": targets_encoder}
    return {"targets": targets_encoder}

  def tabbed_generator(self, source_path, source_vocab, target_vocab, eos=None):
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
        if line:
          if "\t" in line:
            parts = line.split("\t", 1)
            source, target = parts[0].strip(), parts[1].strip()
            source_ints = source_vocab.encode(source) + eos_list
            target_ints = target_vocab.encode(target) + eos_list
            yield {"inputs": source_ints, "targets": target_ints}
          elif " " in line:
            parts = line.split(" ")
            source, target = parts[0].strip(), " ".join(parts[1:]).strip()
            source_ints = source_vocab.encode(source) + eos_list
            target_ints = target_vocab.encode(target) + eos_list
            yield {"inputs": source_ints, "targets": target_ints}

  def dataset(self,
              mode,
              data_dir=None,
              num_threads=None,
              output_buffer_size=None,
              shuffle_files=None,
              hparams=None,
              preprocess=True,
              dataset_split=None,
              shard=None):
    """Build a Dataset for this problem.

    Args:
      mode: tf.estimator.ModeKeys; determines which files to read from.
      data_dir: directory that contains data files.
      num_threads: int, number of threads to use for decode and preprocess
        Dataset.map calls.
      output_buffer_size: int, how many elements to prefetch in Dataset.map
        calls.
      shuffle_files: whether to shuffle input files. Default behavior (i.e. when
        shuffle_files=None) is to shuffle if mode == TRAIN.
      hparams: tf.contrib.training.HParams; hparams to be passed to
        Problem.preprocess_example and Problem.hparams. If None, will use a
        default set that is a no-op.
      preprocess: bool, whether to map the Dataset through
        Problem.preprocess_example.
      dataset_split: tf.estimator.ModeKeys + ["test"], which split to read data
        from (TRAIN:"-train", EVAL:"-dev", "test":"-test"). Defaults to mode.
      shard: int, if provided, will only read data from the specified shard.

    Returns:
      Dataset containing dict<feature name, Tensor>.
    """
    if dataset_split or mode == "train":
      # In case when pathes to preprocessed files pointed out or if train mode
      # launched, we save preprocessed data first, and then create dataset from
      # that files.
      dataset_split = dataset_split or mode
      assert data_dir

      if not hasattr(hparams, "data_dir"):
        hparams.add_hparam("data_dir", data_dir)
      if not hparams.data_dir:
        hparams.data_dir = data_dir
      # Construct the Problem's hparams so that items within it are accessible
      _ = self.get_hparams(hparams)

      data_fields, data_items_to_decoders = self.example_reading_spec()
      if data_items_to_decoders is None:
        data_items_to_decoders = {
            field: tf.contrib.slim.tfexample_decoder.Tensor(field)
            for field in data_fields}

      is_training = mode == tf.estimator.ModeKeys.TRAIN
      data_filepattern = self.filepattern(data_dir, dataset_split, shard=shard)
      tf.logging.info("Reading data files from %s", data_filepattern)
      data_files = tf.contrib.slim.parallel_reader.get_data_files(
          data_filepattern)
      if shuffle_files or shuffle_files is None and is_training:
        random.shuffle(data_files)

      dataset = tf.contrib.data.TFRecordDataset(data_files)

    else:
      # In case when pathes to preprocessed files not pointed out, we create
      # dataset from generator object.
      eos_list = [] if EOS is None else [EOS]
      data_list = []
      with tf.gfile.GFile(self.file_path, mode="r") as source_file:
        for line in source_file:
          if line:
            if "\t" in line:
              parts = line.split("\t", 1)
              source, target = parts[0].strip(), parts[1].strip()
              source_ints = self.source_vocab.encode(source) + eos_list
              target_ints = self.target_vocab.encode(target) + eos_list
              data_list.append({"inputs":source_ints, "targets":target_ints})
            else:
              source_ints = self.source_vocab.encode(line) + eos_list
              data_list.append(generator_utils.to_example(
                  {"inputs":source_ints}))

      gen = Gen(self.generator(self.file_path, self.source_vocab,
                               self.target_vocab))
      dataset = dataset_ops.Dataset.from_generator(gen, tf.string)

      preprocess = False

    def decode_record(record):
      """Serialized Example to dict of <feature name, Tensor>."""
      decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
          data_fields, data_items_to_decoders)

      decode_items = list(data_items_to_decoders)
      decoded = decoder.decode(record, items=decode_items)
      return dict(zip(decode_items, decoded))

    def _preprocess(example):
      """Whether preprocess data into required format."""
      example = self.preprocess_example(example, mode, hparams)
      self.maybe_reverse_features(example)
      self.maybe_copy_features(example)
      return example

    dataset = dataset.map(decode_record, num_parallel_calls=4)

    if preprocess:
      dataset = dataset.map(
          _preprocess,
          num_threads=num_threads,
          output_buffer_size=output_buffer_size)

    return dataset


class Gen:
  """Generator class for dataset creation.
  Function dataset_ops.Dataset.from_generator() required callable generator
  object."""

  def __init__(self, gen):
    """ Initialize generator."""
    self._gen = gen

  def __call__(self):
    for case in self._gen:
      source_ints = case["inputs"]
      target_ints = case["targets"]
      yield generator_utils.to_example({"inputs":source_ints,
                                        "targets":target_ints})


def generate_files(generator, output_filename):
  """Generate cases from a generator and save as TFRecord files.

  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

  Args:
    generator: a generator yielding (string -> int/float/str list) dictionaries.
    output_filename: output file paths.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for case in generator:
    sequence_example = generator_utils.to_example(case)
    writer.write(sequence_example.SerializeToString())
  writer.close()
