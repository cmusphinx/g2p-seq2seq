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
import re
from collections import OrderedDict
import tensorflow as tf

from tensorflow.python.data.ops import dataset_ops as dataset_ops
from g2p_seq2seq import g2p_encoder
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import text_problems

EOS = text_encoder.EOS_ID


@registry.register_problem
class GraphemeToPhonemeProblem(text_problems.Text2TextProblem):
  """Problem spec for cmudict PRONALSYL Grapheme-to-Phoneme translation."""

  def __init__(self, model_dir, train_path=None, dev_path=None, test_path=None,
               cleanup=False, p2g_mode=False):
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
    self.train_path, self.dev_path, self.test_path = train_path, dev_path,\
        test_path
    self.cleanup = cleanup
    self.p2g_mode = p2g_mode
    vocab_filename = os.path.join(self._model_dir, "vocab.g2p")
    if train_path:
      self.train_path, self.dev_path, self.test_path = create_data_files(
          init_train_path=train_path, init_dev_path=dev_path,
          init_test_path=test_path, cleanup=self.cleanup,
          p2g_mode=self.p2g_mode)
      self.source_vocab, self.target_vocab = g2p_encoder.load_create_vocabs(
          vocab_filename, train_path=self.train_path, dev_path=self.dev_path,
          test_path=self.test_path, p2g_mode=self.p2g_mode)
    elif not os.path.exists(os.path.join(self._model_dir, "checkpoint")):
      raise Exception("Model not found in {}".format(self._model_dir))
    else:
      self.source_vocab, self.target_vocab = g2p_encoder.load_create_vocabs(
          vocab_filename, p2g_mode=self.p2g_mode)

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
    if not (".preprocessed" in dataset_split):
      return os.path.join(self._model_dir, dataset_split + ".preprocessed")
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

  def generate_preprocess_data(self):
    """Generate and save preprocessed data as TFRecord files.

    Args:
      train_path: the path to the train data file.
      eval_path: the path to the evaluation data file.

    Returns:
      train_preprocess_path: the path where the preprocessed train data
          was saved.
      eval_preprocess_path: the path where the preprocessed evaluation data
          was saved.
    """
    train_preprocess_path = os.path.join(self._model_dir, "train.preprocessed")
    eval_preprocess_path = os.path.join(self._model_dir, "eval.preprocessed")
    train_gen = self.generator(self.train_path, self.source_vocab,
                               self.target_vocab)
    eval_gen = self.generator(self.dev_path, self.source_vocab,
                              self.target_vocab)

    generate_preprocess_files(train_gen, eval_gen, train_preprocess_path,
                              eval_preprocess_path)
    return train_preprocess_path, eval_preprocess_path

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
      for line_idx, line in enumerate(source_file):
        if line:
          source, target = split_graphemes_phonemes(line,
                                                    cleanup=self.cleanup,
                                                    p2g_mode=self.p2g_mode)
          if not (source and target):
            tf.logging.warning("Invalid data format in line {} in {}:\n"
                "{}\nGraphemes and phonemes should be separated by white space."
                .format(line_idx, source_path, line))
            continue
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
              shard=None,
              partition_id=0,
              num_partitions=1):
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
    if dataset_split or (mode in ["train", "eval"]):
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

    else:
      # In case when pathes to preprocessed files not pointed out, we create
      # dataset from generator object.
      eos_list = [] if EOS is None else [EOS]
      data_list = []
      with tf.gfile.GFile(self.test_path, mode="r") as source_file:
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

      gen = Gen(self.generator(self.test_path, self.source_vocab,
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

    dataset = (tf.data.Dataset.from_tensor_slices(data_files)
               .interleave(lambda x:
                   tf.data.TFRecordDataset(x).map(decode_record,
                                                  num_parallel_calls=4),
                   cycle_length=4, block_length=16))

    if preprocess:
      dataset = dataset.map(_preprocess, num_parallel_calls=4)

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


def generate_preprocess_files(train_gen, dev_gen, train_preprocess_path,
                              dev_preprocess_path):
  """Generate cases from a generators and save as TFRecord files.

  Generated cases are transformed to tf.Example protos and saved as TFRecords
  in sharded files named output_dir/output_name-00..N-of-00..M=num_shards.

  Args:
    train_gen: a generator yielding (string -> int/float/str list) train data.
    dev_gen: a generator yielding development data.
    train_preprocess_path: path to the file where preprocessed train data
        will be saved.
    dev_preprocess_path: path to the file where preprocessed development data
        will be saved.
  """
  if dev_gen:
    gen_file(train_gen, train_preprocess_path)
    gen_file(dev_gen, dev_preprocess_path)
  else:
    # In case when development generator was not given, we create development
    # preprocess file from train generator.
    train_writer = tf.python_io.TFRecordWriter(train_preprocess_path)
    dev_writer = tf.python_io.TFRecordWriter(dev_preprocess_path)
    line_counter = 1
    for case in train_gen:
      sequence_example = generator_utils.to_example(case)
      if line_counter % 20 == 0:
        dev_writer.write(sequence_example.SerializeToString())
      else:
        train_writer.write(sequence_example.SerializeToString())
      line_counter += 1
    train_writer.close()
    dev_writer.close()


def gen_file(generator, output_file_path):
  """Generate cases from generator and save as TFRecord file.

  Args:
    generator: a generator yielding (string -> int/float/str list) data.
    output_file_path: path to the file where preprocessed data will be saved.
  """
  writer = tf.python_io.TFRecordWriter(output_file_path)
  for case in generator:
    sequence_example = generator_utils.to_example(case)
    writer.write(sequence_example.SerializeToString())
  writer.close()


def create_data_files(init_train_path, init_dev_path, init_test_path,
                      cleanup=False, p2g_mode=False):
  """Create train, development and test data files from initial data files
  in case when not provided development or test data files or active cleanup
  flag.

  Args:
    init_train_path: path to the train data file.
    init_dev_path: path to the development data file.
    init_test_path: path to the test data file.
    cleanup: flag indicating whether to cleanup datasets from stress and
             comments.

  Returns:
    train_path: path to the new train data file generated from initially
      provided data.
    dev_path: path to the new development data file generated from initially
      provided data.
    test_path: path to the new test data file generated from initially
      provided data.
  """
  train_path, dev_path, test_path = init_train_path, init_dev_path,\
      init_test_path

  if (init_dev_path and init_test_path and os.path.exists(init_dev_path) and
      os.path.exists(init_test_path)):
    if not cleanup:
      return init_train_path, init_dev_path, init_test_path

  else:
    train_path = init_train_path + ".part.train"
    if init_dev_path:
      if not os.path.exists(init_dev_path):
        raise IOError("File {} not found.".format(init_dev_path))
    else:
      dev_path = init_train_path + ".part.dev"

    if init_test_path:
      if not os.path.exists(init_test_path):
        raise IOError("File {} not found.".format(init_test_path))
    else:
      test_path = init_train_path + ".part.test"

  if cleanup:
    train_path += ".cleanup"
    dev_path += ".cleanup"
    test_path += ".cleanup"

  train_dic, dev_dic, test_dic = OrderedDict(), OrderedDict(), OrderedDict()

  source_dic = collect_pronunciations(source_path=init_train_path,
                                      cleanup=cleanup, p2g_mode=p2g_mode)
  if init_dev_path:
    dev_dic = collect_pronunciations(source_path=init_dev_path,
                                     cleanup=cleanup, p2g_mode=p2g_mode)
  if init_test_path:
    test_dic = collect_pronunciations(source_path=init_test_path,
                                      cleanup=cleanup, p2g_mode=p2g_mode)

  #Split dictionary to train, validation and test (if not assigned).
  for word_counter, (word, pronunciations) in enumerate(source_dic.items()):
    if word_counter % 20 == 19 and not init_dev_path:
      dev_dic[word] = pronunciations
    elif ((word_counter % 20 == 18 or word_counter % 20 == 17) and
          not init_test_path):
      test_dic[word] = pronunciations
    else:
      train_dic[word] = pronunciations

  save_dic(train_dic, train_path)
  if not init_dev_path or cleanup:
    save_dic(dev_dic, dev_path)
  if not init_test_path or cleanup:
    save_dic(test_dic, test_path)
  return train_path, dev_path, test_path


def collect_pronunciations(source_path, cleanup=False, p2g_mode=False):
  """Create dictionary mapping word to its different pronounciations.

  Args:
    source_path: path to the data file;
    cleanup: flag indicating whether to cleanup datasets from stress and
             comments.

  Returns:
    dic: dictionary mapping word to its pronunciations.
  """
  dic = OrderedDict()
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    word_counter = 0
    for line in source_file:
      if line:
        source, target = split_graphemes_phonemes(line, cleanup=cleanup,
                                                  p2g_mode=p2g_mode)
        if not (source, target):
          tf.logging.warning("Invalid data format in line {} in {}:\n"
              "{}\nGraphemes and phonemes should be separated by white space."
              .format(line_idx, source_path, line))
          continue
        if source in dic:
          dic[source].append(target)
        else:
          dic[source] = [target]
  return dic


def split_graphemes_phonemes(input_line, cleanup=False, p2g_mode=False):
  """Split line into graphemes and phonemes.

  Args:
    input_line: raw input line;
    cleanup: flag indicating whether to cleanup datasets from stress and
             comments.

  Returns:
    graphemes: graphemes string;
    phonemes: phonemes string.
  """
  line = input_line
  if cleanup:
    clean_pattern = re.compile(r"(\[.*\]|\{.*\}|\(.*\)|#.*)")
    stress_pattern = re.compile(r"(?<=[a-zA-Z])\d+")
    line = re.sub(clean_pattern, r"", line)
    line = re.sub(stress_pattern, r"", line)

  items = line.split()
  source, target = None, None
  if len(items) > 1:
    if not p2g_mode:
      source, target = items[0].strip(), " ".join(items[1:]).strip()
    else:
      source, target = " ".join(items[:-1]).strip(), items[-1].strip()
  return source, target


def save_dic(dic, save_path):
  with tf.gfile.GFile(save_path, mode="w") as save_file:
    for word, pronunciations in dic.items():
      for pron in pronunciations:
        save_file.write(word + " " + pron + "\n")
