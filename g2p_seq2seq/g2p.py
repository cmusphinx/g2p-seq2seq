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
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
import tensorflow as tf
import numpy as np

from g2p_seq2seq import g2p_trainer_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import decoding

from tensor2tensor.data_generators import text_encoder

from IPython.core.debugger import Tracer

EOS = text_encoder.EOS


class G2PModel(object):
  """Grapheme-to-Phoneme translation model class.
  """
  def __init__(self, params, file_path="", is_training=False):
    # Point out the current directory with t2t problem specified for g2p task.
    usr_dir.import_usr_dir(os.path.dirname(os.path.abspath(__file__)))
    self.params = params
    self.file_path = file_path
    # Register g2p problem.
    self.problem = registry._PROBLEMS[self.params.problem_name](
        self.params.model_dir, file_path=file_path, is_training=is_training)
    trainer_utils.log_registry()
    if not os.path.exists(self.params.model_dir):
      os.makedirs(self.params.model_dir)
    self.train_preprocess_file_path, self.dev_preprocess_file_path = None, None

  def prepare_data(self, train_path=None, dev_path=None):
    """Prepare preprocessed datafiles."""
    self.train_preprocess_file_path = self.problem.generate_data(train_path)
    self.dev_preprocess_file_path = self.problem.generate_data(dev_path)

  def train(self):
    """Run training."""
    g2p_trainer_utils.run(
        params=self.params,
        problem_instance=self.problem,
        train_preprocess_file_path=self.train_preprocess_file_path,
        dev_preprocess_file_path=self.dev_preprocess_file_path)

  def __prepare_decode_model(self):
    """Prepare utilities for decoding."""
    hparams = trainer_utils.create_hparams(
        self.params.hparams_set,
        self.params.data_dir,
        passed_hparams=self.params.hparams)
    estimator, _ = g2p_trainer_utils.create_experiment_components(
        params=self.params,
        hparams=hparams,
        run_config=trainer_utils.create_run_config(self.params.model_dir),
        problem_instance=self.problem)

    decode_hp = decoding.decode_hparams(self.params.decode_hparams)
    decode_hp.add_hparam("shards", 1)
    return estimator, decode_hp

  def interactive(self):
    """Run interactive mode."""
    estimator, decode_hp = self.__prepare_decode_model()
    decoding.decode_interactively(estimator, decode_hp)

  def decode(self, output_file_path):
    """Run decoding mode."""
    estimator, decode_hp = self.__prepare_decode_model()
    sorted_inputs, sorted_keys, decodes = decode_from_file(
        estimator, self.file_path, decode_hp)

    # Dumping inputs and outputs to file filename.decodes in
    # format result\tinput in the same order as original inputs
    if output_file_path:
      tf.logging.info("Writing decodes into %s" % output_file_path)
      outfile = tf.gfile.Open(output_file_path, "w")
      for index in range(len(sorted_inputs)):
        outfile.write("%s%s" % (decodes[sorted_keys[index]],
                                decode_hp.delimiter))

  def evaluate(self):
    """Run evaluation mode."""
    words, pronunciations = [], []
    for case in self.problem.generator(self.file_path,
                                       self.problem.source_vocab,
                                       self.problem.target_vocab):
      word = self.problem.source_vocab.decode(case["inputs"]).replace(
          EOS, "").strip()
      pronunciation = self.problem.target_vocab.decode(case["targets"]).replace(
          EOS, "").strip()
      words.append(word)
      pronunciations.append(pronunciation)

    g2p_gt_map = create_g2p_gt_map(words, pronunciations)

    estimator, decode_hp = self.__prepare_decode_model()
    errors = calc_errors(g2p_gt_map, estimator, self.file_path, decode_hp)

    print("Words: %d" % len(g2p_gt_map))
    print("Errors: %d" % errors)
    print("WER: %.3f" % (float(errors)/len(g2p_gt_map)))
    print("Accuracy: %.3f" % float(1.-(float(errors)/len(g2p_gt_map))))
    return estimator, decode_hp, g2p_gt_map


def create_g2p_gt_map(words, pronunciations):
  """Create grapheme-to-phoneme ground true mapping."""
  g2p_gt_map = {}
  for word, pronunciation in zip(words, pronunciations):
    if word in g2p_gt_map:
      g2p_gt_map[word].append(pronunciation)
    else:
      g2p_gt_map[word] = [pronunciation]
  return g2p_gt_map


def calc_errors(g2p_gt_map, estimator, decode_file_path, decode_hp):
  """Calculate a number of prediction errors."""
  sorted_inputs, sorted_keys, decodes = decode_from_file(
      estimator, decode_file_path, decode_hp)

  errors = 0
  for index, word in enumerate(sorted_inputs):
    if decodes[sorted_keys[index]] not in g2p_gt_map[word]:
      errors += 1
  return errors


def decode_from_file(estimator, filename, decode_hp):
  """Compute predictions on entries in filename and write them out."""

  if not decode_hp.batch_size:
    decode_hp.batch_size = 32
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)

  hparams = estimator.params
  problem_id = decode_hp.problem_idx
  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  inputs_vocab = hparams.problems[problem_id].vocabulary["inputs"]
  targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
  problem_name = "grapheme_to_phoneme_problem"
  tf.logging.info("Performing decoding from a file.")
  sorted_inputs, sorted_keys = _get_sorted_inputs(filename)
  num_decode_batches = (len(sorted_inputs) - 1) // decode_hp.batch_size + 1

  def input_fn():
    """Function for inputs generator."""
    input_gen = decoding._decode_batch_input_fn(
        problem_id, num_decode_batches, sorted_inputs, inputs_vocab,
        decode_hp.batch_size, decode_hp.max_input_size)
    gen_fn = decoding.make_input_fn_from_generator(input_gen)
    example = gen_fn()
    return decoding._decode_input_tensor_to_features_dict(example, hparams)

  decodes = []
  result_iter = estimator.predict(input_fn)
  for result in result_iter:
    if decode_hp.return_beams:
      beam_decodes = []
      output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
      for k, beam in enumerate(output_beams):
        tf.logging.info("BEAM %d:" % k)
        decoded_outputs, _ = decoding.log_decode_results(
            result["inputs"],
            beam,
            problem_name,
            None,
            inputs_vocab,
            targets_vocab)
        beam_decodes.append(decoded_outputs)
      decodes.append("\t".join(beam_decodes))
    else:
      decoded_outputs, _ = decoding.log_decode_results(
          result["inputs"],
          result["outputs"],
          problem_name,
          None,
          inputs_vocab,
          targets_vocab)
      decodes.append(decoded_outputs)

  # Reversing the decoded inputs and outputs because they were reversed in
  # _decode_batch_input_fn
  sorted_inputs.reverse()
  decodes.reverse()
  return sorted_inputs, sorted_keys, decodes


def _get_sorted_inputs(filename):
  """Returning inputs sorted according to length.

  Args:
    filename: path to file with inputs, 1 per line.
    num_shards: number of input shards. If > 1, will read from file filename.XX,
      where XX is FLAGS.worker_id.
    delimiter: str, delimits records in the file.

  Returns:
    a sorted list of inputs

  """
  tf.logging.info("Getting sorted inputs")

  inputs = []
  with tf.gfile.Open(filename) as input_file:
    lines = input_file.readlines()
    for line in lines:
      if "\t" in line:
        parts = line.strip().split("\t")
        inputs.append(parts[0])
      elif " " in line:
        parts = line.strip().split(" ")
        inputs.append(parts[0])
      else:
        inputs.append(line.strip())
  input_lens = [(i, len(line)) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1))
  # We'll need the keys to rearrange the inputs back into their original order
  sorted_keys = {}
  sorted_inputs = []
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs.append(inputs[index])
    sorted_keys[index] = i
  return sorted_inputs, sorted_keys
