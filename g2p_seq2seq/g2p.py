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
import tensorflow as tf
import numpy as np

from g2p_seq2seq import g2p_trainer_utils
from g2p_seq2seq import g2p_encoder
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import decoding

from IPython.core.debugger import Tracer


class G2PModel(object):
  """Grapheme-to-Phoneme translation model class.
  """
  def __init__(self, params):
    # Point out the current directory with t2t problem specified for g2p task.
    usr_dir.import_usr_dir(os.path.dirname(os.path.abspath(__file__)))
    self.params = params
    # Register g2p problem.
    self.problem = registry._PROBLEMS[self.params.problem_name](
        self.params.model_dir)
    trainer_utils.log_registry()
    if not os.path.exists(self.params.model_dir):
      os.makedirs(self.params.model_dir)
    self.train_preprocess_file_path, self.dev_preprocess_file_path,\
      self.test_preprocess_file_path = None, None, None


  def prepare_data(self, train_path=None, dev_path=None, test_path=None):
    """Prepare preprocessed datafiles.
    """
    if train_path:
      source_vocab, target_vocab = g2p_encoder.load_create_vocabs(
          self.params.model_dir, data_path=train_path)
      if dev_path:
        self.train_preprocess_file_path = self.problem.generate_data(
            train_path, source_vocab, target_vocab)
        self.dev_preprocess_file_path = self.problem.generate_data(
            dev_path, source_vocab, target_vocab)
    elif test_path:
      source_vocab, target_vocab = g2p_encoder.load_create_vocabs(
          self.params.model_dir)
      self.test_preprocess_file_path = self.problem.generate_data(
          test_path, source_vocab, target_vocab)

  def train(self):
    """Run training."""
    g2p_trainer_utils.run(
        params=self.params,
        train_preprocess_file_path=self.train_preprocess_file_path,
        dev_preprocess_file_path=self.dev_preprocess_file_path)

  def __prepare_decode_model(self):
    """Prepare utilities for decoding."""
    hparams = trainer_utils.create_hparams(
        self.params.hparams_set,
        self.params.data_dir,
        passed_hparams=self.params.hparams)
    g2p_trainer_utils.add_problem_hparams(
        hparams,
        self.params.problem_name,
        self.params.model_dir)
    estimator, _ = g2p_trainer_utils.create_experiment_components(
        params=self.params,
        hparams=hparams,
        run_config=trainer_utils.create_run_config(self.params.model_dir),
        dev_preprocess_file_path=self.test_preprocess_file_path)

    decode_hp = decoding.decode_hparams(self.params.decode_hparams)
    decode_hp.add_hparam("shards", 1)
    return estimator, decode_hp

  def interactive(self):
    """Run interactive mode."""
    estimator, decode_hp = self.__prepare_decode_model()
    decoding.decode_interactively(estimator, decode_hp)

  def decode(self, decode_file_path, output_file_path):
    """Run decoding mode."""
    estimator, decode_hp = self.__prepare_decode_model()
    sorted_inputs, sorted_keys, decodes = decode_from_file(
        estimator, decode_file_path, decode_hp)

    # Dumping inputs and outputs to file filename.decodes in
    # format result\tinput in the same order as original inputs
    if output_file_path:
      tf.logging.info("Writing decodes into %s" % output_file_path)
      outfile = tf.gfile.Open(output_file_path, "w")
      for index in range(len(sorted_inputs)):
        outfile.write("%s%s" % (decodes[sorted_keys[index]],
                                decode_hp.delimiter))

  def evaluate(self, gt_file_path, decode_file_path):
    """Run evaluation mode."""
    gt_lines = open(gt_file_path).readlines()
    g2p_gt_map = create_g2p_gt_map(gt_lines)

    estimator, decode_hp = self.__prepare_decode_model()
    errors = calc_errors(g2p_gt_map, decode_file_path, estimator, decode_hp)

    print("Words: %d" % len(g2p_gt_map))
    print("Errors: %d" % errors)
    print("WER: %.3f" % (float(errors)/len(g2p_gt_map)))
    print("Accuracy: %.3f" % float(1.-(float(errors)/len(g2p_gt_map))))
    return estimator, decode_hp


def create_g2p_gt_map(gt_lines):
  """Create grapheme-to-phoneme ground true mapping."""
  g2p_gt_map = {}
  for line in gt_lines:
    line_split = line.strip().split("\t")
    if line_split[0] in g2p_gt_map:
      g2p_gt_map[line_split[0]].append(line_split[1])
    else:
      g2p_gt_map[line_split[0]] = [line_split[1]]
  return g2p_gt_map


def calc_errors(g2p_gt_map, decode_file_path, estimator, decode_hp):
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
  has_input = "inputs" in hparams.problems[problem_id].vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = hparams.problems[problem_id].vocabulary[inputs_vocab_key]
  targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
  problem_name = "grapheme_to_phoneme_problem"
  tf.logging.info("Performing decoding from a file.")
  sorted_inputs, sorted_keys = decoding._get_sorted_inputs(
      filename, decode_hp.shards, decode_hp.delimiter)
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
