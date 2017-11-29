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
import codecs
import tensorflow as tf
import six

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

  def prepare_data(self, train_path=None, dev_path=None, test_path=None):
    if train_path:
      source_vocab, target_vocab = g2p_encoder.load_create_vocabs(
        self.params.model_dir, data_path=train_path)
      if dev_path:
        self.train_preprocess_file_path = self.problem.generate_data(train_path,
          self.params.model_dir, source_vocab, target_vocab)
        self.dev_preprocess_file_path = self.problem.generate_data(dev_path,
          source_vocab, target_vocab)
    elif test_path:
      source_vocab, target_vocab = g2p_encoder.load_create_vocabs(
        self.params.model_dir)
      self.test_preprocess_file_path = self.problem.generate_data(test_path,
        source_vocab, target_vocab)

  def train(self):
    g2p_trainer_utils.run(params=self.params,
      train_preprocess_file_path=self.train_preprocess_file_path,
      dev_preprocess_file_path=self.dev_preprocess_file_path)

  def __prepare_decode_model(self):
    hparams = trainer_utils.create_hparams(self.params.hparams_set,
      self.params.data_dir)
    g2p_trainer_utils.add_problem_hparams(hparams, self.params.problem_name,
      self.params.model_dir)
    self.estimator, _ = g2p_trainer_utils.create_experiment_components(
      params=self.params,
      hparams=hparams,
      run_config=trainer_utils.create_run_config(self.params.model_dir),
      dev_preprocess_file_path=self.test_preprocess_file_path)

    self.decode_hp = decoding.decode_hparams(self.params.decode_hparams)
    self.decode_hp.add_hparam("shards", 1)

  def interactive(self):
    self.__prepare_decode_model()
    decoding.decode_interactively(self.estimator, self.decode_hp)

  def decode(self, decode_from_file, decode_to_file):
    self.__prepare_decode_model()
    decoding.decode_from_file(self.estimator, decode_from_file, self.decode_hp,
      decode_to_file)

