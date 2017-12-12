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
# =============================================================================

"""Default Parameters class.
"""

import os
import json


class Params(object):
  """Class with training parameters."""
  def __init__(self, model_dir, data_path, flags=None):
    self.model_dir = os.path.expanduser(model_dir)
    self.data_dir = os.path.dirname(data_path)
    # Set default parameters first. Then update the parameters that
    # pointed out in flags.
    self.hparams_set = "transformer_small"
    self.schedule = "train_and_evaluate"
    self.model_name = "transformer"
    self.problem_name = "grapheme_to_phoneme_problem"
    self.train_steps = 10
    self.eval_steps = 1
    self.hparams = "batch_size=1,num_hidden_layers=1,hidden_size=4" +\
        ",filter_size=8,num_heads=1,length_bucket_step=2.0,max_length=50," +\
        ",min_length_bucket=5"
    self.decode_hparams = "beam_size=4,alpha=0.6,return_beams=True"

    if flags:
      self.batch_size = flags.batch_size
      self.eval_steps = flags.eval_steps
      if flags.max_epochs > 0:
        self.train_steps = len(open(data_path).readlines()) * flags.max_epochs
      elif flags.train:
        self.train_steps = 1000000
      self.hparams = "batch_size=" + str(flags.batch_size) +\
          ",num_hidden_layers=" + str(flags.num_layers) +\
          ",hidden_size=" + str(flags.size) +\
          ",filter_size=" + str(flags.filter_size) +\
          ",num_heads=" + str(flags.num_heads) +\
          ",length_bucket_step=" + str(flags.length_bucket_step) +\
          ",max_length=" + str(flags.max_length) +\
          ",min_length_bucket" + str(flags.min_length_bucket)

    saved_hparams_path = os.path.join(self.model_dir, "hparams.json")
    if os.path.exists(saved_hparams_path):
      saved_hparams_dic = json.load(open(saved_hparams_path))
      self.hparams = ""
      for hparam_idx, (hparam, hparam_value) in enumerate(
          saved_hparams_dic.items()):
        self.hparams += hparam + "=" + str(hparam_value)
        if hparam_idx < len(saved_hparams_dic) - 1:
          self.hparams += ","
