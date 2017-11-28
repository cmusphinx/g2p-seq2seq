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
    self.decode_hparams = "beam_size=4,alpha=0.6"
    if flags:
      self.batch_size = flags.batch_size
      self.eval_steps = flags.eval_steps
      self.train_steps = len(open(data_path).readlines()) * flags.max_epochs
