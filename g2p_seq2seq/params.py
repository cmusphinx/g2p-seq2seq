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
    self.data_dir_name = os.path.dirname(data_path)
    # Set default parameters first. Then update the parameters that
    # pointed out in flags.
    self.hparams_set = "transformer_base"
    self.schedule = "continuous_train_and_eval"
    self.model_name = "transformer"
    self.problem_name = "grapheme_to_phoneme_problem"
    self.train_steps = 10
    self.eval_steps = 1
    self.iterations_per_loop = 2
    self.local_eval_frequency = 5
    self.hparams = "eval_drop_long_sequences=1,batch_size=1," +\
        "num_hidden_layers=1,hidden_size=4,filter_size=8,num_heads=1," +\
        "length_bucket_step=2.0,max_length=10,min_length_bucket=5"
    self.decode_hparams = "beam_size=1,alpha=0.6,return_beams=False"
    self.master = ""
    self.p2g = False

    if flags:
      self.p2g = flags.p2g
      self.batch_size = flags.batch_size
      self.iterations_per_loop = min(1000, max(10, int(self.batch_size/10)))
      if flags.max_epochs > 0:
        self.train_steps = max(10000, 
                               int(len(open(data_path).readlines()) /\
                                   self.batch_size) *\
                               self.iterations_per_loop *\
                               flags.max_epochs)
      elif flags.train:
        self.train_steps = 200000

      self.eval_steps = min(200, int(self.train_steps/1000))
      self.local_eval_frequency = min(2000, max(20, int(self.train_steps/100)))

      if flags.hparams:
          self.hparams = flags.hparams + ","
      else:
          self.hparams = ""
      self.hparams += "eval_drop_long_sequences=1" +\
          ",batch_size=" + str(flags.batch_size) +\
          ",num_hidden_layers=" + str(flags.num_layers) +\
          ",hidden_size=" + str(flags.size) +\
          ",filter_size=" + str(flags.filter_size) +\
          ",num_heads=" + str(flags.num_heads) +\
          ",length_bucket_step=" + str(flags.length_bucket_step) +\
          ",max_length=" + str(flags.max_length) +\
          ",min_length_bucket=" + str(flags.min_length_bucket)
      self.decode_hparams = "beam_size=" + str(flags.beam_size) +\
          ",alpha=" + str(flags.alpha)
      if flags.return_beams:
          self.decode_hparams += ",return_beams=True"
      else:
          self.decode_hparams += ",return_beams=False"

    self.tpu_num_shards = 8
    self.log_device_replacement = False
    self.keep_checkpoint_max = 1
    self.keep_checkpoint_every_n_hours = 1
    self.worker_gpu = 1
    self.gpu_order = ""
    self.locally_shard_to_cpu = False
    self.worker_replicas = 1
    self.worker_gpu_memory_fraction = 0.95
    self.experimental_optimize_placement = False
    self.use_tpu = False
    self.no_data_parallelism = False
    self.daisy_chain_variables = True
    self.ps_replicas = 0
    self.ps_job = "/job:ps"
    self.ps_gpu = 0
    self.sync = False
    self.worker_id = 0
    self.worker_job = "/job:localhost"
    self.export_saved_model = False
    self.tfdbg = False
    self.dbgprofile = False
    self.eval_early_stopping_steps = None
    self.eval_early_stopping_metric = "loss"
    self.eval_early_stopping_metric_minimize = True
    self.profile = False
    self.decode_shards = 1

    saved_hparams_path = os.path.join(self.model_dir, "hparams.json")
    if os.path.exists(saved_hparams_path):
      saved_hparams_dic = json.load(open(saved_hparams_path))
      self.hparams = ""
      for hparam_idx, (hparam, hparam_value) in enumerate(
          saved_hparams_dic.items()):
        self.hparams += hparam + "=" + str(hparam_value)
        if hparam_idx < len(saved_hparams_dic) - 1:
          self.hparams += ","
