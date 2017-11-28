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
import codecs
import tensorflow as tf
import six

from . import g2p_trainer_utils
#from tensor2tensor.data_generators import text_encoder
#from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import decoding

#EOS = text_encoder.EOS_ID

from IPython.core.debugger import Tracer

tf.flags.DEFINE_string("model_dir", None, "Training directory.")
tf.flags.DEFINE_boolean("interactive", False,
                        "Set to True for interactive decoding.")
tf.flags.DEFINE_string("evaluate", "", "Count word error rate for file.")
tf.flags.DEFINE_string("decode", "", "Decode file.")
tf.flags.DEFINE_string("output", "", "Decoding result file.")
tf.flags.DEFINE_string("train", "", "Train dictionary.")
tf.flags.DEFINE_string("valid", "", "Development dictionary.")
tf.flags.DEFINE_string("test", "", "Test dictionary.")
tf.flags.DEFINE_boolean("reinit", False,
                        "Set to True for training from scratch.")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64,
                        "Batch size to use during training.")
tf.flags.DEFINE_integer("max_epochs", 10,
                        "How many training steps to do until stop training"
                        " (0: no limit).")
tf.flags.DEFINE_integer("eval_every_n_steps", 1000,
                        "Run evaluation on validation data every N steps.")
tf.flags.DEFINE_string("master", "", "Address of TensorFlow master.")
tf.flags.DEFINE_string("schedule", "train_and_evaluate", "Set schedule.")

FLAGS = tf.app.flags.FLAGS


def main(_=[]):
  """Main function.
  """

  FLAGS.hparams_set = "transformer_small"
  FLAGS.schedule = "train_and_evaluate"
  FLAGS.problems = "grapheme_to_phoneme_problem"
  FLAGS.decode_hparams = "beam_size=4,alpha=0.6"
  FLAGS.decode_to_file = "decode_output.txt"
  problem_name = "grapheme_to_phoneme_problem"

  tf.logging.set_verbosity(tf.logging.INFO)
  usr_dir.import_usr_dir(os.path.dirname(os.path.abspath(__file__)))
  problem = registry._PROBLEMS[problem_name](FLAGS.model_dir)
  trainer_utils.log_registry()
  output_dir = os.path.expanduser(FLAGS.model_dir)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  if FLAGS.train:
    train_preprocess_file_path = problem.generate_data(FLAGS.train, FLAGS.model_dir, train_flag=True)
    dev_preprocess_file_path = problem.generate_data(FLAGS.valid, FLAGS.model_dir, train_flag=False)
    data_dir = os.path.dirname(FLAGS.train)
    g2p_trainer_utils.run(
      problem_name=problem_name,
      model_dir=FLAGS.model_dir,
      data_dir=data_dir,
      model="transformer",
      output_dir=output_dir,
      train_steps=10000,
      eval_steps=10,
      schedule=FLAGS.schedule,
      train_preprocess_file_path=train_preprocess_file_path,
      dev_preprocess_file_path=dev_preprocess_file_path)

  else:
    test_preprocess_file_path = problem.generate_data(FLAGS.decode, FLAGS.model_dir, train_flag=False)
    data_dir = os.path.dirname(FLAGS.decode)
    hparams = trainer_utils.create_hparams(
      FLAGS.hparams_set, data_dir, passed_hparams=FLAGS.hparams)
    g2p_trainer_utils.add_problem_hparams(hparams, problem_name, FLAGS.model_dir)
    estimator, _ = g2p_trainer_utils.create_experiment_components(
      problem_name=problem_name,
      data_dir=data_dir,
      model_name="transformer",
      hparams=hparams,
      run_config=trainer_utils.create_run_config(output_dir),
      model_dir=FLAGS.model_dir,
      dev_preprocess_file_path=test_preprocess_file_path)

    decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
    decode_hp.add_hparam("shards", 1)
    decode_hp.add_hparam("shard_id", FLAGS.worker_id)
    if FLAGS.interactive:
      decoding.decode_interactively(estimator, decode_hp)
    elif FLAGS.decode:
      decoding.decode_from_file(estimator, FLAGS.decode, decode_hp,
                                FLAGS.decode_to_file)
    #else:
    #  decoding.decode_from_dataset(
    #    estimator,
    #    FLAGS.problems.split("-"),
    #    decode_hp,
    #    decode_to_file=FLAGS.decode_to_file,
    #    dataset_split="test" if FLAGS.eval_use_test_set else None)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
