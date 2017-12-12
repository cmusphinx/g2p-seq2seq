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
import shutil
import tensorflow as tf

from g2p_seq2seq.g2p import G2PModel
from g2p_seq2seq.params import Params


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
tf.flags.DEFINE_integer("batch_size", 256,
                        "Batch size to use during training.")
tf.flags.DEFINE_integer("min_length_bucket", 5,
                        "Set the size of the minimal bucket.")
tf.flags.DEFINE_integer("max_length", 40,
                        "Set the size of the maximal bucket.")
tf.flags.DEFINE_integer("length_bucket_step", 2.0,
    """This flag controls the number of length buckets in the data reader.
    The buckets have maximum lengths from min_bucket_length to max_length,
    increasing (approximately) by factors
    of length_bucket_step.""")
tf.flags.DEFINE_integer("num_layers", 2, "Number of hidden layers.")
tf.flags.DEFINE_integer("size", 64,
                        "The number of neurons in the hidden layer.")
tf.flags.DEFINE_integer("filter_size", 256,
                        "The size of the filter in a convolutional layer.")
tf.flags.DEFINE_integer("dropout", 0.5, "The proportion of dropping out units"
                        "in hidden layers.")
tf.flags.DEFINE_integer("attention_dropout", 0.5,
                        "The proportion of dropping out units"
                        "in an attention layer.")
tf.flags.DEFINE_integer("num_heads", 2,
                        "Number of applied heads in Multi-attention mechanism.")
tf.flags.DEFINE_integer("max_epochs", 0,
                        "How many training steps to do until stop training"
                        " (0: no limit).")
tf.flags.DEFINE_integer("eval_steps", 10, "Number of steps for evaluation.")
tf.flags.DEFINE_string("schedule", "train_and_evaluate",
    """Set schedule. More info about training configurations you can read in
    tensor2tensor docs: https://github.com/tensorflow/tensor2tensor/blob/master/
docs/distributed_training.md""")
tf.flags.DEFINE_string("master", "",
                       "TensorFlow master. Defaults to empty string for local."
                       "Specifies the configurations for distributed run.")

FLAGS = tf.app.flags.FLAGS


def main(_=[]):
  """Main function.
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  data_path = FLAGS.train if FLAGS.train else FLAGS.decode
  data_path = FLAGS.evaluate if not data_path else data_path

  if not FLAGS.model_dir:
    raise RuntimeError("Model directory not specified.")

  if FLAGS.reinit and os.path.exists(FLAGS.model_dir):
    shutil.rmtree(FLAGS.model_dir)

  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

  params = Params(FLAGS.model_dir, data_path, flags=FLAGS)

  if FLAGS.train:
    g2p_model = G2PModel(params, file_path=FLAGS.train, is_training=True)
    g2p_model.prepare_data(train_path=FLAGS.train, dev_path=FLAGS.valid)
    g2p_model.train()

  elif FLAGS.interactive:
    g2p_model = G2PModel(params)
    g2p_model.interactive()

  elif FLAGS.decode:
    g2p_model = G2PModel(params, file_path=FLAGS.decode, is_training=False)
    g2p_model.decode(output_file_path=FLAGS.output)

  elif FLAGS.evaluate:
    g2p_model = G2PModel(params, file_path=FLAGS.evaluate, is_training=False)
    g2p_model.evaluate()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
