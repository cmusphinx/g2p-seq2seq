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

from g2p_seq2seq.g2p import G2PModel
from g2p_seq2seq.params import Params

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
tf.flags.DEFINE_integer("eval_steps", 10,
                        "Run evaluation on validation data every N steps.")
tf.flags.DEFINE_string("schedule", "train_and_evaluate", "Set schedule.")
tf.flags.DEFINE_string("master", "", "Address of TensorFlow master.")

FLAGS = tf.app.flags.FLAGS


def main(_=[]):
  """Main function.
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  data_path = FLAGS.train if FLAGS.train else FLAGS.decode
  params = Params(FLAGS.model_dir, data_path, flags=FLAGS)
  g2p_model = G2PModel(params)

  if FLAGS.train:
    g2p_model.prepare_data(train_path=FLAGS.train, dev_path=FLAGS.valid)
    g2p_model.train()

  elif FLAGS.decode:
    g2p_model.prepare_data(test_path=FLAGS.decode)
    g2p_model.decode(decode_from_file=FLAGS.decode, decode_to_file=FLAGS.output)

  elif FLAGS.interactive:
    g2p_model.interactive()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
