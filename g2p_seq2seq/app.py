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

from g2p_seq2seq.g2p import G2PModel
from g2p_seq2seq.g2p import TrainingParams


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 64, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("model", None, "Training directory.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("interactive", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_string("evaluate", "", "Count word error rate for file.")
tf.app.flags.DEFINE_string("decode", "", "Decode file.")
tf.app.flags.DEFINE_string("output", "", "Decoding result file.")
tf.app.flags.DEFINE_string("train", "", "Train dictionary.")
tf.app.flags.DEFINE_string("valid", "", "Development dictionary.")
tf.app.flags.DEFINE_string("test", "", "Test dictionary.")
tf.app.flags.DEFINE_integer("max_steps", 0,
                            "How many training steps to do until stop training"
                            " (0: no limit).")
tf.app.flags.DEFINE_boolean("reinit", False,
                            "Set to True for training from scratch.")
tf.app.flags.DEFINE_string("optimizer", "sgd", "Optimizer type: sgd, adam, rms-prop. Default: sgd.")

FLAGS = tf.app.flags.FLAGS

def main(_=[]):
  """Main function.
  """
  with tf.Graph().as_default():
    if not FLAGS.model:
      raise RuntimeError("Model directory not specified.")
    g2p_model = G2PModel(FLAGS.model)
    if FLAGS.train:
      g2p_params = TrainingParams(FLAGS)
      g2p_model.prepare_data(FLAGS.train, FLAGS.valid, FLAGS.test)
      if (not os.path.exists(os.path.join(FLAGS.model,
                                          "model.data-00000-of-00001"))
          or FLAGS.reinit):
        g2p_model.create_train_model(g2p_params)
      else:
        g2p_model.load_train_model(g2p_params)
      g2p_model.train()
    else:
      g2p_model.load_decode_model()
      if FLAGS.decode:
        decode_lines = codecs.open(FLAGS.decode, "r", "utf-8").readlines()
        output_file = None
        if FLAGS.output:
          output_file = codecs.open(FLAGS.output, "w", "utf-8")
        g2p_model.decode(decode_lines, output_file)
      elif FLAGS.interactive:
        g2p_model.interactive()
      elif FLAGS.evaluate:
        test_lines = codecs.open(FLAGS.evaluate, "r", "utf-8").readlines()
        g2p_model.evaluate(test_lines)

if __name__ == "__main__":
  tf.app.run()
