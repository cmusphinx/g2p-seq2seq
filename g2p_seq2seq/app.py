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

import g2p_seq2seq.g2p_trainer_utils as g2p_trainer_utils
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
tf.flags.DEFINE_boolean("freeze", False,
                        "Set to True for freeze the graph.")
tf.flags.DEFINE_boolean("p2g", False,
    "Set to True for switching to the phoneme-to-grapheme mode.")

# Training parameters
tf.flags.DEFINE_string("hparams", "",
    "Customize hyper parameters for transformer model.")
tf.flags.DEFINE_integer("batch_size", 4096,
                        "Batch size to use during training.")
tf.flags.DEFINE_integer("min_length_bucket", 6,
                        "Set the size of the minimal bucket.")
tf.flags.DEFINE_integer("max_length", 30,
                        "Set the size of the maximal bucket.")
tf.flags.DEFINE_float("length_bucket_step", 1.5,
    """This flag controls the number of length buckets in the data reader.
    The buckets have maximum lengths from min_bucket_length to max_length,
    increasing (approximately) by factors
    of length_bucket_step.""")
tf.flags.DEFINE_integer("num_layers", 3, "Number of hidden layers.")
tf.flags.DEFINE_integer("size", 256,
                        "The number of neurons in the hidden layer.")
tf.flags.DEFINE_integer("filter_size", 512,
                        "The size of the filter in a convolutional layer.")
tf.flags.DEFINE_integer("num_heads", 4,
                        "Number of applied heads in Multi-attention mechanism.")
tf.flags.DEFINE_integer("max_epochs", 0,
                        "How many epochs to train the model."
                        " (0: no limit).")
tf.flags.DEFINE_boolean("cleanup", False,
                        "Set to True for cleanup dictionary from stress and "
                        "comments (after hash or inside braces).")

# Decoding parameters
tf.flags.DEFINE_boolean("return_beams", False,
                        "Set to true for beams decoding.")
tf.flags.DEFINE_integer("beam_size", 1, "Number of decoding beams.")
tf.flags.DEFINE_float("alpha", 0.6,
    """Float that controls the length penalty. Larger the alpha, stronger the
    preference for longer sequences.""")

FLAGS = tf.app.flags.FLAGS


def main(_=[]):
  """Main function.
  """
  tf.logging.set_verbosity(tf.logging.INFO)
  file_path = FLAGS.train or FLAGS.decode or FLAGS.evaluate
  test_path = FLAGS.decode or FLAGS.evaluate or FLAGS.test

  if not FLAGS.model_dir:
    raise RuntimeError("Model directory not specified.")

  if FLAGS.reinit and os.path.exists(FLAGS.model_dir):
    shutil.rmtree(FLAGS.model_dir)

  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)

  params = Params(FLAGS.model_dir, file_path, flags=FLAGS)

  if FLAGS.train:
    g2p_trainer_utils.save_params(FLAGS.model_dir, params.hparams)
    g2p_model = G2PModel(params, train_path=FLAGS.train, dev_path=FLAGS.valid,
                         test_path=test_path, cleanup=FLAGS.cleanup,
                         p2g_mode=FLAGS.p2g)
    g2p_model.train()

  else:
    params.hparams = g2p_trainer_utils.load_params(FLAGS.model_dir)
    g2p_model = G2PModel(params, test_path=test_path, p2g_mode=FLAGS.p2g)

    if FLAGS.freeze:
      g2p_model.freeze()

    elif FLAGS.interactive:
      g2p_model.interactive()

    elif FLAGS.decode:
      g2p_model.decode(output_file_path=FLAGS.output)

    elif FLAGS.evaluate:
      g2p_model.evaluate()


if __name__ == "__main__":
  tf.app.run()
