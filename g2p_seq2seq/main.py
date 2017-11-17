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

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import decoding

from g2p_seq2seq import data_utils
from g2p_seq2seq.g2p_t2t import G2PModel
from g2p_seq2seq.params import Params

EOS = text_encoder.EOS_ID

#import yaml
#from six import string_types

#from seq2seq import tasks, models
#from seq2seq.configurable import _maybe_load_yaml, _deep_merge_dict
#from seq2seq.data import input_pipeline
#from seq2seq.inference import create_inference_graph
#from seq2seq.training import utils as training_utils

from IPython.core.debugger import Tracer

tf.flags.DEFINE_string("model_dir", None, "Training directory.")
tf.flags.DEFINE_string("t2t_usr_dir", "/home/nurtas/projects/g2p-seq2seq/g2p_seq2seq", "Data directory.")
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
tf.flags.DEFINE_string("hooks", "",
                       """YAML configuration string for the
                       training hooks to use.""")
tf.flags.DEFINE_string("model_params", "",
                       """YAML configuration string for the model
                       parameters.""")
tf.flags.DEFINE_string("master", "", "Address of TensorFlow master.")
tf.flags.DEFINE_string("schedule", "train_and_evaluate", "Set schedule.")
#tf.flags.DEFINE_string("metrics", "",
#                       """YAML configuration string for the
#                       training metrics to use.""")
#tf.flags.DEFINE_string("input_pipeline", None,
#                       """Defines how input data should be loaded.
#                       A YAML string.""")
# RunConfig Flags
#tf.flags.DEFINE_integer("save_checkpoints_secs", None,
#                        """Save checkpoints every this many seconds.
#                        Can not be specified with save_checkpoints_steps.""")
#tf.flags.DEFINE_integer("save_checkpoints_steps", None,
#                        """Save checkpoints every this many steps.
#                        Can not be specified with save_checkpoints_secs.""")

FLAGS = tf.app.flags.FLAGS


def tabbed_parsing_character_generator(data_dir, train):
  """Generate source and target data from a single file."""
  character_vocab = text_encoder.ByteTextEncoder()
  filename = "cmudict.dic.{0}".format("train" if train else "dev")
  pair_filepath = os.path.join(data_dir, filename)
  return translate.tabbed_generator(pair_filepath, character_vocab,
                                    character_vocab, EOS)


@registry.register_problem
class Translate_g2p_cmu_pronalsyl(translate.TranslateProblem):
  """Problem spec for cmudict PRONALSYL Grapheme-to-Phoneme translation."""

  @property
  def targeted_vocab_size(self):
    return 39

  @property
  def vocab_name(self):
    return "vocab.grph"

  def generator(self, data_dir, tmp_dir, train):
    tag = True if train else False
    return tabbed_parsing_character_generator(data_dir, tag)

  @property
  def input_space_id(self):
    return 0#problem.SpaceID.EN_TOK

  @property
  def target_space_id(self):
    return 0#problem.SpaceID.DE_TOK

  @property
  def num_shards(self):
    return 1

  @property
  def use_subword_tokenizer(self):
    return False

  @property
  def is_character_level(self):
    return True


def main(_=[]):
  """Main function.
  """

  #if FLAGS.save_checkpoints_secs is None \
  #  and FLAGS.save_checkpoints_steps is None:
  #  FLAGS.save_checkpoints_secs = 600
  #  tf.logging.info("Setting save_checkpoints_secs to %d",
  #                  FLAGS.save_checkpoints_secs)
  FLAGS.hparams_set = "transformer_small"
  FLAGS.schedule = "train_and_evaluate"
  FLAGS.problems = "translate_g2p_cmu_pronalsyl"
  FLAGS.decode_hparams = "beam_size=4,alpha=0.6"
  FLAGS.decode_to_file = "decode_output.txt"

  problem = registry.problem("translate_g2p_cmu_pronalsyl")
  #problem = Translate_g2p_cmu_pronalsyl()
  #task_id = None if FLAGS.task_id < 0 else FLAGS.task_id
  #Tracer()()
  problem.generate_data(os.path.expanduser(FLAGS.model_dir), None,
    task_id=None)#task_id)
  usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
  trainer_utils.log_registry()
  #trainer_utils.validate_flags()
  output_dir = os.path.expanduser(FLAGS.model_dir)
  data_dir = os.path.expanduser(FLAGS.train)

  if FLAGS.train:
    trainer_utils.run(
      data_dir=data_dir,
      model="transformer",#FLAGS.model_dir,
      output_dir=output_dir,
      train_steps=10000,
      eval_steps=10,
      schedule=FLAGS.schedule)#"train_and_evaluate")

  else:
    hparams = trainer_utils.create_hparams(
      FLAGS.hparams_set, data_dir, passed_hparams=FLAGS.hparams)
    trainer_utils.add_problem_hparams(hparams, FLAGS.problems)
    estimator, _ = trainer_utils.create_experiment_components(
      data_dir=data_dir,
      model_name="transformer",#FLAGS.model,
      hparams=hparams,
      run_config=trainer_utils.create_run_config(output_dir))

    decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
    decode_hp.add_hparam("shards", 1)#FLAGS.decode_shards)
    decode_hp.add_hparam("shard_id", FLAGS.worker_id)
    if FLAGS.interactive:#decode_interactive:
      decoding.decode_interactively(estimator, decode_hp)
    elif FLAGS.decode:#decode_from_file:
      decoding.decode_from_file(estimator, FLAGS.decode, decode_hp,
                                FLAGS.decode_to_file)
    #else:
    #  decoding.decode_from_dataset(
    #    estimator,
    #    FLAGS.problems.split("-"),
    #    decode_hp,
    #    decode_to_file=FLAGS.decode_to_file,
    #    dataset_split="test" if FLAGS.eval_use_test_set else None)

  #with tf.Graph().as_default():
  #  if not FLAGS.model:
  #    raise RuntimeError("Model directory not specified.")
  #  g2p_model = G2PModel(FLAGS.model)
  #  if FLAGS.train:
  #    data_utils.create_vocabulary(FLAGS.train, FLAGS.model)
  #    g2p_params = Params(FLAGS.model, decode_flag=False, flags=FLAGS)
  #    g2p_model.load_train_model(g2p_params)
  #    g2p_model.train()
  #  else:
  #    vocab_source_path, vocab_target_path =\
  #      os.path.join(FLAGS.model, "vocab.grapheme"),\
  #      os.path.join(FLAGS.model, "vocab.phoneme")
  #    if not (os.path.exists(vocab_source_path)
  #            and os.path.exists(vocab_target_path)):
  #      raise StandardError("Vocabularies: %s, %s not found."
  #                          % (vocab_source_path, vocab_target_path))
  #    g2p_params = Params(FLAGS.model, decode_flag=True, flags=FLAGS)
  #    g2p_model.load_decode_model(g2p_params)
  #    if FLAGS.decode:
  #      output_file = None
  #      if FLAGS.output:
  #        output_file = codecs.open(FLAGS.output, "w", "utf-8")
  #      g2p_model.decode(output_file)

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
