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
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import decoding

EOS = text_encoder.EOS_ID

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


class GraphemePhonemeEncoder(text_encoder.TextEncoder):
  """Encodes each grapheme or phoneme to an id. For 8-bit strings only."""

  def __init__(self,
               vocab_filepath=None,
               vocab_list=None,
               separator="",
               num_reserved_ids=text_encoder.NUM_RESERVED_TOKENS):
    """Initialize from a file or list, one token per line.

    Handling of reserved tokens works as follows:
    - When initializing from a list, we add reserved tokens to the vocab.
    - When initializing from a file, we do not add reserved tokens to the vocab.
    - When saving vocab files, we save reserved tokens to the file.

    Args:
      vocab_filename: If not None, the full filename to read vocab from. If this
         is not None, then vocab_list should be None.
      vocab_list: If not None, a list of elements of the vocabulary. If this is
         not None, then vocab_filename should be None.
      num_reserved_ids: Number of IDs to save for reserved tokens like <EOS>.
    """
    super(GraphemePhonemeEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
    if vocab_filepath and os.path.exists(vocab_filepath):
      self._init_vocab_from_file(vocab_filepath)
    else:
      assert vocab_list is not None
      self._init_vocab_from_list(vocab_list)
    self._separator = separator

  def encode(self, symbols_line):
    if isinstance(symbols_line, unicode):
      symbols_line = symbols_line.encode("utf-8")
    if self._separator:
      symbols_list = symbols_line.strip().split(self._separator)
    else:
      symbols_list = list(symbols_line.strip())
    return [self._sym_to_id[sym] for sym in symbols_list]

  def decode(self, ids):
    return " ".join(self.decode_list(ids))

  def decode_list(self, ids):
    return [self._id_to_sym[id_] for id_ in ids]

  @property
  def vocab_size(self):
    return len(self._id_to_sym)

  def _init_vocab_from_file(self, filename):
    """Load vocab from a file.

    Args:
      filename: The file to load vocabulary from.
    """
    def sym_gen():
      with tf.gfile.Open(filename) as f:
        for line in f:
          sym = line.strip()
          yield sym

    self._init_vocab(sym_gen(), add_reserved_symbols=False)

  def _init_vocab_from_list(self, vocab_list):
    """Initialize symbols from a list of symbols.

    It is ok if reserved symbols appear in the vocab list. They will be
    removed. The set of symbols in vocab_list should be unique.

    Args:
      vocab_list: A list of symbols.
    """
    def sym_gen():
      for sym in vocab_list:
        if sym not in text_encoder.RESERVED_TOKENS:
          yield sym

    self._init_vocab(sym_gen())

  def _init_vocab(self, sym_generator, add_reserved_symbols=True):
    """Initialize vocabulary with sym from sym_generator."""

    self._id_to_sym = {}
    non_reserved_start_index = 0

    if add_reserved_symbols:
      self._id_to_sym.update(enumerate(text_encoder.RESERVED_TOKENS))
      non_reserved_start_index = len(text_encoder.RESERVED_TOKENS)

    self._id_to_sym.update(
        enumerate(sym_generator, start=non_reserved_start_index))

    # _sym_to_id is the reverse of _id_to_sym
    self._sym_to_id = dict((v, k)
      for k, v in six.iteritems(self._id_to_sym))

  def store_to_file(self, filename):
    """Write vocab file to disk.

    Vocab files have one symbol per line. The file ends in a newline. Reserved
    symbols are written to the vocab file as well.

    Args:
      filename: Full path of the file to store the vocab to.
    """
    with tf.gfile.Open(filename, "w") as f:
      for i in xrange(len(self._id_to_sym)):
        f.write(self._id_to_sym[i] + "\n")


def tabbed_parsing_character_generator(data_dir, train, model_dir):
  """Generate source and target data from a single file."""
  #character_vocab = text_encoder.ByteTextEncoder()
  filename = "cmudict.dic.{0}".format("train" if train else "dev")
  pair_filepath = os.path.join(data_dir, filename)
  src_vocab_path = os.path.join(model_dir, "vocab.gr")
  tgt_vocab_path = os.path.join(model_dir, "vocab.ph")
  if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
    source_vocab = GraphemePhonemeEncoder(src_vocab_path)
    target_vocab = GraphemePhonemeEncoder(tgt_vocab_path, separator=" ")
    return tabbed_generator(pair_filepath, source_vocab, target_vocab, EOS)
  elif train:
    graphemes, phonemes = {}, {}
    data_filepath = os.path.join(data_dir, "cmudict.dic.train")
    with tf.gfile.GFile(data_filepath, mode="r") as data_file:
      for line in data_file:
        line_split = line.strip().split("\t")
        line_grs, line_phs = list(line_split[0]), line_split[1].split(" ")
        graphemes = update_vocab_symbols(graphemes, line_grs)
        phonemes = update_vocab_symbols(phonemes, line_phs)
    graphemes, phonemes = sorted(graphemes.keys()), sorted(phonemes.keys())
    source_vocab = GraphemePhonemeEncoder(vocab_filepath=src_vocab_path,
      vocab_list=graphemes)
    target_vocab = GraphemePhonemeEncoder(vocab_filepath=tgt_vocab_path,
      vocab_list=phonemes, separator=" ")
    source_vocab.store_to_file(src_vocab_path)
    target_vocab.store_to_file(tgt_vocab_path)
    return tabbed_generator(pair_filepath, source_vocab, target_vocab, EOS)
  else:
    raise IOError("Vocabulary files {} and {} not found.".format(src_vocab_path,
      tgt_vocab_path))


def update_vocab_symbols(init_vocab, update_syms):
  updated_vocab = init_vocab
  for sym in update_syms:
    updated_vocab.update({sym : 1})
  return updated_vocab


def tabbed_generator(source_path, source_vocab, target_vocab, eos=None):
  r"""Generator for sequence-to-sequence tasks using tabbed files.

  Tokens are derived from text files where each line contains both
  a source and a target string. The two strings are separated by a tab
  character ('\t'). It yields dictionaries of "inputs" and "targets" where
  inputs are characters from the source lines converted to integers, and
  targets are characters from the target lines, also converted to integers.

  Args:
    source_path: path to the file with source and target sentences.
    source_vocab: a SubwordTextEncoder to encode the source string.
    target_vocab: a SubwordTextEncoder to encode the target string.
    eos: integer to append at the end of each sequence (default: None).
  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from characters in the file lines.
  """
  eos_list = [] if eos is None else [eos]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    for line in source_file:
      if line and "\t" in line:
        parts = line.split("\t", 1)
        source, target = parts[0].strip(), parts[1].strip()
        source_ints = source_vocab.encode(source) + eos_list
        target_ints = target_vocab.encode(target) + eos_list
        yield {"inputs": source_ints, "targets": target_ints}


@registry.register_problem
class Translate_g2p_cmu_pronalsyl(problem.Text2TextProblem):#translate.TranslateProblem):
  """Problem spec for cmudict PRONALSYL Grapheme-to-Phoneme translation."""

  def __init__(self, model_dir):#, was_reversed, was_copy):
    """Create a Problem.

    Args:
      was_reversed: bool, whether to reverse inputs and targets.
      was_copy: bool, whether to copy inputs to targets. Can be composed with
        was_reversed so that if both are true, the targets become the inputs,
        which are then copied to targets so that the task is targets->targets.
    """
    super(Translate_g2p_cmu_pronalsyl, self).__init__()
    #self._was_reversed = was_reversed
    #self._was_copy = was_copy
    self._encoders = None
    self._hparams = None
    self._feature_info = None
    self._model_dir = model_dir#"/home/nurtas/models/g2p/cmudict-test2"

  def generator(self, data_dir, model_dir, train):
    tag = True if train else False
    return tabbed_parsing_character_generator(data_dir, tag, model_dir)

  @property
  def input_space_id(self):
    return 0

  @property
  def target_space_id(self):
    return 0

  @property
  def num_shards(self):
    return 1

  @property
  def use_subword_tokenizer(self):
    return False

  @property
  def is_character_level(self):
    return False

  def get_feature_encoders(self, data_dir=None):
    if self._encoders is None:
      self._encoders = self.feature_encoders(self._model_dir)
    return self._encoders

  def feature_encoders(self, model_dir):
    tgt_vocab_path = os.path.join(model_dir, "vocab.ph")
    targets_encoder = GraphemePhonemeEncoder(tgt_vocab_path, separator=" ")
    if self.has_inputs:
      src_vocab_path = os.path.join(model_dir, "vocab.gr")
      inputs_encoder = GraphemePhonemeEncoder(src_vocab_path)
      return {"inputs": inputs_encoder, "targets": targets_encoder}
    return {"targets": targets_encoder}


def main(_=[]):
  """Main function.
  """

  FLAGS.hparams_set = "transformer_small"
  FLAGS.schedule = "train_and_evaluate"
  FLAGS.problems = "translate_g2p_cmu_pronalsyl"
  FLAGS.decode_hparams = "beam_size=4,alpha=0.6"
  FLAGS.decode_to_file = "decode_output.txt"
  problem_name = "translate_g2p_cmu_pronalsyl"
  model_dir = "/home/nurtas/models/g2p/cmudict-test2"

  tf.logging.set_verbosity(tf.logging.INFO)
  usr_dir.import_usr_dir(os.path.dirname(os.path.abspath(__file__)))
  problem = registry._PROBLEMS[problem_name](model_dir)#problem("translate_g2p_cmu_pronalsyl")
  #problem = reg_problem("translate_g2p_cmu_pronalsyl")
  #problem.generate_data(os.path.expanduser(FLAGS.model_dir), None, task_id=None)
  trainer_utils.log_registry()
  output_dir = os.path.expanduser(FLAGS.model_dir)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  #data_dir = os.path.expanduser(FLAGS.train)

  if FLAGS.train:
    problem.generate_data(os.path.dirname(os.path.expanduser(FLAGS.train)),
      output_dir, task_id=None)
    data_dir = os.path.expanduser(FLAGS.train)
    trainer_utils.run(
      data_dir=data_dir,
      model="transformer",
      output_dir=output_dir,
      train_steps=10000,
      eval_steps=10,
      schedule=FLAGS.schedule)

  else:
    problem.generate_data(os.path.dirname(os.path.expanduser(FLAGS.decode)), 
      output_dir, task_id=None)
    data_dir = os.path.expanduser(FLAGS.decode)
    hparams = trainer_utils.create_hparams(
      FLAGS.hparams_set, data_dir, passed_hparams=FLAGS.hparams)
    #trainer_utils.add_problem_hparams(hparams, FLAGS.problems)
    g2p_trainer_utils.add_problem_hparams(hparams, problem_name, model_dir)#FLAGS.problems)
    estimator, _ = g2p_trainer_utils.create_experiment_components(
#trainer_utils.create_experiment_components(
      problem_name=problem_name,
      data_dir=data_dir,
      model_name="transformer",
      hparams=hparams,
      run_config=trainer_utils.create_run_config(output_dir),
      model_dir=model_dir)

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
