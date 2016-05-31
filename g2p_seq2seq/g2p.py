# Copyright 2016 AC Technology LLC. All Rights Reserved.
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

import math
import os
import sys
import time
import codecs

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile

import data_utils as data_utils
from tensorflow.models.rnn.translate import seq2seq_model

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 64, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_string("model", "/tmp", "Training directory.")
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
tf.app.flags.DEFINE_integer("max_steps", 10000,
                            "How many training steps to do until stop training"
                            " (0: no limit).")


FLAGS = tf.app.flags.FLAGS

class G2PModel(seq2seq_model.Seq2SeqModel):
  """Grapheme-to-Phoneme translation model class.

  Constructor parameters (for training mode only):
    train_dic: Train dictionary;
    valid_dic: Development dictionary;
    test_dic: Test dictionary.

  Attributes:
    gr_vocab: Grapheme vocabulary;
    ph_vocab: Phoneme vocabulary;
    train_set: Training buckets: words and sounds are mapped to ids;
    valid_set: Validation buckets: words and sounds are mapped to ids;
    session: Tensorflow session;
    model: Tensorflow Seq2Seq model for G2PModel object.
    train: Train method.
    interactive: Interactive decode method;
    evaluate: Word-Error-Rate counting method;
    decode: Decode file method.
  """
  # We use a number of buckets and pad to the closest one for efficiency.
  # See seq2seq_model.Seq2SeqModel for details of how they work.
  _BUCKETS = [(5, 10), (10, 15), (40, 50)]

  def __init__(self, train_dic=None, valid_dic=None, test_dic=None):
    """Create G2P model and initialize or load parameters in session."""
    #Load model parameters.
    num_layers, size = load_params()
    self.test_dic = test_dic

    # Preliminary actions before model creation.
    if FLAGS.train:
      batch_size = FLAGS.batch_size
      # Prepare G2P data.
      print("Preparing G2P data")
      train_gr_ids, train_ph_ids, valid_gr_ids, valid_ph_ids, self.gr_vocab,\
      self.ph_vocab = data_utils.prepare_g2p_data(FLAGS.model, train_dic,
                                                  valid_dic)
      # Read data into buckets and compute their sizes.
      print ("Reading development and training data.")
      self.valid_set = self.__put_into_buckets(valid_gr_ids, valid_ph_ids)
      self.train_set = self.__put_into_buckets(train_gr_ids, train_ph_ids)
    else:
      batch_size = 1 # We decode one word at a time.
      # Load vocabularies
      self.gr_vocab = data_utils.load_vocabulary(os.path.join(FLAGS.model,
                                                              "vocab.grapheme"))
      self.ph_vocab = data_utils.load_vocabulary(os.path.join(FLAGS.model,
                                                              "vocab.phoneme"))

    self.session = tf.Session()
    decode_flag = False if FLAGS.train else True

    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    self.model = seq2seq_model.Seq2SeqModel(len(self.gr_vocab),
                                            len(self.ph_vocab), self._BUCKETS,
                                            size, num_layers,
                                            FLAGS.max_gradient_norm, batch_size,
                                            FLAGS.learning_rate,
                                            FLAGS.learning_rate_decay_factor,
                                            forward_only=decode_flag)

    self.__run_session()


  def __put_into_buckets(self, source, target):
    """Put data from source and target into buckets.

    Args:
      source: data with ids for the source language.
      target: data with ids for the target language;
        it must be aligned with the source data: n-th line contains the desired
        output for n-th line from the source.

    Returns:
      data_set: a list of length len(_BUCKETS); data_set[n] contains a list of
        (source, target) pairs read from the provided data that fit
        into the n-th bucket, i.e., such that len(source) < _BUCKETS[n][0] and
        len(target) < _BUCKETS[n][1]; source and target are lists of ids.
    """
    data_set = [[] for _ in self._BUCKETS]
    for i in range(len(source)):
      source_ids = source[i]
      target_ids = target[i]
      target_ids.append(data_utils.EOS_ID)
      for bucket_id, (source_size, target_size) in enumerate(self._BUCKETS):
        if len(source_ids) < source_size and len(target_ids) < target_size:
          data_set[bucket_id].append([source_ids, target_ids])
          break
    return data_set


  def __run_session(self):
    """Check for saved models and restore them, otherwise create new model.
    """
    ckpt = tf.train.get_checkpoint_state(FLAGS.model)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
      print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
    elif tf.gfile.Exists(os.path.join(FLAGS.model, "model")):
      self.model.saver.restore(self.session, os.path.join(FLAGS.model, "model"))
    elif FLAGS.train:
      print("Created model with fresh parameters.")
      self.session.run(tf.initialize_all_variables())
    else:
      raise ValueError("Model not found in %s" % ckpt.model_checkpoint_path)


  def train(self):
    """Train a gr->ph translation model using G2P data."""
    train_bucket_sizes = [len(self.train_set[b])
                          for b in xrange(len(self._BUCKETS))]
    train_total_size = float(sum(train_bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while (FLAGS.max_steps == 0
           or self.model.global_step.eval(self.session) <= FLAGS.max_steps):
      # Get a batch and make a step.
      start_time = time.time()
      step_loss = self.__calc_step_loss(train_buckets_scale)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (self.model.global_step.eval(self.session),
                         self.model.learning_rate.eval(self.session), step_time,
                         perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          self.session.run(self.model.learning_rate_decay_op)
        if len(previous_losses) > 34 and \
        previous_losses[-35:-34] <= min(previous_losses[-35:]):
          break
        previous_losses.append(loss)
        step_time, loss = 0.0, 0.0
        self.__save_model_run_evals()
    print('Training process stopped.')

    print('Beginning calculation word error rate (WER) on test sample.')
    self.model.forward_only = True
    self.model.batch_size = 1  # We decode one word at a time.
    self.evaluate(self.test_dic)


  def __calc_step_loss(self, train_buckets_scale):
    """Choose a bucket according to data distribution. We pick a random number
    in [0, 1] and use the corresponding interval in train_buckets_scale.
    """
    random_number_01 = np.random.random_sample()
    bucket_id = min([i for i in xrange(len(train_buckets_scale))
                     if train_buckets_scale[i] > random_number_01])

    # Get a batch and make a step.
    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
        self.train_set, bucket_id)
    _, step_loss, _ = self.model.step(self.session, encoder_inputs,
                                      decoder_inputs, target_weights,
                                      bucket_id, False)
    return step_loss


  def __save_model_run_evals(self):
    """Save model and then run evaluation on validation set.
    """
    # Save checkpoint and zero timer and loss.
    checkpoint_path = os.path.join(FLAGS.model, "translate.ckpt")
    self.model.saver.save(self.session, checkpoint_path,
                          global_step=self.model.global_step)
    # Run evals on development set and print their perplexity.
    for bucket_id in xrange(len(self._BUCKETS)):
      encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
          self.valid_set, bucket_id)
      _, eval_loss, _ = self.model.step(self.session, encoder_inputs,
                                        decoder_inputs, target_weights,
                                        bucket_id, True)
      eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
      print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))


  def __decode_word(self, word, phonetics=None):
    """Decode input word to sequence of phonemes.

    Args:
      word: input word;
      phonetics: target phoneme sequence. This argument is used only with model
                 created in train mode

    Returns:
      res_phoneme_seq: decoded phoneme sequence for input word;
    """
    ph_ids = []
    if phonetics:
      ph_ids = data_utils.symbols_to_ids(phonetics, self.ph_vocab)

    res_phoneme_seq = ""
    # Check if all graphemes attended in vocabulary
    gr_absent = set(gr for gr in word if gr not in self.gr_vocab)
    if not gr_absent:
      # Get token-ids for the input word.
      token_ids = [self.gr_vocab.get(s, data_utils.UNK_ID) for s in word]
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(self._BUCKETS))
                       if self._BUCKETS[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the word to the model.
      encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
          {bucket_id: [(token_ids, ph_ids)]}, bucket_id)
      # Get output logits for the word.
      _, _, output_logits = self.model.step(self.session, encoder_inputs,
                                            decoder_inputs, target_weights,
                                            bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Phoneme sequence corresponding to outputs.
      rev_ph_vocab = reverse_vocab(self.ph_vocab)
      res_phoneme_seq = " ".join([rev_ph_vocab[output] for output in outputs])
    else:
      print("Symbols '%s' are not in vocabulary" % "','".join(gr_absent))
    return res_phoneme_seq


  def interactive(self):
    """Decode word from standard input.
    """
    while True:
      print("> ", end="")
      word = sys.stdin.readline().decode("utf-8").strip()
      if word:
        res_phoneme_seq = self.__decode_word(word)
        if res_phoneme_seq:
          print(res_phoneme_seq)
      else: break


  def __calc_error(self, w_ph_dict):
    """Calculate number of wrong predicted words.
    """
    errors = 0
    for word, speech in w_ph_dict.items():
      #if len(phonetics) == 1:
      phonetics = []
      if FLAGS.train:
        phonetics = speech[0].split()

      model_assumption = self.__decode_word(word, phonetics)
      if model_assumption not in speech:
        errors += 1
    return errors


  def evaluate(self, test_dic=None):
    """Calculate and print out word error rate (WER) and Accuracy
       on test sample.

    Args:
      No arguments need if this function called not from active Session
      in tensorflow. Otherwise test dictionary should be pointed out:
      test_dic: List of test dictionary. Each element of list must be String
                containing word and its pronounciation (e.g., "word W ER D");
    """
    if not test_dic:
      # Decode from input file.
      test_dic = codecs.open(FLAGS.evaluate, "r", "utf-8").readlines()

    w_ph_dict = {}
    for line in test_dic:
      lst = line.strip().split()
      if len(lst) >= 2:
        if lst[0] not in w_ph_dict:
          w_ph_dict[lst[0]] = [" ".join(lst[1:])]
        else:
          w_ph_dict[lst[0]].append(" ".join(lst[1:]))

    errors = self.__calc_error(w_ph_dict)
    print("WER : ", errors/len(w_ph_dict))
    print("Accuracy : ", (1-(errors/len(w_ph_dict))))


  def decode(self, word_list_file_path):
    """Decode words from file.

    Args:
      word_list_file_path: path to input file. File must be
                           in one-word-per-line format.

    Returns:
      if [--output output_file] pointed out, write decoded word sequences in
      this file. Otherwise, print decoded words in standard output.
    """
    # Decode from input file.
    graphemes = codecs.open(word_list_file_path, "r", "utf-8").readlines()

    output_file_path = FLAGS.output

    if output_file_path:
      with codecs.open(output_file_path, "w", "utf-8") as output_file:
        for word in graphemes:
          word = word.strip()
          res_phoneme_seq = self.__decode_word(word)
          output_file.write(word)
          output_file.write(' ')
          output_file.write(res_phoneme_seq)
          output_file.write('\n')
    else:
      for word in graphemes:
        word = word.strip()
        res_phoneme_seq = self.__decode_word(word)
        print(word + ' ' + res_phoneme_seq)


def load_params():
  """On train mode save model parameters.
  On decode mode load parameters from 'model.params' file, or if file is absent,
  use Default parameters.

  Returns:
    num_layers: Number of layers in the model;
    size: Size of each model layer.
  """
  num_layers = FLAGS.num_layers
  size = FLAGS.size

  if FLAGS.train:
    if not os.path.exists(FLAGS.model):
      os.makedirs(FLAGS.model)
    # Save model's architecture
    with open(os.path.join(FLAGS.model, "model.params"), 'w') as param_file:
      param_file.write("num_layers:" + str(FLAGS.num_layers) + "\n")
      param_file.write("size:" + str(FLAGS.size))
  else:
    # Checking model's architecture for decode processes.
    if gfile.Exists(os.path.join(FLAGS.model, "model.params")):
      params = open(os.path.join(FLAGS.model, "model.params")).readlines()
      for line in params:
        split_line = line.strip().split(":")
        if split_line[0] == "num_layers":
          num_layers = int(split_line[1])
        if split_line[0] == "size":
          size = int(split_line[1])
  return num_layers, size


def reverse_vocab(vocab):
  """Reverse mapping of input vocabulary: from dictionary with string keys and
     integer values ({"d": 0, "c": 1}) to reverse vocabulary list (a list
     where symbol's order correspond to its id) and vice-versa.
  """
  if isinstance(vocab, dict):
    rev_vocab = [data_utils.UNK_ID]*len(vocab)
    for symbol, ids in vocab.items():
      rev_vocab[ids] = symbol
  elif isinstance(vocab, list):
    rev_vocab = dict([(x, y) for (y, x) in enumerate(vocab)])
  else:
    raise ValueError("Dictionary type neither dict nor list.")
  return rev_vocab


def main(_):
  """Main function.
  """
  if FLAGS.decode:
    g2p_model = G2PModel()
    g2p_model.decode(FLAGS.decode)
  elif FLAGS.interactive:
    g2p_model = G2PModel()
    g2p_model.interactive()
  elif FLAGS.evaluate:
    g2p_model = G2PModel()
    g2p_model.evaluate()
  else:
    if FLAGS.train:
      source_dic = codecs.open(FLAGS.train, "r", "utf-8").readlines()
      train_dic, valid_dic, test_dic = [], [], []
      if (not FLAGS.valid) and (not FLAGS.test):
        for i, word in enumerate(source_dic):
          if i % 20 == 0 or i % 20 == 1:
            test_dic.append(word)
          elif i % 20 == 2:
            valid_dic.append(word)
          else: train_dic.append(word)
      elif not FLAGS.valid:
        test_dic = codecs.open(FLAGS.test, "r", "utf-8").readlines()
        for i, word in enumerate(source_dic):
          if i % 20 == 0:
            valid_dic.append(word)
          else: train_dic.append(word)
      elif not FLAGS.test:
        valid_dic = codecs.open(FLAGS.valid, "r", "utf-8").readlines()
        for i, word in enumerate(source_dic):
          if i % 10 == 0:
            test_dic.append(word)
          else: train_dic.append(word)
      else:
        valid_dic = codecs.open(FLAGS.valid, "r", "utf-8").readlines()
        test_dic = codecs.open(FLAGS.test, "r", "utf-8").readlines()
        train_dic = source_dic
    else:
      raise ValueError("Train dictionary absent.")
    g2p_model = G2PModel(train_dic, valid_dic, test_dic)
    g2p_model.train()

if __name__ == "__main__":
  tf.app.run()
