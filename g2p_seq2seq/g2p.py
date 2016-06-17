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

import math
import os
import sys
import time
import codecs

import numpy as np

import tensorflow as tf

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
tf.app.flags.DEFINE_integer("max_steps", 0,
                            "How many training steps to do until stop training"
                            " (0: no limit).")


FLAGS = tf.app.flags.FLAGS

class G2PModel():
  """Grapheme-to-Phoneme translation model class.

  Constructor parameters (for training mode only):
    train_file: Train dictionary;
    valid_file: Development dictionary;
    test_file: Test dictionary.

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

  def __init__(self, model_dir):
    """Create G2P model and initialize or load parameters in session."""
    self.model_dir = model_dir

    # Preliminary actions before model creation.
    if not os.path.exists(os.path.join(self.model_dir, "model")):
      return

    #Load model parameters.
    num_layers, size = data_utils.load_params(self.model_dir)
    batch_size = 1 # We decode one word at a time.
    # Load vocabularies
    self.gr_vocab = data_utils.load_vocabulary(os.path.join(self.model_dir,
                                                            "vocab.grapheme"))
    self.ph_vocab = data_utils.load_vocabulary(os.path.join(self.model_dir,
                                                            "vocab.phoneme"))

    self.rev_ph_vocab =\
      data_utils.load_vocabulary(os.path.join(self.model_dir, "vocab.phoneme"),
                                 reverse=True)

    self.session = tf.Session()

    # Create model.
    print("Creating %d layers of %d units." % (num_layers, size))
    self.model = seq2seq_model.Seq2SeqModel(len(self.gr_vocab),
                                            len(self.ph_vocab), self._BUCKETS,
                                            size, num_layers, 0, batch_size,
                                            0, 0, forward_only=True)
    self.model.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
    # Check for saved models and restore them.
    print("Reading model parameters from %s" % self.model_dir)
    self.model.saver.restore(self.session, os.path.join(self.model_dir,
                                                        "model"))


  def __train_init(self, params, train_path, valid_path=None, test_path=None):
    """Create G2P model and initialize or load parameters in session."""

    # Preliminary actions before model creation.
    # Load model parameters.
    num_layers, size = data_utils.save_params(params.num_layers,
                                              params.size,
                                              self.model_dir)
    # Prepare G2P data.
    print("Preparing G2P data")
    train_gr_ids, train_ph_ids, valid_gr_ids, valid_ph_ids, self.gr_vocab,\
    self.ph_vocab, self.test_file =\
    data_utils.prepare_g2p_data(self.model_dir, train_path, valid_path,
                                test_path)
    # Read data into buckets and compute their sizes.
    print ("Reading development and training data.")
    self.valid_set = self.__put_into_buckets(valid_gr_ids, valid_ph_ids)
    self.train_set = self.__put_into_buckets(train_gr_ids, train_ph_ids)

    self.rev_ph_vocab =\
        data_utils.load_vocabulary(os.path.join(self.model_dir,
                                                "vocab.phoneme"),
                                   reverse=True)

    self.session = tf.Session()

    # Create model.
    print("Creating %d layers of %d units." % (num_layers, size))
    self.model = seq2seq_model.Seq2SeqModel(len(self.gr_vocab),
                                            len(self.ph_vocab), self._BUCKETS,
                                            size, num_layers,
                                            params.max_gradient_norm,
                                            params.batch_size,
                                            params.learning_rate,
                                            params.lr_decay_factor,
                                            forward_only=False)
    self.model.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
    print("Created model with fresh parameters.")
    self.session.run(tf.initialize_all_variables())


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


  def train(self, params, train_path, valid_path, test_path):
    """Train a gr->ph translation model using G2P data."""
    if os.path.exists(os.path.join(self.model_dir, "model")):
      raise StandardError("Model already exists in %s" % self.model_dir)
    else:
      self.__train_init(params, train_path, valid_path, test_path)

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
    while (params.max_steps == 0
           or self.model.global_step.eval(self.session) <= params.max_steps):
      # Get a batch and make a step.
      start_time = time.time()
      step_loss = self.__calc_step_loss(train_buckets_scale)
      step_time += (time.time() - start_time) / params.steps_per_checkpoint
      loss += step_loss / params.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % params.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (self.model.global_step.eval(self.session),
                         self.model.learning_rate.eval(self.session),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          self.session.run(self.model.learning_rate_decay_op)
        if len(previous_losses) > 34 and \
        previous_losses[-35:-34] <= min(previous_losses[-35:]):
          break
        previous_losses.append(loss)
        step_time, loss = 0.0, 0.0
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(self.model_dir, "model")
        self.model.saver.save(self.session, checkpoint_path,
                              write_meta_graph=False)
        self.__run_evals()
    print('Training process stopped.')

    print('Beginning calculation word error rate (WER) on test sample.')
    self.model.forward_only = True
    self.model.batch_size = 1  # We decode one word at a time.
    self.evaluate(self.test_file)


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


  def __run_evals(self):
    """Run evals on development set and print their perplexity.
    """
    for bucket_id in xrange(len(self._BUCKETS)):
      encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
          self.valid_set, bucket_id)
      _, eval_loss, _ = self.model.step(self.session, encoder_inputs,
                                        decoder_inputs, target_weights,
                                        bucket_id, True)
      eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
      print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))


  def decode_word(self, word, phonetics=None):
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
      res_phoneme_seq = " ".join([self.rev_ph_vocab[output]
                                  for output in outputs])
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
        res_phoneme_seq = self.decode_word(word)
        if res_phoneme_seq:
          print(res_phoneme_seq)
      else: break


  def __calc_error(self, dictionary):
    """Calculate a number of prediction errors.
    """
    errors = 0
    for word, pronunciations in dictionary.items():
      phonemes = []
      if not FLAGS.evaluate:
        phonemes = pronunciations[0].split()

      hyp = self.decode_word(word, phonemes)
      if hyp not in pronunciations:
        errors += 1
    return errors


  def evaluate(self, test_file=None):
    """Calculate and print out word error rate (WER) and Accuracy
       on test sample.

    Args:
      No arguments need if this function called not from active Session
      in tensorflow. Otherwise test dictionary should be pointed out:
      test_file: List of test dictionary. Each element of list must be String
                containing word and its pronounciation (e.g., "word W ER D");
    """
    if not test_file:
      # Decode from input file.
      test_file = codecs.open(FLAGS.evaluate, "r", "utf-8").readlines()

    test_dic = data_utils.collect_pronunciations(test_file)
    print('Beginning calculation word error rate (WER) on test sample.')
    errors = self.__calc_error(test_dic)
    print("WER : ", errors/len(test_dic))
    print("Accuracy : ", (1-(errors/len(test_dic))))


  def decode(self):
    """Decode words from file.

    Returns:
      if [--output output_file] pointed out, write decoded word sequences in
      this file. Otherwise, print decoded words in standard output.
    """
    # Decode from input file.
    graphemes = codecs.open(FLAGS.decode, "r", "utf-8").readlines()

    output_file_path = FLAGS.output

    if output_file_path:
      with codecs.open(output_file_path, "w", "utf-8") as output_file:
        print('Beginning decoding words from %s to %s'
              % (FLAGS.decode, FLAGS.output))
        for word in graphemes:
          word = word.strip()
          res_phoneme_seq = self.decode_word(word)
          output_file.write(word)
          output_file.write(' ')
          output_file.write(res_phoneme_seq)
          output_file.write('\n')
    else:
      print('Beginning decoding words from %s' % FLAGS.decode)
      for word in graphemes:
        word = word.strip()
        res_phoneme_seq = self.decode_word(word)
        print(word + ' ' + res_phoneme_seq)


class G2P_Params():
  """Class with training parameters."""
  def __init__(self):
    self.learning_rate = FLAGS.learning_rate
    self.lr_decay_factor = FLAGS.learning_rate_decay_factor
    self.max_gradient_norm = FLAGS.max_gradient_norm
    self.batch_size = FLAGS.batch_size
    self.size = FLAGS.size
    self.num_layers = FLAGS.num_layers
    self.steps_per_checkpoint = FLAGS.steps_per_checkpoint
    self.max_steps = FLAGS.max_steps


def main(_):
  """Main function.
  """
  if FLAGS.train:
    g2p_model = G2PModel(FLAGS.model)
    g2p_params = G2P_Params()
    g2p_model.train(g2p_params, FLAGS.train, FLAGS.valid, FLAGS.test)
  else:
    g2p_model = G2PModel(FLAGS.model)
    if not hasattr(g2p_model, "model"):
      raise StandardError("Model not found in %s" % g2p_model.model_dir)
    if FLAGS.decode:
      g2p_model.decode()
    elif FLAGS.interactive:
      g2p_model.interactive()
    elif FLAGS.evaluate:
      g2p_model.evaluate()

if __name__ == "__main__":
  tf.app.run()
