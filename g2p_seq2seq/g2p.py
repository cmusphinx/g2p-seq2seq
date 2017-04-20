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

"""Main class for g2p.

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
import time
import random

import numpy as np
import tensorflow as tf

from g2p_seq2seq import data_utils
from g2p_seq2seq import seq2seq_model

from six.moves import xrange, input  # pylint: disable=redefined-builtin
from six import text_type

class G2PModel(object):
  """Grapheme-to-Phoneme translation model class.

  Constructor parameters (for training mode only):
    train_lines: Train dictionary;
    valid_lines: Development dictionary;
    test_lines: Test dictionary.

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
    """Initialize model directory."""
    self.model_dir = model_dir

  def load_decode_model(self):
    """Load G2P model and initialize or load parameters in session."""
    if not os.path.exists(os.path.join(self.model_dir, 'checkpoint')):
      raise RuntimeError("Model not found in %s" % self.model_dir)

    self.batch_size = 1 # We decode one word at a time.
    #Load model parameters.
    num_layers, size = data_utils.load_params(self.model_dir)
    # Load vocabularies
    print("Loading vocabularies from %s" % self.model_dir)
    self.gr_vocab = data_utils.load_vocabulary(os.path.join(self.model_dir,
                                                            "vocab.grapheme"))
    self.ph_vocab = data_utils.load_vocabulary(os.path.join(self.model_dir,
                                                            "vocab.phoneme"))

    self.rev_ph_vocab =\
      data_utils.load_vocabulary(os.path.join(self.model_dir, "vocab.phoneme"),
                                 reverse=True)

    self.session = tf.Session()

    # Restore model.
    print("Creating %d layers of %d units." % (num_layers, size))
    self.model = seq2seq_model.Seq2SeqModel(len(self.gr_vocab),
                                            len(self.ph_vocab), self._BUCKETS,
                                            size, num_layers, 0,
                                            self.batch_size, 0, 0,
                                            forward_only=True)
    self.model.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    # Check for saved models and restore them.
    print("Reading model parameters from %s" % self.model_dir)
    self.model.saver.restore(self.session, os.path.join(self.model_dir,
                                                        "model"))


  def __put_into_buckets(self, source, target):
    """Put data from source and target into buckets.

    Args:
      source: data with ids for graphemes;
      target: data with ids for phonemes;
        it must be aligned with the source data: n-th line contains the desired
        output for n-th line from the source.

    Returns:
      data_set: a list of length len(_BUCKETS); data_set[n] contains a list of
        (source, target) pairs read from the provided data that fit
        into the n-th bucket, i.e., such that len(source) < _BUCKETS[n][0] and
        len(target) < _BUCKETS[n][1]; source and target are lists of ids.
    """

    # By default unk to unk
    data_set = [[[[4], [4]]] for _ in self._BUCKETS]

    for source_ids, target_ids in zip(source, target):
      target_ids.append(data_utils.EOS_ID)
      for bucket_id, (source_size, target_size) in enumerate(self._BUCKETS):
        if len(source_ids) < source_size and len(target_ids) < target_size:
          data_set[bucket_id].append([source_ids, target_ids])
          break

    for bucket_id in range(len(self._BUCKETS)):
      random.shuffle(data_set[bucket_id])
    return data_set


  def prepare_data(self, train_path, valid_path, test_path):
    """Prepare train/validation/test sets. Create or load vocabularies."""
    # Prepare data.
    print("Preparing G2P data")
    train_gr_ids, train_ph_ids, valid_gr_ids, valid_ph_ids, self.gr_vocab,\
    self.ph_vocab, self.test_lines =\
      data_utils.prepare_g2p_data(self.model_dir, train_path, valid_path,
                                  test_path)
    # Read data into buckets and compute their sizes.
    print ("Reading development and training data.")
    self.valid_set = self.__put_into_buckets(valid_gr_ids, valid_ph_ids)
    self.train_set = self.__put_into_buckets(train_gr_ids, train_ph_ids)

    self.rev_ph_vocab = dict([(x, y) for (y, x) in enumerate(self.ph_vocab)])


  def __prepare_model(self, params):
    """Prepare G2P model for training."""

    self.params = params

    self.session = tf.Session()

    # Prepare model.
    print("Creating model with parameters:")
    print(params)
    self.model = seq2seq_model.Seq2SeqModel(len(self.gr_vocab),
                                            len(self.ph_vocab), self._BUCKETS,
                                            self.params.size,
                                            self.params.num_layers,
                                            self.params.max_gradient_norm,
                                            self.params.batch_size,
                                            self.params.learning_rate,
                                            self.params.lr_decay_factor,
                                            forward_only=False,
                                            optimizer=self.params.optimizer)
    self.model.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


  def load_train_model(self, params):
    """Load G2P model for continuing train."""
    # Check for saved model.
    if not os.path.exists(os.path.join(self.model_dir, 'checkpoint')):
      raise RuntimeError("Model not found in %s" % self.model_dir)

    # Load model parameters.
    params.num_layers, params.size = data_utils.load_params(self.model_dir)

    # Prepare data and G2P Model.
    self.__prepare_model(params)

    # Restore model.
    print("Reading model parameters from %s" % self.model_dir)
    self.model.saver.restore(self.session, os.path.join(self.model_dir,
                                                        "model"))


  def create_train_model(self, params):
    """Create G2P model for train from scratch."""
    # Save model parameters.
    data_utils.save_params(params.num_layers, params.size, self.model_dir)

    # Prepare data and G2P Model
    self.__prepare_model(params)

    print("Created model with fresh parameters.")
    self.session.run(tf.global_variables_initializer())


  def train(self):
    """Train a gr->ph translation model using G2P data."""

    train_bucket_sizes = [len(self.train_set[b])
                          for b in xrange(len(self._BUCKETS))]
    # This is the training loop.
    step_time, train_loss, allow_excess_min = 0.0, 0.0, 1.5
    current_step, self.epochs_wo_improvement,\
      self.allow_epochs_wo_improvement = 0, 0, 2
    train_losses, eval_losses, epoch_losses = [], [], []
    while (self.params.max_steps == 0
           or self.model.global_step.eval(self.session)
           <= self.params.max_steps):
      # Get a batch and make a step.
      start_time = time.time()
      for from_row in range(0, max(train_bucket_sizes), self.params.batch_size):
        for bucket_id in range(len(self._BUCKETS)):
          if from_row <= train_bucket_sizes[bucket_id]:
            step_loss = self.__calc_step_loss(bucket_id, from_row)
            step_time += (time.time() - start_time) /\
              self.params.steps_per_checkpoint
            train_loss += step_loss / self.params.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics,
            # and run evals.
            if current_step % self.params.steps_per_checkpoint == 0:
              # Print statistics for the previous steps.
              train_ppx =\
                math.exp(train_loss) if train_loss < 300 else float('inf')
              print ("global step %d learning rate %.4f step-time %.2f "
                     "perplexity %.3f" %
                     (self.model.global_step.eval(self.session),
                      self.model.learning_rate.eval(self.session),
                      step_time, train_ppx))
              eval_loss = self.__calc_eval_loss()
              eval_ppx =\
                math.exp(eval_loss) if eval_loss < 300 else float('inf')
              print("  eval: perplexity %.3f" % (eval_ppx))
              # Decrease learning rate if no improvement was seen on train set
              # over last 3 times.
              if (len(train_losses) > 2
                  and train_loss > max(train_losses[-3:])):
                self.session.run(self.model.learning_rate_decay_op)

              # Save checkpoint and zero timer and loss.
              self.model.saver.save(self.session,
                                    os.path.join(self.model_dir, "model"),
                                    write_meta_graph=False)

              train_losses.append(train_loss)
              eval_losses.append(eval_loss)
              step_time, train_loss = 0.0, 0.0

      # After epoch pass, calculate average validation loss during
      # the previous epoch
      eval_losses = [loss for loss in eval_losses
                     if loss < (min(eval_losses) * allow_excess_min)]
      epoch_loss = (sum(eval_losses) / len(eval_losses)
                    if len(eval_losses) > 0 else float('inf'))
      epoch_losses.append(epoch_loss)

      # Make a decision to continue/stop training.
      stop_training = self.__should_stop_training(epoch_losses)
      if stop_training:
        break

      eval_losses = []

    print('Training done.')
    with tf.Graph().as_default():
      g2p_model_eval = G2PModel(self.model_dir)
      g2p_model_eval.load_decode_model()
      g2p_model_eval.evaluate(self.test_lines)


  def __should_stop_training(self, epoch_losses, window_scale=1.5):
    """Check stop training condition.
    Because models with different sizes need different number of epochs
    for improvement, we implemented stop criteria based on a expanding window
    of allowable number of epochs without improvement. Assuming how many
    maximum epochs it was needed for the previous improvements, we may increase
    allowable number of epochs without improvement. Model will stop training
    if number of epochs passed from previous improvement exceed maximal
    allowable number.

      Args:
        epoch_losses: losses on a validation set during the previous epochs;

      Returns:
        True/False: should or should not stop training;
    """
    if len(epoch_losses) > 1:
      print('Prev min epoch eval loss: %f, curr epoch eval loss: %f' %
            (min(epoch_losses[:-1]), epoch_losses[-1]))
      # Check if there was an improvement during the last epoch
      if epoch_losses[-1] < min(epoch_losses[:-1]):
        # Increase window if major part of previous window have been passed
        if (self.allow_epochs_wo_improvement <
            (self.epochs_wo_improvement * window_scale)):
          self.allow_epochs_wo_improvement =\
            int(math.ceil(self.epochs_wo_improvement * window_scale))
        print('Improved during the last epoch.')
        self.epochs_wo_improvement = 0
      else:
        print('No improvement during the last epoch.')
        self.epochs_wo_improvement += 1

      print('Number of the epochs passed from the last improvement: %d'
            % self.epochs_wo_improvement)
      print('Max allowable number of epochs for improvement: %d'
            % self.allow_epochs_wo_improvement)

      # Stop training if no improvement was seen during last
      # max allowable number of epochs
      if self.epochs_wo_improvement > self.allow_epochs_wo_improvement:
        return True
    return False


  def __calc_step_loss(self, bucket_id, from_row):
    """Choose a bucket according to data distribution. We pick a random number
    in [0, 1] and use the corresponding interval in train_buckets_scale.
    """
    # Get a batch and make a step.
    encoder_inputs, decoder_inputs, target_weights =\
      self.model.get_batch(self.train_set, bucket_id, from_row)
    _, step_loss, _ = self.model.step(self.session, encoder_inputs,
                                      decoder_inputs, target_weights,
                                      bucket_id, False)
    return step_loss


  def __calc_eval_loss(self):
    """Run evals on development set and print their perplexity.
    """
    eval_loss, steps = 0.0, 0
    for bucket_id in xrange(len(self._BUCKETS)):
      for from_row in xrange(0, len(self.valid_set[bucket_id]),
                             self.params.batch_size):
        encoder_inputs, decoder_inputs, target_weights =\
          self.model.get_batch(self.valid_set, bucket_id, from_row)
        _, loss, _ = self.model.step(self.session, encoder_inputs,
                                     decoder_inputs, target_weights,
                                     bucket_id, True)
        eval_loss += loss
        steps += 1
    return eval_loss/steps if steps > 0 else float('inf')


  def decode_word(self, word):
    """Decode input word to sequence of phonemes.

    Args:
      word: input word;

    Returns:
      phonemes: decoded phoneme sequence for input word;
    """
    # Check if all graphemes attended in vocabulary
    gr_absent = set([gr for gr in word if gr not in self.gr_vocab])
    if gr_absent:
      print("Symbols '%s' are not in vocabulary" % (
          "','".join(gr_absent).encode('utf-8')))
      return ""

    # Get token-ids for the input word.
    token_ids = [self.gr_vocab.get(s, data_utils.UNK_ID) for s in word]
    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(self._BUCKETS))
                     if self._BUCKETS[b][0] > len(token_ids)])
    # Get a 1-element batch to feed the word to the model.
    encoder_inputs, decoder_inputs, target_weights =\
      self.model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id, 0)
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
    return " ".join([self.rev_ph_vocab[output] for output in outputs])


  def interactive(self):
    """Decode word from standard input.
    """
    while True:
      try:
        word = input("> ")
        if not issubclass(type(word), text_type):
          word = text_type(word, encoding='utf-8', errors='replace')
      except EOFError:
        break
      if not word:
        break
      print(self.decode_word(word))


  def calc_error(self, dictionary):
    """Calculate a number of prediction errors.
    """
    errors = 0
    for word, pronunciations in dictionary.items():
      hyp = self.decode_word(word)
      if hyp not in pronunciations:
        errors += 1
    return errors


  def evaluate(self, test_lines):
    """Calculate and print out word error rate (WER) and Accuracy
       on test sample.

    Args:
      test_lines: List of test dictionary. Each element of list must be String
                containing word and its pronounciation (e.g., "word W ER D");
    """
    test_dic = data_utils.collect_pronunciations(test_lines)

    if len(test_dic) < 1:
      print("Test dictionary is empty")
      return

    print('Beginning calculation word error rate (WER) on test sample.')
    errors = self.calc_error(test_dic)

    print("Words: %d" % len(test_dic))
    print("Errors: %d" % errors)
    print("WER: %.3f" % (float(errors)/len(test_dic)))
    print("Accuracy: %.3f" % float(1-(errors/len(test_dic))))


  def decode(self, decode_lines, output_file=None):
    """Decode words from file.

    Returns:
      if [--output output_file] pointed out, write decoded word sequences in
      this file. Otherwise, print decoded words in standard output.
    """
    phoneme_lines = []

    # Decode from input file.
    if output_file:
      for word in decode_lines:
        word = word.strip()
        phonemes = self.decode_word(word)
        output_file.write(word)
        output_file.write(' ')
        output_file.write(phonemes)
        output_file.write('\n')
        phoneme_lines.append(phonemes)
      output_file.close()
    else:
      for word in decode_lines:
        word = word.strip()
        phonemes = self.decode_word(word)
        print(word + ' ' + phonemes)
        phoneme_lines.append(phonemes)
    return phoneme_lines


class TrainingParams(object):
  """Class with training parameters."""
  def __init__(self, flags=None):
    if flags:
      self.learning_rate = flags.learning_rate
      self.lr_decay_factor = flags.learning_rate_decay_factor
      self.max_gradient_norm = flags.max_gradient_norm
      self.batch_size = flags.batch_size
      self.size = flags.size
      self.num_layers = flags.num_layers
      self.steps_per_checkpoint = flags.steps_per_checkpoint
      self.max_steps = flags.max_steps
      self.optimizer = flags.optimizer
    else:
      self.learning_rate = 0.5
      self.lr_decay_factor = 0.99
      self.max_gradient_norm = 5.0
      self.batch_size = 64
      self.size = 64
      self.num_layers = 2
      self.steps_per_checkpoint = 200
      self.max_steps = 0
      self.optimizer = "sgd"

  def __str__(self):
    return ("Learning rate:        {}\n"
            "LR decay factor:      {}\n"
            "Max gradient norm:    {}\n"
            "Batch size:           {}\n"
            "Size of layer:        {}\n"
            "Number of layers:     {}\n"
            "Steps per checkpoint: {}\n"
            "Max steps:            {}\n"
            "Optimizer:            {}\n").format(
      self.learning_rate,
      self.lr_decay_factor,
      self.max_gradient_norm,
      self.batch_size,
      self.size,
      self.num_layers,
      self.steps_per_checkpoint,
      self.max_steps,
      self.optimizer)
