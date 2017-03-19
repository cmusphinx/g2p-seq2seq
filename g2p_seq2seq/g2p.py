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

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2

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
    train_total_size = float(sum(train_bucket_sizes))
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, train_loss = 0.0, 0.0
    current_step, num_iter_wo_improve = 0, 0
    prev_train_losses, prev_valid_losses = [], []
    num_iter_cover_train = int(sum(train_bucket_sizes) /
                               self.params.batch_size /
                               self.params.steps_per_checkpoint)
    while (self.params.max_steps == 0
           or self.model.global_step.eval(self.session)
           <= self.params.max_steps):
      # Get a batch and make a step.
      start_time = time.time()
      step_loss = self.__calc_step_loss(train_buckets_scale)
      step_time += (time.time() - start_time) / self.params.steps_per_checkpoint
      train_loss += step_loss / self.params.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % self.params.steps_per_checkpoint == 0:
        # Print statistics for the previous steps.
        train_ppx = math.exp(train_loss) if train_loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (self.model.global_step.eval(self.session),
                         self.model.learning_rate.eval(self.session),
                         step_time, train_ppx))
        eval_loss = self.__calc_eval_loss()
        eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
        print("  eval: perplexity %.2f" % (eval_ppx))
        # Decrease learning rate if no improvement was seen on train set
        # over last 3 times.
        if (len(prev_train_losses) > 2
            and train_loss > max(prev_train_losses[-3:])):
          self.session.run(self.model.learning_rate_decay_op)

        if (len(prev_valid_losses) > 0
            and eval_loss <= min(prev_valid_losses)):
          # Save checkpoint and zero timer and loss.
          self.model.saver.save(self.session,
                                os.path.join(self.model_dir, "model"),
                                write_meta_graph=False)

        if (len(prev_valid_losses) > 0
            and eval_loss >= min(prev_valid_losses)):
          num_iter_wo_improve += 1
        else:
          num_iter_wo_improve = 0

        if num_iter_wo_improve > num_iter_cover_train * 2:
          print("No improvement over last %d times. Training will stop after %d"
                "iterations if no improvement was seen."
                % (num_iter_wo_improve,
                   num_iter_cover_train - num_iter_wo_improve))

        # Stop train if no improvement was seen on validation set
        # over last 3 epochs.
        if num_iter_wo_improve > num_iter_cover_train * 3:
          break

        prev_train_losses.append(train_loss)
        prev_valid_losses.append(eval_loss)
        step_time, train_loss = 0.0, 0.0

    print('Training done.')
    with tf.Graph().as_default():
      g2p_model_eval = G2PModel(self.model_dir)
      g2p_model_eval.load_decode_model()
      g2p_model_eval.evaluate(self.test_lines)


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


  def __calc_eval_loss(self):
    """Run evals on development set and print their perplexity.
    """
    eval_loss, num_iter_total = 0.0, 0.0
    for bucket_id in xrange(len(self._BUCKETS)):
      num_iter_cover_valid = int(math.ceil(len(self.valid_set[bucket_id])/
                                           self.params.batch_size))
      num_iter_total += num_iter_cover_valid
      for batch_id in xrange(num_iter_cover_valid):
        encoder_inputs, decoder_inputs, target_weights =\
            self.model.get_eval_set_batch(self.valid_set, bucket_id,
                                          batch_id * self.params.batch_size)
        _, eval_batch_loss, _ = self.model.step(self.session, encoder_inputs,
                                                decoder_inputs, target_weights,
                                                bucket_id, True)
        eval_loss += eval_batch_loss
    eval_loss = eval_loss/num_iter_total if num_iter_total > 0 else float('inf')
    return eval_loss


  def decode_word(self, word):
    """Decode input word to sequence of phonemes.

    Args:
      word: input word;

    Returns:
      phonemes: decoded phoneme sequence for input word;
    """
    # Check if all graphemes attended in vocabulary
    gr_absent = [gr for gr in word if gr not in self.gr_vocab]
    if gr_absent:
      print("Symbols '%s' are not in vocabulary" % "','".join(gr_absent).encode('utf-8'))
      return ""

    # Get token-ids for the input word.
    token_ids = [self.gr_vocab.get(s, data_utils.UNK_ID) for s in word]
    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(self._BUCKETS))
                     if self._BUCKETS[b][0] > len(token_ids)])
    # Get a 1-element batch to feed the word to the model.
    encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)
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
