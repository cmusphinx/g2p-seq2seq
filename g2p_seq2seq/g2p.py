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
import random
import sys
import time
import codecs

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.platform import gfile

import data_utils as data_utils
from tensorflow.models.rnn.translate import seq2seq_model

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.8,
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
tf.app.flags.DEFINE_string("evaluate", "","Count Word Error rate for file.")
tf.app.flags.DEFINE_string("decode", "", "Decode file.")
tf.app.flags.DEFINE_string("output", "", "Decoding result file.")
tf.app.flags.DEFINE_string("train", "", "Train dictionary.")
tf.app.flags.DEFINE_string("valid", "", "Development dictionary.")
tf.app.flags.DEFINE_string("test", "", "Test dictionary.")
tf.app.flags.DEFINE_integer("max_steps", 10000,
                            "How many training steps to do until stop training (0: no limit).")


FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (40, 50)]


def put_into_buckets(source, target):
  """Put data from source and target into buckets.

  Args:
    source: data with ids for the source language.
    target: data with ids for the target language;
      it must be aligned with the source data: n-th line contains the desired
      output for n-th line from the source.

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of ids.
  """
  data_set = [[] for _ in _buckets]
  for i in range(len(source)):
    source_ids = source[i]
    target_ids = target[i]
    target_ids.append(data_utils.EOS_ID)
    for bucket_id, (source_size, target_size) in enumerate(_buckets):
      if len(source_ids) < source_size and len(target_ids) < target_size:
        data_set[bucket_id].append([source_ids, target_ids])
        break
  return data_set


def create_model(session, forward_only, gr_vocab_size, ph_vocab_size):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      gr_vocab_size, ph_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.model)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  elif tf.gfile.Exists(os.path.join(FLAGS.model, "model")):
    model.saver.restore(session, os.path.join(FLAGS.model, "model"))
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train(train_gr, train_ph, valid_gr, valid_ph, test_dic):
  """Train a gr->ph translation model using G2P data."""
  # Prepare G2P data.
  print("Preparing G2P data")
  train_gr_ids, train_ph_ids, valid_gr_ids, valid_ph_ids, gr_vocab, ph_vocab = data_utils.prepare_g2p_data(FLAGS.model, train_gr, train_ph, valid_gr, valid_ph)
  gr_vocab_size = len(gr_vocab)
  ph_vocab_size = len(ph_vocab)
  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False, gr_vocab_size, ph_vocab_size)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data.")
    valid_set = put_into_buckets(valid_gr_ids, valid_ph_ids)
    train_set = put_into_buckets(train_gr_ids, train_ph_ids)
    
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
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

    while (FLAGS.max_steps == 0 or model.global_step.eval() <= FLAGS.max_steps ):
      # Choose a bucket according to data distribution. We pick a random number
      # in [0, 1] and use the corresponding interval in train_buckets_scale.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        # Print statistics for the previous epoch.
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        if len(previous_losses) > 34 and previous_losses[-35:-34] <= min(previous_losses[-35:]):
          break
        previous_losses.append(loss)
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.model, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        # Run evals on development set and print their perplexity.
        for bucket_id in xrange(len(_buckets)):
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              valid_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
    else:
      print('Training process stopped.')
      print('Beginning calculation word error rate (WER) on test sample.')
      ph_vocab_path = os.path.join(FLAGS.model, "vocab.phoneme")
      _, rev_ph_vocab = data_utils.initialize_vocabulary(ph_vocab_path)
      model.batch_size = 1  # We decode one word at a time.
      evaluate(test_dic, sess, model, gr_vocab, rev_ph_vocab)


def get_vocabs_load_model(sess):
  """Initialize and return vocabularies and pathes to them.
  And load saved model.

  Returns:
    gr_vocab: Graphemes vocabulary;
    rev_ph_vocab: Reversed phonemes vocabulary;
    model: Trained model.
  """
  # Initialize vocabularies
  gr_vocab_path = os.path.join(FLAGS.model, "vocab.grapheme")
  ph_vocab_path = os.path.join(FLAGS.model, "vocab.phoneme")
  gr_vocab, _ = data_utils.initialize_vocabulary(gr_vocab_path)
  _, rev_ph_vocab = data_utils.initialize_vocabulary(ph_vocab_path)

  # Get vocabulary sizes
  gr_vocab_size = len(gr_vocab)
  ph_vocab_size = len(rev_ph_vocab)
  # Load model
  model = create_model(sess, True, gr_vocab_size, ph_vocab_size)
  model.batch_size = 1  # We decode one word at a time.
  return (gr_vocab, rev_ph_vocab, model)


def decode_word(word, sess, model, gr_vocab, rev_ph_vocab):
  # Get token-ids for the input sentence.
  token_ids = [gr_vocab.get(s, data_utils.UNK_ID) for s in word]
  # Which bucket does it belong to?
  bucket_id = min([b for b in xrange(len(_buckets))
                   if _buckets[b][0] > len(token_ids)])
  # Get a 1-element batch to feed the sentence to the model.
  encoder_inputs, decoder_inputs, target_weights = model.get_batch(
      {bucket_id: [(token_ids, [])]}, bucket_id)
  # Get output logits for the sentence.
  _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True)
  # This is a greedy decoder - outputs are just argmaxes of output_logits.
  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
  # If there is an EOS symbol in outputs, cut them at that point.
  if data_utils.EOS_ID in outputs:
    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
  # Print out phoneme corresponding to outputs.
  res_phoneme_seq = " ".join([rev_ph_vocab[output] for output in outputs])
  return res_phoneme_seq



def interactive():
  with tf.Session() as sess:
    gr_vocab, rev_ph_vocab, model = get_vocabs_load_model(sess)
    print("> ", end="")

    while True:
      word = sys.stdin.readline().decode("utf-8").strip()
      if word:
        gr_absent = set(gr for gr in word if gr not in gr_vocab)
        if not gr_absent:
          res_phoneme_seq = decode_word(word, sess, model, gr_vocab, rev_ph_vocab)
          print(res_phoneme_seq)
        else:
          print("Symbols '%s' are not in vocabulary" % "','".join(gr_absent) )
      else: break
      print("> ", end="")


def calc_error(sess, model, w_ph_dict, gr_vocab, rev_ph_vocab):
  errors = 0
  for word, phonetics in w_ph_dict.items():
    if len(phonetics) == 1:
      gr_absent = set(gr for gr in word if gr not in gr_vocab)
      if not gr_absent:
        model_assumption = decode_word(word, sess, model, gr_vocab, rev_ph_vocab)
        if model_assumption not in phonetics:
          errors += 1
      else:
        raise ValueError("Symbols '%s' are not in vocabulary" % "','".join(gr_absent) )
  return errors


def evaluate(test_dic=None, sess=None, model=None, gr_vocab=None, rev_ph_vocab=None):
  """Calculate and print out word error rate (WER) and Accuracy on test sample.

  Args:
    No arguments need if this function called not from active Session in tensorflow.
    Otherwise following arguments should be pointed out:
    test_dic: List of test dictionary. Each element of list must be String contained 
              word and its pronounciation (e.g., "word W ER D");
    sess: Current tensorflow session;
    model: Current active model;
    gr_vocab: Graphemes vocabulary dictionary (a dictionary mapping string to integers);
    rev_ph_vocab: reversed Phonemes vocabulary list (a list, which reverses the vocabulary mapping).
  """
  if not test_dic:
    # Decode from input file.
    test_dic = codecs.open(FLAGS.evaluate, "r", "utf-8").readlines()

  w_ph_dict = {}
  for line in test_dic:
    lst = line.strip().split()
    if len(lst)>=2:
      if lst[0] not in w_ph_dict: w_ph_dict[lst[0]] = [" ".join(lst[1:])]
      else: w_ph_dict[lst[0]].append(" ".join(lst[1:]))

  errors = 0
  # Calculate errors
  if not sess:
    with tf.Session() as sess:
      gr_vocab, rev_ph_vocab, model = get_vocabs_load_model(sess)
      errors = calc_error(sess, model, w_ph_dict, gr_vocab, rev_ph_vocab)
  else:
    errors = calc_error(sess, model, w_ph_dict, gr_vocab, rev_ph_vocab)
  print("WER : ", errors/len(w_ph_dict) )
  print("Accuracy : ", ( 1-(errors/len(w_ph_dict)) ) )


def decode(word_list_file_path):
  with tf.Session() as sess:
    gr_vocab, rev_ph_vocab, model = get_vocabs_load_model(sess)

    # Decode from input file.
    graphemes = codecs.open(word_list_file_path, "r", "utf-8").readlines()

    output_file_path = FLAGS.output

    if output_file_path:
      with codecs.open(output_file_path, "w", "utf-8") as output_file:
        for word in graphemes:
          word = word.strip()
          gr_absent = set(gr for gr in word if gr not in gr_vocab)
          if not gr_absent:
            res_phoneme_seq = decode_word(word, sess, model, gr_vocab, rev_ph_vocab)
            output_file.write(word)
            output_file.write(' ')
            output_file.write(res_phoneme_seq)
            output_file.write('\n')
          else:
            raise ValueError("Symbols '%s' are not in vocabulary" % "','".join(gr_absent) )
    else:
      for word in graphemes:
        word = word.strip()
        gr_absent = set(gr for gr in word if gr not in gr_vocab)
        if not gr_absent:
          res_phoneme_seq = decode_word(word, sess, model, gr_vocab, rev_ph_vocab)
          print(word + ' ' + res_phoneme_seq)
        else:
          raise ValueError("Symbols '%s' are not in vocabulary" % "','".join(gr_absent) )


def main(_):
  if FLAGS.decode:
    decode(FLAGS.decode)
  elif FLAGS.interactive:
    interactive()
  elif FLAGS.evaluate:
    evaluate()
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
    train_gr, train_ph = data_utils.split_to_grapheme_phoneme(train_dic)
    valid_gr, valid_ph = data_utils.split_to_grapheme_phoneme(valid_dic)
    train(train_gr, train_ph, valid_gr, valid_ph, test_dic)

if __name__ == "__main__":
  tf.app.run()
