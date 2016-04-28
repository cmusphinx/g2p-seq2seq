# Copyright 2015 Google Inc. All Rights Reserved.
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
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_string("model", "/tmp", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("interactive", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_string("count_wer", "","Count Word Error rate for file.")
tf.app.flags.DEFINE_string("decode", "", "Decode file.")
tf.app.flags.DEFINE_string("output", "", "Decoding result file.")
tf.app.flags.DEFINE_string("train", "", "Train dictionary.")
tf.app.flags.DEFINE_string("valid", "", "Development dictionary.")
tf.app.flags.DEFINE_string("test", "", "Test dictionary.")


FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (40, 50)]


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
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
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  """Train a gr->ph translation model using G2P data."""
  # Prepare G2P data.
  print("Preparing G2P data in %s" % FLAGS.model)
  gr_train, ph_train, gr_dev, ph_dev, _, _, gr_vocab_size, ph_vocab_size = data_utils.prepare_g2p_data(
      FLAGS.model)

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False, gr_vocab_size, ph_vocab_size)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(gr_dev, ph_dev)
    train_set = read_data(gr_train, ph_train, FLAGS.max_train_data_size)
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
    while True:
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
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def decode_word(word, sess, model, gr_vocab, rev_ph_vocab):
  # Get token-ids for the input sentence.
  token_ids = data_utils.sentence_to_token_ids(word, gr_vocab)
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
  # Print out French sentence corresponding to outputs.
  res_phoneme_seq = " ".join([rev_ph_vocab[output] for output in outputs])
  return res_phoneme_seq



def interactive():
  with tf.Session() as sess:
    # Create model and load parameters.
    gr_vocab_path = os.path.join(FLAGS.model, "vocab.grapheme")
    ph_vocab_path = os.path.join(FLAGS.model, "vocab.phoneme")
    gr_vocab_size = data_utils.get_vocab_size(gr_vocab_path)
    ph_vocab_size = data_utils.get_vocab_size(ph_vocab_path)
    model = create_model(sess, True, gr_vocab_size, ph_vocab_size)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    gr_vocab, _ = data_utils.initialize_vocabulary(gr_vocab_path)
    _, rev_ph_vocab = data_utils.initialize_vocabulary(ph_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    word = " ".join(list(sys.stdin.readline()))
    while word:
      res_phoneme_seq = decode_word(word, sess, model, gr_vocab, rev_ph_vocab)
      print(res_phoneme_seq)
      print("> ", end="")
      sys.stdout.flush()
      word = " ".join(list(sys.stdin.readline()))


def count_wer():
  with tf.Session() as sess:
    # Create model and load parameters.
    gr_vocab_path = os.path.join(FLAGS.model, "vocab.grapheme")
    ph_vocab_path = os.path.join(FLAGS.model, "vocab.phoneme")
    gr_vocab_size = data_utils.get_vocab_size(gr_vocab_path)
    ph_vocab_size = data_utils.get_vocab_size(ph_vocab_path)
    model = create_model(sess, True, gr_vocab_size, ph_vocab_size)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    gr_vocab_path = os.path.join(FLAGS.model, "vocab.grapheme")
    ph_vocab_path = os.path.join(FLAGS.model, "vocab.phoneme")
    gr_vocab, _ = data_utils.initialize_vocabulary(gr_vocab_path)
    _, rev_ph_vocab = data_utils.initialize_vocabulary(ph_vocab_path)

    # Decode from input file.
    test = open(FLAGS.count_wer).read().split('\n')
    #test_gr_path = os.path.join(FLAGS.model, "test.grapheme")
    #test_ph_path = os.path.join(FLAGS.model, "test.phoneme")
    #test_graphemes = open(test_gr_path).read().split('\n')
    #test_phonemes = open(test_ph_path).read().split('\n')
    test_graphemes = []
    test_phonemes = []

    for line in test:
      lst = line.split('\t')
      if len(lst)>=2:
        test_graphemes.append(lst[0])
        test_phonemes.append(" ".join(lst[1:]))

    duplicates = {}
    total_dupl_num = 0
    for i, gr in enumerate(test_graphemes):
      if test_graphemes.count(gr) > 1:
        total_dupl_num += test_graphemes.count(gr) - 1
        if gr in duplicates:
          duplicates[gr].append(test_phonemes[i])
        else:
          duplicates[gr] = [test_phonemes[i]]


    errors = 0
    counter = 0
    dupl_error_calculated = []
    for i in range(len(test_graphemes)-1):
      if test_graphemes[i] not in duplicates:
        counter += 1
        word = " ".join(list(test_graphemes[i]))
        model_assumption = decode_word(word, sess, model, gr_vocab, rev_ph_vocab) 
        if model_assumption != test_phonemes[i]:
          errors += 1
      elif test_graphemes[i] not in dupl_error_calculated:
        counter += 1
        dupl_error_calculated.append(test_graphemes[i])
        word = " ".join(list(test_graphemes[i]))
        model_assumption = decode_word(word, sess, model, gr_vocab, rev_ph_vocab)
        if model_assumption not in duplicates[test_graphemes[i]]:
          errors += 1
          print(test_graphemes[i], " : ", model_assumption, "->", duplicates[test_graphemes[i]])

    print("WER : ", errors/counter )
    print("Accuracy : ", (1-errors/counter) )


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    gr_vocab_path = os.path.join(FLAGS.model, "vocab.grapheme")
    ph_vocab_path = os.path.join(FLAGS.model, "vocab.phoneme")
    gr_vocab_size = data_utils.get_vocab_size(gr_vocab_path)
    ph_vocab_size = data_utils.get_vocab_size(ph_vocab_path)
    model = create_model(sess, True, gr_vocab_size, ph_vocab_size)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    gr_vocab_path = os.path.join(FLAGS.model, "vocab.grapheme")
    ph_vocab_path = os.path.join(FLAGS.model, "vocab.phoneme")
    gr_vocab, _ = data_utils.initialize_vocabulary(gr_vocab_path)
    _, rev_ph_vocab = data_utils.initialize_vocabulary(ph_vocab_path)

    # Decode from input file.
    graphemes = open(FLAGS.decode).readlines()

    output_file_path = FLAGS.output

    if output_file_path:
      with gfile.GFile(output_file_path, mode="w") as output_file:
        for i in range(len(graphemes)-1):
          word = " ".join(list(graphemes[i]))
          res_phoneme_seq = decode_word(word, sess, model, gr_vocab, rev_ph_vocab)
          output_file.write(res_phoneme_seq)
          output_file.write('\n')
    else:
      for i in range(len(graphemes)-1):
        word = " ".join(list(graphemes[i]))
        res_phoneme_seq = decode_word(word, sess, model, gr_vocab, rev_ph_vocab)
        print(res_phoneme_seq)
        sys.stdout.flush()


def main(_):
  if FLAGS.decode:
    decode()
  elif FLAGS.interactive:
    interactive()
  elif FLAGS.count_wer:
    count_wer()
  else:
    if FLAGS.train:
      source_dic = open(FLAGS.train).readlines()
      train_dic, valid_dic, test_dic = [], [], []
      if (not FLAGS.valid) and (not FLAGS.test):
        for i, word in enumerate(source_dic):
          if i % 20 == 0 or i % 20 == 1: 
            test_dic.append(word)
          elif i % 20 == 2:
            valid_dic.append(word)
          else: train_dic.append(word)
      elif not FLAGS.valid:
        test_dic = open(FLAGS.test).readlines()
        for i, word in enumerate(source_dic):
          if i % 20 == 0:
            valid_dic.append(word)
          else: train_dic.append(word)
      elif not FLAGS.test:
        valid_dic = open(FLAGS.valid).readlines()
        for i, word in enumerate(source_dic):
          if i % 10 == 0:
            test_dic.append(word)
          else: train_dic.append(word)
      else:
        valid_dic = open(FLAGS.valid).readlines()
        test_dic = open(FLAGS.test).readlines()
        train_dic = source_dic
    else:
      raise ValueError("Train dictionary absent.")
    train_path = FLAGS.model + "train"
    valid_path = FLAGS.model + "valid"
    test_path = FLAGS.model + "test"
    data_utils.split_to_grapheme_phoneme(train_dic, train_path)
    data_utils.split_to_grapheme_phoneme(valid_dic, valid_path)
    data_utils.split_to_grapheme_phoneme(test_dic, test_path)
    train()

if __name__ == "__main__":
  tf.app.run()
