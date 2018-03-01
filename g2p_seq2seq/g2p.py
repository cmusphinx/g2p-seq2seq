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
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
import tensorflow as tf
import numpy as np
import re

from g2p_seq2seq import g2p_trainer_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_utils
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import decoding

from tensor2tensor.data_generators import text_encoder
from six.moves import input
from six import text_type

from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.framework import graph_util

EOS = text_encoder.EOS


class G2PModel(object):
  """Grapheme-to-Phoneme translation model class.
  """
  def __init__(self, params, file_path="", is_training=False):
    # Point out the current directory with t2t problem specified for g2p task.
    usr_dir.import_usr_dir(os.path.dirname(os.path.abspath(__file__)))
    self.params = params
    self.file_path = file_path
    if not os.path.exists(self.params.model_dir):
      os.makedirs(self.params.model_dir)

    # Register g2p problem.
    self.problem = registry._PROBLEMS[self.params.problem_name](
        self.params.model_dir, file_path=file_path, is_training=is_training)
    trainer_utils.log_registry()

    self.frozen_graph_filename = os.path.join(self.params.model_dir,
                                              "frozen_model.pb")
    self.first_ex = False
    if is_training:
      self.train_preprocess_file_path, self.dev_preprocess_file_path =\
          None, None
      self.estimator, self.decode_hp = self.__prepare_decode_model()

    elif os.path.exists(self.frozen_graph_filename):
      self.estimator, self.decode_hp = self.__prepare_decode_model()
      self.__load_graph()
      self.checkpoint_path = tf.train.latest_checkpoint(self.params.model_dir)

    else:
      self.estimator, self.decode_hp = self.__prepare_decode_model()

  def prepare_datafiles(self, train_path, dev_path):
    """Prepare preprocessed datafiles."""
    self.train_preprocess_file_path, self.dev_preprocess_file_path =\
        self.problem.generate_preprocess_data(train_path, dev_path)

  def __prepare_decode_model(self):
    """Prepare utilities for decoding."""
    hparams = trainer_utils.create_hparams(
        self.params.hparams_set,
        self.params.data_dir,
        passed_hparams=self.params.hparams)
    estimator, _ = g2p_trainer_utils.create_experiment_components(
        params=self.params,
        hparams=hparams,
        run_config=trainer_utils.create_run_config(self.params.model_dir),
        problem_instance=self.problem)

    decode_hp = decoding.decode_hparams(self.params.decode_hparams)
    decode_hp.add_hparam("shards", 1)
    return estimator, decode_hp

  def __prepare_interactive_model(self):
    """Create monitored session and generator that reads from the terminal and
    yields "interactive inputs".

    Due to temporary limitations in tf.learn, if we don't want to reload the
    whole graph, then we are stuck encoding all of the input as one fixed-size
    numpy array.

    We yield int32 arrays with shape [const_array_size].  The format is:
    [num_samples, decode_length, len(input ids), <input ids>, <padding>]

    Raises:
      ValueError: Could not find a trained model in model_dir.
      ValueError: if batch length of predictions are not same.
    """
    word = self.__get_word()

    self.first_ex = True
    self.decode_word(word)
    self.first_ex = False
    prob_choice = np.array(0).astype(np.int32)

    def input_fn():
      """Input function returning features which is a dictionary of
        string feature name to `Tensor` or `SparseTensor`. If it returns a
        tuple, first item is extracted as features. Prediction continues until
        `input_fn` raises an end-of-input exception (`OutOfRangeError` or
        `StopIteration`)."""
      gen_fn = make_input_fn(self.inputs, prob_choice)
      example = gen_fn()
      example = decoding._interactive_input_tensor_to_features_dict(
          example, self.estimator.params)
      return example

    self.input_fn = input_fn

    if os.path.exists(self.frozen_graph_filename):
      return

    with estimator_lib.ops.Graph().as_default() as graph:
      self.features = self.estimator._get_features_from_input_fn(
          input_fn, estimator_lib.model_fn_lib.ModeKeys.PREDICT)
      # List of `SessionRunHook` subclass instances. Used for callbacks inside
      # the prediction call.
      hooks = estimator_lib._check_hooks_type(None)
      # Check that model has been trained.
      # Path of a specific checkpoint to predict. The latest checkpoint
      # in `model_dir` is used
      checkpoint_path = estimator_lib.saver.latest_checkpoint(
          self.params.model_dir)
      if not checkpoint_path:
        raise ValueError('Could not find trained model in model_dir: {}.'
                         .format(self.params.model_dir))

      estimator_lib.random_seed.set_random_seed(
          self.estimator._config.tf_random_seed)
      self.estimator._create_and_assert_global_step(graph)
      self.estimator_spec = self.estimator._call_model_fn(
          self.features, None, estimator_lib.model_fn_lib.ModeKeys.PREDICT,
          self.estimator.config)
      self.mon_sess = estimator_lib.training.MonitoredSession(
          session_creator=estimator_lib.training.ChiefSessionCreator(
              checkpoint_filename_with_path=checkpoint_path,
              scaffold=self.estimator_spec.scaffold,
              config=self.estimator._session_config),
          hooks=hooks)

      pronunciations = self.decode_word(word)
      print("Pronunciations: {}".format(pronunciations))

  def decode_word(self, word):
    """Decode word.

    Args:
      word: word for decoding.

    Returns:
      pronunciation: a decoded phonemes sequence for input word.
    """
    num_samples = 1
    decode_length = 100
    p_hparams = self.estimator.params.problems[0]
    vocabulary = p_hparams.vocabulary["inputs"]
    # This should be longer than the longest input.
    const_array_size = 50

    input_ids = vocabulary.encode(word)
    input_ids.append(text_encoder.EOS_ID)
    self.inputs = [num_samples, decode_length, len(input_ids)] + input_ids
    assert len(self.inputs) < const_array_size
    self.inputs += [0] * (const_array_size - len(self.inputs))

    if self.first_ex:
      return

    res_iter = self.estimator.predict(self.input_fn)
    result = res_iter.next()
    pronunciations = []
    if self.decode_hp.return_beams:
      beams = np.split(result["outputs"], self.decode_hp.beam_size, axis=0)
      for k, beam in enumerate(beams):
        tf.logging.info("BEAM %d:" % k)
        beam_string = self.problem.target_vocab.decode(
            decoding._save_until_eos(beam, is_image=False))
        pronunciations.append(beam_string)
        tf.logging.info(beam_string)
    else:
      if self.decode_hp.identity_output:
        tf.logging.info(" ".join(map(str, result["outputs"].flatten())))
      else:
        res = result["outputs"].flatten()
        if text_encoder.EOS_ID in res:
          index = list(res).index(text_encoder.EOS_ID)
          res = res[0:index]
        pronunciations.append(self.problem.target_vocab.decode(res))
    return pronunciations

  def __run_op(self, sess, decode_op, feed_input):
    """Run tensorflow operation for decoding."""
    saver = tf.train.import_meta_graph(self.checkpoint_path + ".meta",
                                       import_scope=None, clear_devices=True)
    saver.restore(sess, self.checkpoint_path)
    inp = tf.placeholder(tf.string, name="inp_decode")[0]
    results = sess.run(decode_op,
                       feed_dict={"inp_decode:0" : [feed_input]})
    return results

  def __get_word(self):
    """Get next word in the interactive mode."""
    word = ""
    try:
      word = input("> ")
      if not issubclass(type(word), text_type):
        word = text_type(word, encoding="utf-8", errors="replace")
    except EOFError:
      pass
    if not word:
      pass
    return word

  def train(self):
    """Run training."""
    g2p_trainer_utils.run(
        params=self.params,
        problem_instance=self.problem,
        train_preprocess_file_path=self.train_preprocess_file_path,
        dev_preprocess_file_path=self.dev_preprocess_file_path)

  def interactive(self):
    """Interactive decoding."""
    self.__prepare_interactive_model()

    if os.path.exists(self.frozen_graph_filename):
      with tf.Session(graph=self.graph) as sess:
        inp = tf.placeholder(tf.string, name="inp_decode")[0]
        decode_op = tf.py_func(self.decode_word, [inp], tf.string)
        while True:
          word = self.__get_word()
          result = self.__run_op(sess, decode_op, word)
          print ("output: " + result)
    else:
      while not self.mon_sess.should_stop():
        self.__get_word()
        pronunciations = self.decode_word(word)
        print("Pronunciations: {}".format(pronunciations))

  def decode(self, output_file_path):
    """Run decoding mode."""
    if os.path.exists(self.frozen_graph_filename):
      with tf.Session(graph=self.graph) as sess:
        inp = tf.placeholder(tf.string, name="inp_decode")[0]
        decode_op = tf.py_func(self.__decode_from_file, [inp],
                               [tf.string, tf.string])
        [inputs, decodes] = self.__run_op(sess, decode_op, self.file_path)
    else:
      inputs, decodes = self.__decode_from_file(self.file_path)

    # If path to the output file pointed out, dump decoding results to the file
    if output_file_path:
      tf.logging.info("Writing decodes into %s" % output_file_path)
      outfile = tf.gfile.Open(output_file_path, "w")
      if self.decode_hp.return_beams:
        for index in range(len(inputs)):
          outfile.write("%s%s" % ("\t".join(decodes[index]),
                                  self.decode_hp.delimiter))
      else:
        for index in range(len(inputs)):
          outfile.write("%s%s" % (decodes[index], self.decode_hp.delimiter))

  def evaluate(self):
    """Run evaluation mode."""
    words, pronunciations = [], []
    for case in self.problem.generator(self.file_path,
                                       self.problem.source_vocab,
                                       self.problem.target_vocab):
      word = self.problem.source_vocab.decode(case["inputs"]).replace(
          EOS, "").strip()
      pronunciation = self.problem.target_vocab.decode(case["targets"]).replace(
          EOS, "").strip()
      words.append(word)
      pronunciations.append(pronunciation)

    self.g2p_gt_map = create_g2p_gt_map(words, pronunciations)

    if os.path.exists(self.frozen_graph_filename):
      with tf.Session(graph=self.graph) as sess:
        inp = tf.placeholder(tf.string, name="inp_decode")[0]
        decode_op = tf.py_func(self.calc_errors, [inp], [tf.int64, tf.int64])
        [correct, errors] = self.__run_op(sess, decode_op, self.file_path)

    else:
      correct, errors = self.calc_errors(self.file_path)

    print("Words: %d" % (correct+errors))
    print("Errors: %d" % errors)
    print("WER: %.3f" % (float(errors)/(correct+errors)))
    print("Accuracy: %.3f" % float(1.-(float(errors)/(correct+errors))))
    return self.g2p_gt_map

  def freeze(self):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(self.params.model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"

    # Before exporting our graph, we need to precise what is our output node
    # This is how TF decides what part of the Graph he has to keep and what
    # part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = ["transformer/body/model/parallel_0/body/decoder/layer_0/self_attention/multihead_attention/dot_product_attention/Softmax",
                         "transformer/body/model/parallel_0/body/decoder/layer_0/encdec_attention/multihead_attention/dot_product_attention/Softmax",
                         "transformer/body/model/parallel_0/body/decoder/layer_1/self_attention/multihead_attention/dot_product_attention/Softmax",
                         "transformer/body/model/parallel_0/body/decoder/layer_1/encdec_attention/multihead_attention/dot_product_attention/Softmax"]

    # We clear devices to allow TensorFlow to control on which device it will
    # load operations
    clear_devices = True
    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                       clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
      saver.restore(sess, input_checkpoint)

      # We use a built-in TF helper to export variables to constants
      output_graph_def = graph_util.convert_variables_to_constants(
          sess, # The session is used to retrieve the weights
          input_graph_def, # The graph_def is used to retrieve the nodes
          output_node_names, # The output node names are used to select the
                             #usefull nodes
          variable_names_blacklist=['global_step'])

      # Finally we serialize and dump the output graph to the filesystem
      with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
      print("%d ops in the final graph." % len(output_graph_def.node))

  def __load_graph(self):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(self.frozen_graph_filename, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as self.graph:
      # The name var will prefix every op/nodes in your graph
      # Since we load everything in a new graph, this is not needed
      tf.import_graph_def(graph_def, name="import")

  def __decode_from_file(self, filename):
    """Compute predictions on entries in filename and write them out."""

    if not self.decode_hp.batch_size:
      self.decode_hp.batch_size = 32
      tf.logging.info("decode_hp.batch_size not specified; default=%d" %
                      self.decode_hp.batch_size)

    hparams = self.estimator.params
    problem_id = self.decode_hp.problem_idx
    # Inputs vocabulary is set to targets if there are no inputs in the problem,
    # e.g., for language models where the inputs are just a prefix of targets.
    inputs_vocab = hparams.problems[problem_id].vocabulary["inputs"]
    targets_vocab = hparams.problems[problem_id].vocabulary["targets"]
    problem_name = "grapheme_to_phoneme_problem"
    tf.logging.info("Performing decoding from a file.")
    inputs = _get_inputs(filename)
    num_decode_batches = (len(inputs) - 1) // self.decode_hp.batch_size + 1

    def input_fn():
      """Function for inputs generator."""
      input_gen = _decode_batch_input_fn(
          problem_id, num_decode_batches, inputs, inputs_vocab,
          self.decode_hp.batch_size, self.decode_hp.max_input_size)
      gen_fn = decoding.make_input_fn_from_generator(input_gen)
      example = gen_fn()
      return decoding._decode_input_tensor_to_features_dict(example, hparams)

    decodes = []
    result_iter = self.estimator.predict(input_fn)
    for result in result_iter:
      if self.decode_hp.return_beams:
        beam_decodes = []
        output_beams = np.split(result["outputs"], self.decode_hp.beam_size,
                                axis=0)
        for k, beam in enumerate(output_beams):
          tf.logging.info("BEAM %d:" % k)
          decoded_outputs, _ = decoding.log_decode_results(
              result["inputs"],
              beam,
              problem_name,
              None,
              inputs_vocab,
              targets_vocab)
          beam_decodes.append(decoded_outputs)
        decodes.append(beam_decodes)
      else:
        decoded_outputs, _ = decoding.log_decode_results(
            result["inputs"],
            result["outputs"],
            problem_name,
            None,
            inputs_vocab,
            targets_vocab)
        decodes.append(decoded_outputs)

    return [inputs, decodes]

  def calc_errors(self, decode_file_path):
    """Calculate a number of prediction errors."""
    inputs, decodes = self.__decode_from_file(decode_file_path)

    correct, errors = 0, 0
    for index, word in enumerate(inputs):
      if self.decode_hp.return_beams:
        beam_correct_found = False
        for beam_decode in decodes[index]:
          if beam_decode in self.g2p_gt_map[word]:
            beam_correct_found = True
            break
        if beam_correct_found:
          correct += 1
        else:
          errors += 1
      else:
        if decodes[index] in self.g2p_gt_map[word]:
          correct += 1
        else:
          errors += 1

    return correct, errors


def make_input_fn(x_out, prob_choice):
  """Use py_func to yield elements from the given generator."""
  inp = {"inputs": np.array(x_out).astype(np.int32),
         "problem_choice": prob_choice}
  flattened = tf.contrib.framework.nest.flatten(inp)
  types = [t.dtype for t in flattened]
  shapes = [[None] * len(t.shape) for t in flattened]
  first_ex_list = [inp]

  def py_func():
    """Flatten example."""
    if first_ex_list:
      example = first_ex_list.pop()
    else:
      example = inp
    return tf.contrib.framework.nest.flatten(example)

  def input_fn():
    """Input function"""
    flat_example = tf.py_func(py_func, [], types)
    _ = [t.set_shape(shape) for t, shape in zip(flat_example, shapes)]
    example = tf.contrib.framework.nest.pack_sequence_as(inp, flat_example)
    return example

  return input_fn


def create_g2p_gt_map(words, pronunciations):
  """Create grapheme-to-phoneme ground true mapping."""
  g2p_gt_map = {}
  for word, pronunciation in zip(words, pronunciations):
    if word in g2p_gt_map:
      g2p_gt_map[word].append(pronunciation)
    else:
      g2p_gt_map[word] = [pronunciation]
  return g2p_gt_map


def _get_inputs(filename, delimiters="\t "):
  """Returning inputs.

  Args:
    filename: path to file with inputs, 1 per line.
    delimiters: str, delimits records in the file.

  Returns:
    a list of inputs

  """
  tf.logging.info("Getting inputs")
  DELIMITERS_REGEX = re.compile("[" + delimiters + "]+")

  inputs = []
  with tf.gfile.Open(filename) as input_file:
    lines = input_file.readlines()
    for line in lines:
      if set("[" + delimiters + "]+$").intersection(line):
        items = re.split(DELIMITERS_REGEX, line.strip(), maxsplit=1)
        inputs.append(items[0])
      else:
        inputs.append(line.strip())
  return inputs


def _decode_batch_input_fn(problem_id, num_decode_batches, inputs,
                           vocabulary, batch_size, max_input_size):
  """Decode batch"""
  tf.logging.info(" batch %d" % num_decode_batches)
  for b in range(num_decode_batches):
    tf.logging.info("Decoding batch %d" % b)
    batch_length = 0
    batch_inputs = []
    for _inputs in inputs[b * batch_size:(b + 1) * batch_size]:
      input_ids = vocabulary.encode(_inputs)
      if max_input_size > 0:
        # Subtract 1 for the EOS_ID.
        input_ids = input_ids[:max_input_size - 1]
      input_ids.append(text_encoder.EOS_ID)
      batch_inputs.append(input_ids)
      if len(input_ids) > batch_length:
        batch_length = len(input_ids)
    final_batch_inputs = []
    for input_ids in batch_inputs:
      assert len(input_ids) <= batch_length
      x = input_ids + [0] * (batch_length - len(input_ids))
      final_batch_inputs.append(x)

    yield {
        "inputs": np.array(final_batch_inputs).astype(np.int32),
        "problem_choice": np.array(problem_id).astype(np.int32),
    }
