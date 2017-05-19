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
from __future__ import unicode_literals

import math
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2

from g2p_seq2seq import data_utils
from g2p_seq2seq import seq2seq_model
#import data_utils
#import seq2seq_model

from six.moves import xrange, input  # pylint: disable=redefined-builtin
from six import text_type, string_types

from pydoc import locate
import yaml

from g2p_seq2seq.seq2seq import tasks, models
from g2p_seq2seq.seq2seq.configurable import _maybe_load_yaml, _deep_merge_dict, _create_from_dict
from g2p_seq2seq.seq2seq.data import input_pipeline
from g2p_seq2seq.seq2seq.inference import create_inference_graph
from g2p_seq2seq.seq2seq.training import utils as training_utils

from tensorflow.contrib.learn.python.learn import learn_runner
#from seq2seq import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
#from seq2seq import run_config
from tensorflow import gfile
from g2p_seq2seq.seq2seq.training import hooks
from g2p_seq2seq.seq2seq.metrics import metric_specs

#from g2p_seq2seq.seq2seq.experiment import Experiment
#from g2p_seq2seq.seq2seq.estimator import Estimator

from IPython.core.debugger import Tracer


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
  #_BUCKETS = [(5, 10), (10, 15), (40, 50)]
  _BUCKETS = [6, 10, 50]
#  _METRICS_DEFAULT_PARAMS = """
#- {separator: ' '}
#- {postproc_fn: seq2seq.data.postproc.strip_bpe}"""

  def __init__(self, model_dir):#, mode = 'g2p'):
    """Initialize model directory."""
    self.model_dir = model_dir
    self.metrics_default_params = """
- {separator: ' '}
- {postproc_fn: seq2seq.data.postproc.strip_bpe}"""

  def load_decode_model(self):
    """Load G2P model and initialize or load parameters in session."""
    if not os.path.exists(os.path.join(self.model_dir, 'checkpoint')):
      raise RuntimeError("Model not found in %s" % self.model_dir)

    self.batch_size = 1 # We decode one word at a time.
    # Load vocabularies
    print("Loading vocabularies from %s" % self.model_dir)
    self.gr_vocab = data_utils.load_vocabulary(os.path.join(self.model_dir,
                                                            "vocab.grapheme"))
    self.ph_vocab = data_utils.load_vocabulary(os.path.join(self.model_dir,
                                                            "vocab.phoneme"))
    #if self.mode == 'g2p':
    self.rev_ph_vocab =\
      data_utils.load_vocabulary(os.path.join(self.model_dir, "vocab.phoneme"),
                                 reverse=True)
    #else:
    #  self.rev_gr_vocab =\
    #    data_utils.load_vocabulary(os.path.join(self.model_dir, "vocab.grapheme"),
    #                              reverse=True)

    tasks_ = """
    - class: DecodeText"""

    if isinstance(tasks_, string_types):
      tasks_ = _maybe_load_yaml(tasks_)

    inp_pipeline = """
    class: ParallelTextInputPipeline
    params:
      source_files:
        - /home/nurtas/data/g2p/cmudict-exp/initial_data/test.grapheme"""

    if isinstance(inp_pipeline, string_types):
      inp_pipeline = _maybe_load_yaml(inp_pipeline)


    input_pipeline_infer = input_pipeline.make_input_pipeline_from_def(
      inp_pipeline, mode=tf.contrib.learn.ModeKeys.INFER,
      shuffle=False, num_epochs=1)

    # Load saved training options
    train_options = training_utils.TrainOptions.load(self.model_dir)

    # Create the model
    model_cls = locate(train_options.model_class) or \
      getattr(models, train_options.model_class)
    model_params = train_options.model_params
    model_params = _deep_merge_dict(
        model_params, _maybe_load_yaml({}))
    model = model_cls(
        params=model_params,
        mode=tf.contrib.learn.ModeKeys.INFER)

    # Load inference tasks
    hooks = []
    for tdict in tasks_:
      if not "params" in tdict:
        tdict["params"] = {}
      task_cls = locate(tdict["class"]) or getattr(tasks, tdict["class"])
      task = task_cls(tdict["params"])
      hooks.append(task)

    # Create the graph used for inference
    predictions, _, _ = create_inference_graph(
        model=model,
        input_pipeline=input_pipeline_infer,
        batch_size=self.batch_size)

    saver = tf.train.Saver()
    checkpoint_path = tf.train.latest_checkpoint(self.model_dir)

    def session_init_op(_scaffold, sess):
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Restored model from %s", checkpoint_path)

    scaffold = tf.train.Scaffold(init_fn=session_init_op)
    session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold)
    with tf.train.MonitoredSession(
        session_creator=session_creator,
        hooks=hooks) as sess:

      # Run until the inputs are exhausted
      while not sess.should_stop():
        sess.run([])


  def prepare_data(self, params, train_path, valid_path, test_path):
    """Prepare train/validation/test sets. Create or load vocabularies."""
    self.params = params
    # Prepare data.
    self.buckets = "5,10,50"

    # Parse YAML FLAGS
    self.params.hooks = _maybe_load_yaml(self.params.hooks)
    self.params.metrics = _maybe_load_yaml(self.params.metrics)
    self.params.model_params = _maybe_load_yaml(self.params.model_params)
    self.params.input_pipeline = _maybe_load_yaml(self.params.input_pipeline)
    self.metrics_default_params = _maybe_load_yaml(self.metrics_default_params)

    # Load flags from config file

    tf.logging.info('default_params:\n%s', yaml.dump(self.metrics_default_params))
    tf.logging.info('hooks:\n%s', yaml.dump(self.params.hooks))
    tf.logging.info('metrics:\n%s', yaml.dump(self.params.metrics))
    tf.logging.info('model_params:\n%s', yaml.dump(self.params.model_params))
    tf.logging.info('input_pipeline:\n%s', yaml.dump(self.params.input_pipeline))


  def create_experiment(self, output_dir):
    """
    Creates a new Experiment instance.

    Args:
      output_dir: Output directory for model checkpoints and summaries.
    """

    config = run_config.RunConfig(
        tf_random_seed=None,
        save_checkpoints_secs=self.params.save_checkpoints_secs,
        save_checkpoints_steps=self.params.save_checkpoints_steps,
        keep_checkpoint_max=1,
        keep_checkpoint_every_n_hours=None,
        gpu_memory_fraction=1.0)
    config.tf_config.gpu_options.allow_growth = False
    config.tf_config.log_device_placement = False

    train_options = training_utils.TrainOptions(
        model_class='AttentionSeq2Seq',
        model_params=self.params.model_params)
    # On the main worker, save training options
    if config.is_chief:
      gfile.MakeDirs(output_dir)
      train_options.dump(output_dir)

    tf.logging.info('buckets: %s', self.buckets)
    self.buckets = self.buckets.split(',')
    bucket_boundaries = list(map(int, self.buckets))


    # Training data input pipeline
    train_input_pipeline = input_pipeline.make_input_pipeline_from_def(
        def_dict=self.params.input_pipeline,
        mode=tf.contrib.learn.ModeKeys.TRAIN)

    # Create training input function
    train_input_fn = training_utils.create_input_fn(
        pipeline=train_input_pipeline,
        batch_size=64,
        bucket_boundaries=None,#bucket_boundaries,
        scope="train_input_fn")

    # Development data input pipeline
    dev_input_pipeline = input_pipeline.make_input_pipeline_from_def(
        def_dict=self.params.input_pipeline,
        mode=tf.contrib.learn.ModeKeys.EVAL,
        shuffle=False, num_epochs=1)

    # Create eval input function
    eval_input_fn = training_utils.create_input_fn(
        pipeline=dev_input_pipeline,
        batch_size=64,
        allow_smaller_final_batch=True,
        scope="dev_input_fn")


    def model_fn(features, labels, params, mode):
      """Builds the model graph"""
      model = _create_from_dict({
          "class": train_options.model_class,
          "params": train_options.model_params
      }, models, mode=mode)
      return model(features, labels, params)

    estimator = tf.contrib.learn.Estimator(
    #estimator = Estimator(
        model_fn=model_fn,
        model_dir=output_dir,
        config=config,
        params=self.params.model_params)

    # Create hooks
    train_hooks = []
    for dict_ in self.params.hooks:
      hook = _create_from_dict(
          dict_, hooks,
          model_dir=estimator.model_dir,
          run_config=config)
      train_hooks.append(hook)

    # Create metrics
    eval_metrics = {}

    for dict_ in self.params.metrics:
      metric = _create_from_dict(dict_, metric_specs)
      eval_metrics[metric.name] = metric

    experiment = tf.contrib.learn.Experiment(
    #experiment = Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        min_eval_frequency=self.params.eval_every_n_steps,
        train_steps=self.params.max_steps,
        eval_steps=None,
        eval_metrics=eval_metrics,
        train_monitors=train_hooks)

    return experiment


  def train(self):
    """Train a gr->ph translation model using G2P data."""

    learn_runner.run(
        experiment_fn=self.create_experiment,
        output_dir=self.model_dir,
        schedule="continuous_train_and_eval")#None)

    print('Training done.')

