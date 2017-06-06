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
import sys, os
import time

sys.path.insert(1, './g2p_seq2seq')

import numpy as np
import tensorflow as tf

from g2p_seq2seq import data_utils
from g2p_seq2seq import seq2seq_model

from six import text_type, string_types

from pydoc import locate
import yaml

from seq2seq import tasks, models
from seq2seq.configurable import _maybe_load_yaml, _deep_merge_dict, _create_from_dict
from seq2seq_lib.data import input_pipeline
from seq2seq.inference import create_inference_graph
from seq2seq.training import utils as training_utils

from seq2seq.training import hooks
from seq2seq.metrics import metric_specs

from tensorflow.contrib.learn.python.learn import learn_runner
#from tensorflow_lib import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
#from tensorflow_lib import run_config
from tensorflow import gfile

#from tensorflow_lib.experiment import Experiment
#from tensorflow_lib.estimator import Estimator


class G2PModel(object):
  """Grapheme-to-Phoneme translation model class.

  Constructor parameters:
    model_dir: Working directory where Model will (saved in)/(loaded from).

  Attributes:
    params: Instance of g2p_seq2seq.params.Params class with all required
      parameters;
    _BUCKET_BOUNDARIES: Buckets input sequences according to these length.
      A comma-separated list of sequence length buckets, e.g. "10,20,30" would
      result in 4 buckets: <10, 10-20, 20-30, >30. None disabled bucketing. 
    hooks: YAML configuration string for the training metrics to use.
    train: Train method.
    decode: Decode file method.
  """
  # We use a number of buckets and pad to the closest one for efficiency.
  # See seq2seq_model.Seq2SeqModel for details of how they work.
  _BUCKET_BOUNDARIES = [6, 10]
  tf.logging.set_verbosity(tf.logging.INFO)

  def __init__(self, model_dir):
    """Initialize model directory."""
    self.model_dir = model_dir

  def load_decode_model(self, params):
    """Load G2P model and initialize or load parameters in session."""
    if not os.path.exists(os.path.join(self.model_dir, 'checkpoint')):
      raise RuntimeError("Model not found in %s" % self.model_dir)

    self.params = params

    if isinstance(self.params.tasks, string_types):
      self.params.tasks = _maybe_load_yaml(self.params.tasks)

    if isinstance(self.params.input_pipeline, string_types):
      self.params.input_pipeline = _maybe_load_yaml(self.params.input_pipeline)

    input_pipeline_infer = input_pipeline.make_input_pipeline_from_def(
      self.params.input_pipeline, mode=tf.contrib.learn.ModeKeys.INFER,
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
    self.hooks = []
    for tdict in self.params.tasks:
      if not "params" in tdict:
        tdict["params"] = {}
      task_cls = locate(tdict["class"]) or getattr(tasks, tdict["class"])
      task = task_cls(tdict["params"])
      self.hooks.append(task)
    # Create the graph used for inference
    predictions, _, _ = create_inference_graph(
        model=model,
        input_pipeline=input_pipeline_infer,
        batch_size=self.params.batch_size)

    saver = tf.train.Saver()
    checkpoint_path = tf.train.latest_checkpoint(self.model_dir)

    def session_init_op(_scaffold, sess):
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Restored model from %s", checkpoint_path)

    scaffold = tf.train.Scaffold(init_fn=session_init_op)
    session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold)
    with tf.train.MonitoredSession(
        session_creator=session_creator,
        hooks=self.hooks) as sess:

      # Run until the inputs are exhausted
      while not sess.should_stop():
        print('Sess Run')
        sess.run([])


  def load_train_model(self, params):
    """Prepare train/validation/test sets. Create or load vocabularies."""
    self.params = params
    # Parse YAML FLAGS
    self.params.hooks = _maybe_load_yaml(self.params.hooks)
    self.params.metrics = _maybe_load_yaml(self.params.metrics)
    self.params.model_params = _maybe_load_yaml(self.params.model_params)
    self.params.input_pipeline = _maybe_load_yaml(self.params.input_pipeline)

    # Load flags from config file

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
        keep_checkpoint_every_n_hours=4,
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

    # Training data input pipeline
    train_input_pipeline = input_pipeline.make_input_pipeline_from_def(
        def_dict=self.params.input_pipeline,
        mode=tf.contrib.learn.ModeKeys.TRAIN)

    tf.logging.info('bucket boundaries: %s', self._BUCKET_BOUNDARIES)

    # Create training input function
    train_input_fn = training_utils.create_input_fn(
        pipeline=train_input_pipeline,
        batch_size=self.params.batch_size,
        bucket_boundaries=self._BUCKET_BOUNDARIES,
        scope="train_input_fn")

    # Development data input pipeline
    dev_input_pipeline = input_pipeline.make_input_pipeline_from_def(
        def_dict=self.params.input_pipeline,
        mode=tf.contrib.learn.ModeKeys.EVAL,
        shuffle=False, num_epochs=1)

    # Create eval input function
    eval_input_fn = training_utils.create_input_fn(
        pipeline=dev_input_pipeline,
        batch_size=self.params.batch_size,
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
        schedule=None)

    print('Training done.')

"""
  def decode(self):
    saver = tf.train.Saver()
    checkpoint_path = tf.train.latest_checkpoint(self.model_dir)

    def session_init_op(_scaffold, sess):
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Restored model from %s", checkpoint_path)

    scaffold = tf.train.Scaffold(init_fn=session_init_op)
    session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold)
    with tf.train.MonitoredSession(
        session_creator=session_creator,
        hooks=self.hooks) as sess:

      # Run until the inputs are exhausted
      while not sess.should_stop():
        print('Sess Run')
        sess.run([])
"""
