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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensor2tensor.utils import devices
from tensor2tensor.utils import input_fn_builder
from tensor2tensor.utils import model_builder
from tensor2tensor.utils import registry
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_utils

from tensorflow.contrib.learn.python.learn import learn_runner

from IPython.core.debugger import Tracer

FLAGS = tf.app.flags.FLAGS


def add_problem_hparams(hparams, problem_name, model_dir):
  """Add problem hparams for the problems."""
  hparams.problems = []
  hparams.problem_instances = []
  problem = registry._PROBLEMS[problem_name](model_dir)
  p_hparams = problem.get_hparams(hparams)

  hparams.problem_instances.append(problem)
  hparams.problems.append(p_hparams)


def create_experiment_components(problem_name, data_dir, model_name, hparams,
                                 run_config, model_dir,
                                 train_preprocess_file_path=None,
                                 dev_preprocess_file_path=None):
  """Constructs and returns Estimator and train/eval input functions."""
  tf.logging.info("Creating experiment, storing model files in %s",
                  run_config.model_dir)

  add_problem_hparams(hparams, problem_name, model_dir)

  # hparams batch_size is used as minibatch size instead of tokens in batch
  batch_size = (hparams.use_fixed_batch_size and hparams.batch_size) or None
  num_datashards = devices.data_parallelism().n
  train_input_fn = input_fn_builder.build_input_fn(
      mode=tf.estimator.ModeKeys.TRAIN,
      hparams=hparams,
      data_dir=data_dir,
      num_datashards=num_datashards,
      worker_replicas=FLAGS.worker_replicas,
      worker_id=FLAGS.worker_id,
      batch_size=batch_size,
      dataset_split=train_preprocess_file_path)

  eval_input_fn = input_fn_builder.build_input_fn(
      mode=tf.estimator.ModeKeys.EVAL,
      hparams=hparams,
      data_dir=data_dir,
      num_datashards=num_datashards,
      worker_replicas=FLAGS.worker_replicas,
      worker_id=FLAGS.worker_id,
      dataset_split=dev_preprocess_file_path)

  model_fn = model_builder.build_model_fn(
      model_name,
      problem_names=FLAGS.problems.split("-"),
      train_steps=FLAGS.train_steps,
      worker_id=FLAGS.worker_id,
      worker_replicas=FLAGS.worker_replicas,
      eval_run_autoregressive=FLAGS.eval_run_autoregressive,
      decode_hparams=decoding.decode_hparams(FLAGS.decode_hparams))

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=run_config.model_dir,
      params=hparams,
      config=run_config)

  return estimator, {
      tf.estimator.ModeKeys.TRAIN: train_input_fn,
      tf.estimator.ModeKeys.EVAL: eval_input_fn
  }


def make_experiment_fn(problem_name, model_dir, data_dir, model_name,
                       train_steps, eval_steps, train_preprocess_file_path,
                       dev_preprocess_file_path):
  """Returns experiment_fn for learn_runner. Wraps create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(
        problem_name,
        model_dir,
        data_dir,
        model_name=model_name,
        train_steps=train_steps,
        eval_steps=eval_steps,
        hparams=hparams,
        run_config=run_config,
        train_preprocess_file_path=train_preprocess_file_path,
        dev_preprocess_file_path=dev_preprocess_file_path)

  return experiment_fn


def create_experiment(problem_name, model_dir, data_dir, model_name,
                      train_steps, eval_steps, hparams, run_config,
                      train_preprocess_file_path, dev_preprocess_file_path):
  """Create Experiment."""
  estimator, input_fns = create_experiment_components(
      problem_name=problem_name,
      data_dir=data_dir,
      model_name=model_name,
      hparams=hparams,
      run_config=run_config,
      model_dir=model_dir,
      train_preprocess_file_path=train_preprocess_file_path,
      dev_preprocess_file_path=dev_preprocess_file_path)

  train_monitors = []
  eval_hooks = []
  if FLAGS.tfdbg:
    hook = debug.LocalCLIDebugHook()
    train_monitors.append(hook)
    eval_hooks.append(hook)
  if FLAGS.dbgprofile:
    # Recorded traces can be visualized with chrome://tracing/
    # The memory/tensor lifetime is also profiled
    train_monitors.append(
        tf.contrib.hooks.ProfilerHook(
            save_steps=10,
            output_dir=run_config.model_dir,
            show_dataflow=True,
            show_memory=True,
        ))
  if FLAGS.schedule == "train_and_evaluate":
    if FLAGS.local_eval_frequency:
      train_monitors.append(
          tf.contrib.learn.monitors.ValidationMonitor(
              input_fn=input_fns[tf.estimator.ModeKeys.EVAL],
              eval_steps=eval_steps,
              every_n_steps=FLAGS.local_eval_frequency,
              hooks=eval_hooks,
              early_stopping_rounds=FLAGS.eval_early_stopping_steps,
              early_stopping_metric=FLAGS.eval_early_stopping_metric,
              early_stopping_metric_minimize=FLAGS.
              eval_early_stopping_metric_minimize))

  optional_kwargs = {}
  if FLAGS.export_saved_model:
    assert len(hparams.problem_instances) == 1
    problem = hparams.problem_instances[0]
    optional_kwargs["export_strategies"] = [
        make_export_strategy(problem, hparams)
    ]

  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=input_fns[tf.estimator.ModeKeys.TRAIN],
      eval_input_fn=input_fns[tf.estimator.ModeKeys.EVAL],
      train_steps=train_steps,
      eval_steps=eval_steps,
      train_monitors=train_monitors,
      eval_hooks=eval_hooks,
      eval_delay_secs=0,
      **optional_kwargs)


def run(problem_name, model_dir, data_dir, model, output_dir, train_steps, eval_steps, schedule, train_preprocess_file_path, dev_preprocess_file_path):
  """Runs an Estimator locally or distributed.

  Args:
    data_dir: The directory the data can be found in.
    model: The name of the model to use.
    output_dir: The directory to store outputs in.
    train_steps: The number of steps to run training for.
    eval_steps: The number of steps to run evaluation for.
    schedule: (str) The schedule to run. The value here must
      be the name of one of Experiment's methods.
  """
  exp_fn = make_experiment_fn(
      problem_name,
      model_dir,
      data_dir=data_dir,
      model_name=model,
      train_steps=train_steps,
      eval_steps=eval_steps,
      train_preprocess_file_path=train_preprocess_file_path,
      dev_preprocess_file_path=dev_preprocess_file_path)

  # Create hparams and run_config
  run_config = trainer_utils.create_run_config(output_dir)
  hparams = trainer_utils.create_hparams(
    FLAGS.hparams_set, data_dir, passed_hparams=FLAGS.hparams)

  if trainer_utils.is_chief():
    trainer_utils.save_metadata(output_dir, hparams)

  learn_runner.run(
      experiment_fn=exp_fn,
      schedule=schedule,
      run_config=run_config,
      hparams=hparams)
