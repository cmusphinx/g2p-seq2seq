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

"""Utilities for G2P trainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner

from tensor2tensor.utils import devices
from tensor2tensor.utils import registry
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("save_checkpoints_steps", None,
    """Save checkpoints every this many steps. Default=None means let
    tensorflow.contrib.learn.python.learn decide, which saves checkpoints
    every 600 seconds.""")


def add_problem_hparams(hparams, problem):
  """Add problem hparams for the problems."""
  p_hparams = problem.get_hparams(hparams)
  hparams.problem = problem
  hparams.problem_hparams = p_hparams


def create_experiment_fn(params, problem_instance):
  use_validation_monitor = (params.schedule in
                            ["train_and_evaluate", "continuous_train_and_eval"]
                            and params.local_eval_frequency)
  return create_experiment_func(
      model_name=params.model_name,
      params=params,
      problem_instance=problem_instance,
      data_dir=os.path.expanduser(params.data_dir_name),
      train_steps=params.train_steps,
      eval_steps=params.eval_steps,
      min_eval_frequency=params.local_eval_frequency,
      schedule=params.schedule,
      export=params.export_saved_model,
      decode_hparams=decoding.decode_hparams(params.decode_hparams),
      use_tfdbg=params.tfdbg,
      use_dbgprofile=params.dbgprofile,
      use_validation_monitor=use_validation_monitor,
      eval_early_stopping_steps=params.eval_early_stopping_steps,
      eval_early_stopping_metric=params.eval_early_stopping_metric,
      eval_early_stopping_metric_minimize=\
        params.eval_early_stopping_metric_minimize,
      use_tpu=params.use_tpu)


def create_experiment_func(*args, **kwargs):
  """Wrapper for canonical experiment_fn. See create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(run_config, hparams, *args, **kwargs)

  return experiment_fn


def create_experiment(run_config,
                      hparams,
                      model_name,
                      params,
                      problem_instance,
                      data_dir,
                      train_steps,
                      eval_steps,
                      min_eval_frequency=2000,
                      eval_throttle_seconds=600,
                      schedule="train_and_evaluate",
                      export=False,
                      decode_hparams=None,
                      use_tfdbg=False,
                      use_dbgprofile=False,
                      use_validation_monitor=False,
                      eval_early_stopping_steps=None,
                      eval_early_stopping_metric=None,
                      eval_early_stopping_metric_delta=None,
                      eval_early_stopping_metric_minimize=True,
                      autotune=False,
                      use_tpu=False):
  """Create Experiment."""
  # HParams
  hparams.add_hparam('model_dir', params.model_dir)
  hparams.add_hparam("data_dir", data_dir)
  hparams.add_hparam("train_steps", train_steps)
  hparams.add_hparam("eval_steps", eval_steps)
  hparams.add_hparam("schedule", schedule)
  add_problem_hparams(hparams, problem_instance)

  # Estimator
  estimator = trainer_lib.create_estimator(
      model_name,
      hparams,
      run_config,
      schedule=schedule,
      decode_hparams=decode_hparams,
      use_tpu=use_tpu)

  # Input fns from Problem
  problem = hparams.problem
  train_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.TRAIN, hparams)
  eval_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.EVAL, hparams)

  # Export
  if export:
    tf.logging.warn("Exporting from the trainer is deprecated. "
                    "See serving/export.py.")

  # Hooks
  validation_monitor_kwargs = dict(
      input_fn=eval_input_fn,
      eval_steps=eval_steps,
      every_n_steps=min_eval_frequency,
      early_stopping_rounds=eval_early_stopping_steps,
      early_stopping_metric=eval_early_stopping_metric,
      early_stopping_metric_minimize=eval_early_stopping_metric_minimize)
  dbgprofile_kwargs = {"output_dir": run_config.model_dir}
  early_stopping_kwargs = dict(
      events_dir=os.path.join(run_config.model_dir, "eval_continuous"),
      tag=eval_early_stopping_metric,
      num_plateau_steps=eval_early_stopping_steps,
      plateau_decrease=eval_early_stopping_metric_minimize,
      plateau_delta=eval_early_stopping_metric_delta,
      every_n_steps=min_eval_frequency)

  # In-process eval (and possible early stopping)
  if schedule == "continuous_train_and_eval" and min_eval_frequency:
    tf.logging.warn("ValidationMonitor only works with "
                    "--schedule=train_and_evaluate")
  use_validation_monitor = (
      schedule == "train_and_evaluate" and min_eval_frequency)
  # Distributed early stopping
  local_schedules = ["train_and_evaluate", "continuous_train_and_eval"]
  use_early_stopping = (
      schedule not in local_schedules and eval_early_stopping_steps)
  train_hooks, eval_hooks = trainer_lib.create_hooks(
      use_tfdbg=use_tfdbg,
      use_dbgprofile=use_dbgprofile,
      dbgprofile_kwargs=dbgprofile_kwargs,
      use_validation_monitor=use_validation_monitor,
      validation_monitor_kwargs=validation_monitor_kwargs,
      use_early_stopping=use_early_stopping,
      early_stopping_kwargs=early_stopping_kwargs)
  train_hooks += t2t_model.T2TModel.get_train_hooks(model_name)
  eval_hooks += t2t_model.T2TModel.get_eval_hooks(model_name)

  train_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
      train_hooks, estimator)
  eval_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
      eval_hooks, estimator)

  train_spec = tf.estimator.TrainSpec(
      train_input_fn, max_steps=train_steps, hooks=train_hooks)
  eval_spec = tf.estimator.EvalSpec(
      eval_input_fn,
      steps=eval_steps,
      hooks=eval_hooks,
      start_delay_secs=0 if hparams.schedule == "evaluate" else 120,
      throttle_secs=eval_throttle_seconds)

  if autotune:
    hooks_kwargs = {"train_monitors": train_hooks, "eval_hooks": eval_hooks}
    return tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=train_steps,
        eval_steps=eval_steps,
        min_eval_frequency=min_eval_frequency,
        train_steps_per_iteration=min(min_eval_frequency, train_steps),
        eval_delay_secs=0 if schedule == "evaluate" else 120,
        **hooks_kwargs if not use_tpu else {})
  return trainer_lib.T2TExperiment(estimator, hparams, train_spec, eval_spec,
                                   use_validation_monitor, decode_hparams)

def create_run_config(hp, params):
  """Create RunConfig"""
  return trainer_lib.create_run_config(
      master=params.master,
      model_dir=params.model_dir,
      iterations_per_loop=params.iterations_per_loop,
      num_shards=params.tpu_num_shards,
      log_device_placement=params.log_device_replacement,
      save_checkpoints_steps=max(params.iterations_per_loop,
                                 params.local_eval_frequency),
      keep_checkpoint_max=params.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=params.keep_checkpoint_every_n_hours,
      num_gpus=params.worker_gpu,
      gpu_order=params.gpu_order,
      shard_to_cpu=params.locally_shard_to_cpu,
      num_async_replicas=params.worker_replicas,
      enable_graph_rewriter=params.experimental_optimize_placement,
      gpu_mem_fraction=params.worker_gpu_memory_fraction,
      no_data_parallelism=params.no_data_parallelism,
      daisy_chain_variables=params.daisy_chain_variables,
      schedule=params.schedule,
      worker_id=params.worker_id,
      worker_job=params.worker_job,
      ps_replicas=params.ps_replicas,
      ps_job=params.ps_job,
      ps_gpu=params.ps_gpu,
      sync=params.sync,
      use_tpu=params.use_tpu)

def save_params(model_dir, hparams):
  """Save customizable model parameters in 'model.params' file.
  """
  params_to_save = {}
  for hp in hparams.split(","):
    param_split = hp.split("=")
    if len(param_split) == 2:
      param_name, param_value = param_split[0], param_split[1]
      params_to_save[param_name] = param_value
    else:
      raise ValueError("HParams line:{} can not be splitted\n"
                       .format(param_split))
  with open(os.path.join(model_dir, "model.params"), "w") as params_file:
    json.dump(params_to_save, params_file)


def load_params(model_dir):
  """Load customizable parameters from 'model.params' file.
  """
  params_file_path = os.path.join(model_dir, "model.params")
  if os.path.exists(params_file_path):
    model_params = json.load(open(params_file_path))
    hparams = ""
    for hp, hp_value in model_params.items():
      if hparams:
        hparams += ","
      hparams += hp + "=" + hp_value
    return hparams
  raise Exception("File {} not exists.".format(params_file_path))
