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

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner

from tensor2tensor.utils import devices
from tensor2tensor.utils import registry
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("save_checkpoints_steps", None,
    """Save checkpoints every this many steps. Default=None means let
    tensorflow.contrib.learn.python.learn decide, which saves checkpoints
    every 600 seconds.""")


def add_problem_hparams(hparams, problem_name, model_dir, problem_instance):
  """Add problem hparams for the problems."""
  hparams.problems = []
  hparams.problem_instances = []
  p_hparams = problem_instance.get_hparams(hparams)
  hparams.problem_instances.append(problem_instance)
  hparams.problems.append(p_hparams)


def create_experiment_fn(params, problem_instance):
  use_validation_monitor = (params.schedule in
                            ["train_and_evaluate", "continuous_train_and_eval"]
                            and params.local_eval_frequency)
  #return tpu_trainer_lib.create_experiment_fn(
  return create_experiment_func(
      model_name=params.model_name,
      #problem_name=params.problem_name,#get_problem_name(),
      params=params,
      problem_instance=problem_instance,
      data_dir=os.path.expanduser(params.data_dir_name),#FLAGS.data_dir),
      train_steps=params.train_steps,#FLAGS.train_steps,
      eval_steps=params.eval_steps,#FLAGS.eval_steps,
      min_eval_frequency=params.local_eval_frequency,#FLAGS.local_eval_frequency,
      schedule=params.schedule,#FLAGS.schedule,
      export=params.export_saved_model,#FLAGS.export_saved_model,
      decode_hparams=decoding.decode_hparams(params.decode_hparams),#FLAGS.decode_hparams),
      use_tfdbg=params.tfdbg,#FLAGS.tfdbg,
      use_dbgprofile=params.dbgprofile,#FLAGS.dbgprofile,
      use_validation_monitor=use_validation_monitor,
      eval_early_stopping_steps=params.eval_early_stopping_steps,#FLAGS.eval_early_stopping_steps,
      eval_early_stopping_metric=params.eval_early_stopping_metric,#FLAGS.eval_early_stopping_metric,
      eval_early_stopping_metric_minimize=params.eval_early_stopping_metric_minimize,#FLAGS.
      #eval_early_stopping_metric_minimize,
      use_tpu=params.use_tpu)#FLAGS.use_tpu)


def create_experiment_func(*args, **kwargs):
  """Wrapper for canonical experiment_fn. See create_experiment."""

  def experiment_fn(run_config, hparams):
    return create_experiment(run_config, hparams, *args, **kwargs)

  return experiment_fn


def create_experiment(run_config,
                      hparams,
                      model_name,
                      params,
                      problem_instance,#problem_name,
                      data_dir,
                      train_steps,
                      eval_steps,
                      min_eval_frequency=2000,
                      schedule="train_and_evaluate",
                      export=False,
                      decode_hparams=None,
                      use_tfdbg=False,
                      use_dbgprofile=False,
                      use_validation_monitor=False,
                      eval_early_stopping_steps=None,
                      eval_early_stopping_metric=None,
                      eval_early_stopping_metric_minimize=True,
                      use_tpu=False):
  """Create Experiment."""
  # HParams
  hparams.add_hparam("data_dir", data_dir)
  add_problem_hparams(hparams, params.problem_name, params.model_dir, problem_instance)

  # Estimator
  #estimator = tpu_trainer_lib.create_estimator(
  estimator = trainer_lib.create_estimator(
      model_name,
      hparams,
      run_config,
      schedule=schedule,
      decode_hparams=decode_hparams,
      use_tpu=use_tpu)

  # Input fns from Problem
  problem = hparams.problem_instances[0]
  train_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.TRAIN, hparams)
  eval_input_fn = problem.make_estimator_input_fn(
      tf.estimator.ModeKeys.EVAL, hparams)

  # Export
  export_strategies = export and [create_export_strategy(problem, hparams)]

  # Hooks
  hooks_kwargs = {}
  if not use_tpu:
    dbgprofile_kwargs = {"output_dir": run_config.model_dir}
    validation_monitor_kwargs = dict(
        input_fn=eval_input_fn,
        eval_steps=eval_steps,
        every_n_steps=min_eval_frequency,
        early_stopping_rounds=eval_early_stopping_steps,
        early_stopping_metric=eval_early_stopping_metric,
        early_stopping_metric_minimize=eval_early_stopping_metric_minimize)
    #train_monitors, eval_hooks = tpu_trainer_lib.create_hooks(
    train_monitors, eval_hooks = trainer_lib.create_hooks(
        use_tfdbg=use_tfdbg,
        use_dbgprofile=use_dbgprofile,
        dbgprofile_kwargs=dbgprofile_kwargs,
        use_validation_monitor=use_validation_monitor,
        validation_monitor_kwargs=validation_monitor_kwargs)
    hooks_kwargs = {"train_monitors": train_monitors, "eval_hooks": eval_hooks}

  # Experiment
  return tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps=train_steps,
      eval_steps=eval_steps,
      min_eval_frequency=min_eval_frequency,
      train_steps_per_iteration=min(min_eval_frequency, train_steps),
      export_strategies=export_strategies,
      **hooks_kwargs)


def create_run_config(hp, params):
  #return tpu_trainer_lib.create_run_config(
  return trainer_lib.create_run_config(
      model_dir=params.model_dir,#os.path.expanduser(FLAGS.output_dir),
      master=params.master,#FLAGS.master,
      iterations_per_loop=params.iterations_per_loop,#FLAGS.iterations_per_loop,
      num_shards=params.tpu_num_shards,#FLAGS.tpu_num_shards,
      log_device_placement=params.log_device_replacement,#FLAGS.log_device_placement,
      save_checkpoints_steps=max(params.iterations_per_loop,#FLAGS.iterations_per_loop,
                                 params.local_eval_frequency),#FLAGS.local_eval_frequency),
      keep_checkpoint_max=params.keep_checkpoint_max,#FLAGS.keep_checkpoint_max,
      keep_checkpoint_every_n_hours=params.keep_checkpoint_every_n_hours,#FLAGS.keep_checkpoint_every_n_hours,
      num_gpus=params.worker_gpu,#FLAGS.worker_gpu,
      gpu_order=params.gpu_order,#FLAGS.gpu_order,
      shard_to_cpu=params.locally_shard_to_cpu,#FLAGS.locally_shard_to_cpu,
      num_async_replicas=params.worker_replicas,#FLAGS.worker_replicas,
      gpu_mem_fraction=params.worker_gpu_memory_fraction,#FLAGS.worker_gpu_memory_fraction,
      enable_graph_rewriter=params.experimental_optimize_placement,#FLAGS.experimental_optimize_placement,
      use_tpu=params.use_tpu,#FLAGS.use_tpu,
      schedule=params.schedule,#FLAGS.schedule,
      no_data_parallelism=params.no_data_parallelism,#hp.no_data_parallelism,
      daisy_chain_variables=params.daisy_chain_variables,#hp.daisy_chain_variables,
      ps_replicas=params.ps_replicas,#FLAGS.ps_replicas,
      ps_job=params.ps_job,#FLAGS.ps_job,
      ps_gpu=params.ps_gpu,#FLAGS.ps_gpu,
      sync=params.sync,#FLAGS.sync,
      worker_id=params.worker_id,#FLAGS.worker_id,
      worker_job=params.worker_job)#FLAGS.worker_job)

