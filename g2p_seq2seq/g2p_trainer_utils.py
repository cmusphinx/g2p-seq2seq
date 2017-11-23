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

import tensorflow as tf

from tensor2tensor.utils import devices
from tensor2tensor.utils import input_fn_builder
from tensor2tensor.utils import model_builder
from tensor2tensor.utils import registry
from tensor2tensor.utils import decoding

FLAGS = tf.app.flags.FLAGS


def add_problem_hparams(hparams, problem_name, model_dir):
  """Add problem hparams for the problems."""
  hparams.problems = []
  hparams.problem_instances = []
  #for problem_name in problems.split("-"):
  problem = registry._PROBLEMS[problem_name](model_dir)##registry.problem(problem_name)
  p_hparams = problem.get_hparams(hparams)

  hparams.problem_instances.append(problem)
  hparams.problems.append(p_hparams)


def create_experiment_components(problem_name, data_dir, model_name, hparams,
                                 run_config, model_dir):
  """Constructs and returns Estimator and train/eval input functions."""
  tf.logging.info("Creating experiment, storing model files in %s",
                  run_config.model_dir)

  add_problem_hparams(hparams, problem_name, model_dir)#FLAGS.problems)

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
      batch_size=batch_size)

  eval_input_fn = input_fn_builder.build_input_fn(
      mode=tf.estimator.ModeKeys.EVAL,
      hparams=hparams,
      data_dir=data_dir,
      num_datashards=num_datashards,
      worker_replicas=FLAGS.worker_replicas,
      worker_id=FLAGS.worker_id,
      dataset_split="test" if FLAGS.eval_use_test_set else None)

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
