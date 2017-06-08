
import numpy as np
import random
import threading
import tensorflow as tf
from IPython.core.debugger import Tracer
import sys


# Special vocabulary symbols.
PAD_ID = 0
GO_ID = 1

class GeneratorRunner(object):
  """Custom runner that that runs an generator in a thread and enqueues
     the outputs."""

  def __init__(self, generator, placeholders, enqueue_op, close_op):
    self._generator = generator
    self._placeholders = placeholders
    self._enqueue_op = enqueue_op
    self._close_op = close_op

  def _run(self, sess, coord):
    try:
      while not coord.should_stop():
        try:
          values = next(self._generator)
          assert len(values) == len(self._placeholders), \
            'generator values and placeholders must have the same length'
          feed_dict = {placeholder: value \
            for placeholder, value in zip(self._placeholders, values)}
          sess.run(self._enqueue_op, feed_dict=feed_dict)
        except (StopIteration, tf.errors.OutOfRangeError):
          try:
            sess.run(self._close_op)
          except Exception:
            pass
          return
    except Exception as ex:
      if coord:
        coord.request_stop(ex)
      else:
        raise

  def create_threads(self, sess, coord=None, daemon=False, start=False):
    """Called by `start_queue_runners`."""
    thread = threading.Thread(target=self._run, args=(sess, coord))
    if coord:
      coord.register_thread(thread)
    if daemon:
      thread.daemon = True
    if start:
      thread.start()
    return [thread]


def read_batch_generator(filename_queue, generator, dtypes, shapes, batch_size,
                         queue_capacity=10000, allow_smaller_final_batch=False):
  """Reads values from an generator, queues, and batches."""

  assert len(dtypes) == len(shapes), 'dtypes and shapes must have the same length'

  queue = tf.FIFOQueue(capacity=queue_capacity, dtypes=dtypes, shapes=shapes)

  placeholders = [tf.placeholder(dtype, shape)
                  for dtype, shape in zip(dtypes, shapes)]

  enqueue_op = queue.enqueue(placeholders)
  close_op = queue.close(cancel_pending_enqueues=True)

  qr = GeneratorRunner(generator, placeholders, enqueue_op, close_op)
  tf.train.add_queue_runner(qr)

  if allow_smaller_final_batch:
    return queue.dequeue_up_to(batch_size)
  else:
    return queue.dequeue_many(batch_size)


def gen(data, num_epochs=1):
  for _ in range(num_epochs):
    for line_inx, line in enumerate(data):
      if ((line_inx + 1) % 1 == 0):
        yield [line]


def F(x):
  return x


class BatchReader(object):

  def __init__(self, data):
    self.data = data
    self.num_records_produced_ = 0
    self.batch_size = 32

  def read2(self, filename_queue):
    if self.num_records_produced_ < 32:
      self.num_records_produced_ += 1
      yield random.choice(self.data)

  def read(self, session):
    encoder_inputs, decoder_inputs, target_weights = self.get_batch()#, bucket_id)
    _, step_loss, _ = self.step(session, encoder_inputs,
                                decoder_inputs, target_weights,
                                #bucket_id,
                                False)
    #if self.num_records_produced_ < 32:
    #  self.num_records_produced_ += 1
    #  return random.choice(self.data)

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           #bucket_id,
           forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = 50, 50#self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    return input_feed, output_feed
    #outputs = session.run(output_feed, input_feed)
    #if not forward_only:
    #  return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    #else:
    #  return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.


  def get_batch(self):
    encoder_size, decoder_size = 50, 50#self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      #encoder_input, decoder_input = random.choice(self.data)
      data_line = random.choice(self.data)
      data_line_split = data_line.split(' ')
      encoder_input = [c for c in data_line_split[0]]
      decoder_input = data_line_split[1:]

      # Encoder inputs are padded and then reversed.
      encoder_pad = [PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([GO_ID] + decoder_input +
                            [PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    #Tracer()()
    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

