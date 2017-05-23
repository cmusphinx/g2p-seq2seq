
import random

class BatchReader(object):

  def __init__(self):
    self.num_records_produced_ = 0

  def read(self, data):
    if self.num_records_produced_ < 32:
      self.num_records_produced_ += 1
      return random.choice(data)
