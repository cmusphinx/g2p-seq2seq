
import unittest
import shutil
from g2p_seq2seq import g2p
from g2p_seq2seq import data_utils

class TestG2P(unittest.TestCase):

  def test_train(self):
    model_dir = "tests/models/train"
    with g2p.tf.Graph().as_default():
      g2p_model = g2p.G2PModel(model_dir, 'p2g')
      train_path = "tests/data/toydict.train"
      valid_path = "tests/data/toydict.test"
      test_path = "tests/data/toydict.test"
      g2p_params = g2p.TrainingParams()
      g2p_params.steps_per_checkpoint = 1
      g2p_params.max_steps = 1
      g2p_params.num_layers = 1
      g2p_params.size = 2
      g2p_params.mode = 'p2g'
      g2p_model.prepare_data(train_path, valid_path, test_path)
      g2p_model.create_train_model(g2p_params)
      g2p_model.train()
    shutil.rmtree(model_dir)


  def test_decode(self):
    model_dir = "tests/models/decode"
    with g2p.tf.Graph().as_default():
      g2p_model = g2p.G2PModel(model_dir, 'p2g')
      g2p_model.load_decode_model()
      with open("tests/data/toydict.phonemes") as f:
        decode_lines = f.readlines()
      grapheme_lines = g2p_model.decode(decode_lines)
      self.assertEqual(grapheme_lines[0].strip(), u'b')
      self.assertEqual(grapheme_lines[1].strip(), u'a')
      self.assertEqual(grapheme_lines[2].strip(), u'a')
