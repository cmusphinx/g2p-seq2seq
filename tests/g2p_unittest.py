import os, sys, inspect

sys.path.insert(0, '../g2p_seq2seq')

import unittest
import g2p
import data_utils

class TestG2P(unittest.TestCase):

  def test_train(self):
    model_dir = "../tests/models/train"
    with g2p.tf.Graph().as_default():
      g2p_train_model = g2p.G2PModel(model_dir)
      train_path = "../tests/data/toydict.train"
      valid_path = "../tests/data/toydict.test"
      test_path = "../tests/data/toydict.test"
      g2p_params = g2p.G2P_Params()
      g2p_params.steps_per_checkpoint = 1
      g2p_params.max_steps = 1
      g2p_params.num_layers = 1
      g2p_params.size = 2
      g2p_params.write_model = False
      g2p_train_model.train(g2p_params, train_path, valid_path, test_path)


  def test_evaluate(self):
    model_dir = "../tests/models/decode"
    with g2p.tf.Graph().as_default():
      g2p_evaluate_model = g2p.G2PModel(model_dir)
      test_lines = g2p.codecs.open("../tests/data/toydict.test", "r", "utf-8").readlines()
      g2p_evaluate_model.evaluate(test_lines)
      test_dic = data_utils.collect_pronunciations(test_lines)
      errors = g2p_evaluate_model.calc_error(test_dic)
      self.assertEqual(round(float(errors)/len(test_dic), 3), 0.667)


  def test_decode(self):
    model_dir = "../tests/models/decode"
    with g2p.tf.Graph().as_default():
      g2p_decode_model = g2p.G2PModel(model_dir)
      decode_lines = g2p.codecs.open("../tests/data/toydict.graphemes",
                                     "r", "utf-8").readlines()
      output_file = g2p.codecs.open("../tests/data/decode_output", "w", "utf-8")
      g2p_decode_model.decode(decode_lines, output_file)
      output_lines = g2p.codecs.open("../tests/data/decode_output",
                                     "r", "utf-8").readlines()
      self.assertEqual(output_lines[0].strip(), u'cabcabbacab C')
      self.assertEqual(output_lines[1].strip(), u'abcabac B C B')
      self.assertEqual(output_lines[2].strip(), u'a B')


suite = unittest.TestLoader().loadTestsFromTestCase(TestG2P)
unittest.TextTestRunner(verbosity=2).run(suite)
