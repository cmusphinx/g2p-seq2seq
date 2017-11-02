
import unittest
import shutil
from g2p_seq2seq import g2p
from g2p_seq2seq import data_utils
from g2p_seq2seq import params
import inspect, os

#from IPython.core.debugger import Tracer

class TestG2P(unittest.TestCase):

  def test_train(self):
    model_dir = "tests/models/train"
    g2p_model = g2p.G2PModel(model_dir)
    train_path = "tests/data/toydict.train"
    #valid_path = "tests/data/toydict.test"
    #test_path = "tests/data/toydict.test"
    data_utils.create_vocabulary(train_path, model_dir)
    g2p_params = params.Params(model_dir, decode_flag=False)
    g2p_params.max_steps = 100#1
    g2p_model.load_train_model(g2p_params)
    g2p_model.train()
    shutil.rmtree(model_dir)

  #def test_evaluate(self):
  #  model_dir = "tests/models/decode"
  #  with g2p.tf.Graph().as_default():
  #    g2p_model = g2p.G2PModel(model_dir)
  #    g2p_model.load_decode_model()
  #    test_lines = open("tests/data/toydict.test").readlines()
  #    g2p_model.evaluate(test_lines)
  #    test_dic = data_utils.collect_pronunciations(test_lines)
  #    errors = g2p_model.calc_error(test_dic)
  #    self.assertAlmostEqual(float(errors)/len(test_dic), 0.667, places=3)

  def test_decode(self):
    model_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))), "models/decode")
    g2p_model = g2p.G2PModel(model_dir)
    g2p_params = params.Params(model_dir, decode_flag=True)
    #phoneme_lines = g2p_model.load_decode_model(g2p_params)
    g2p_model.load_decode_model(g2p_params)
    decode_lines = open("tests/data/toydict.graphemes").readlines()
    phoneme_lines = []
    for pred in g2p_model.decode():
        phoneme_lines.append(pred)
    self.assertEqual(phoneme_lines[0][0][0], u'SEQUENCE_END')
    self.assertEqual(phoneme_lines[1][0][0], u'SEQUENCE_END')
    self.assertEqual(phoneme_lines[2][0][0], u'SEQUENCE_END')
