
import unittest
import shutil
import g2p_seq2seq.g2p as g2p
from g2p_seq2seq.g2p import G2PModel
from g2p_seq2seq.params import Params
import inspect, os

from IPython.core.debugger import Tracer

class TestG2P(unittest.TestCase):

  def test_train(self):
    model_dir = os.path.abspath("tests/models/train")
    train_path = os.path.abspath("tests/data/toydict.train")
    dev_path = os.path.abspath("tests/data/toydict.test")
    params = Params(model_dir, train_path)
    g2p_model = G2PModel(params)
    g2p_model.prepare_data(train_path=train_path, dev_path=dev_path)
    self.assertIsNone(g2p_model.train())
    shutil.rmtree(model_dir)

  def test_decode(self):
    model_dir = os.path.abspath("tests/models/decode")
    decode_file_path = os.path.abspath("tests/data/toydict.graphemes")
    output_file_path = os.path.abspath("tests/models/decode/decode_output.txt")
    params = Params(model_dir, decode_file_path)
    g2p_model = G2PModel(params)
    g2p_model.prepare_data(test_path=decode_file_path)
    g2p_model.decode(decode_file_path=decode_file_path,
      output_file_path=output_file_path)
    out_lines = open(output_file_path).readlines()
    self.assertEqual(out_lines[0].strip(), u"")
    self.assertEqual(out_lines[1].strip(), u"")
    self.assertEqual(out_lines[2].strip(), u"")

  def test_evaluate(self):
    model_dir = os.path.abspath("tests/models/decode")
    test_path = os.path.abspath("tests/data/toydict.graphemes2")
    gt_path = os.path.abspath("tests/data/toydict.test")
    params = Params(model_dir, test_path)
    g2p_model = G2PModel(params)
    g2p_model.prepare_data(test_path=test_path)
    gt_lines = open(gt_path).readlines()
    g2p_gt_map = g2p.create_g2p_gt_map(gt_lines)
    estimator, decode_hp = g2p_model.evaluate(gt_path, test_path)
    errors = g2p.calc_errors(g2p_gt_map, test_path, estimator, decode_hp)
    self.assertAlmostEqual(float(errors)/len(g2p_gt_map), 1.000, places=3)
