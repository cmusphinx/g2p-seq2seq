
import unittest
import shutil
import g2p_seq2seq.g2p as g2p
from g2p_seq2seq.g2p import G2PModel
from g2p_seq2seq.params import Params
import inspect, os


class TestG2P(unittest.TestCase):

  def test_train(self):
    model_dir = os.path.abspath("tests/models/train")
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    train_path = os.path.abspath("tests/data/toydict.train")
    dev_path = os.path.abspath("tests/data/toydict.test")
    params = Params(model_dir, train_path)
    g2p_model = G2PModel(params, file_path=train_path, is_training=True)
    g2p_model.prepare_data(train_path=train_path, dev_path=dev_path)
    g2p_model.train()
    shutil.rmtree(model_dir)

  def test_decode(self):
    model_dir = os.path.abspath("tests/models/decode")
    decode_file_path = os.path.abspath("tests/data/toydict.graphemes")
    output_file_path = os.path.abspath("tests/models/decode/decode_output.txt")
    params = Params(model_dir, decode_file_path)
    g2p_model = G2PModel(params, file_path=decode_file_path, is_training=False)
    g2p_model.decode(output_file_path=output_file_path)
    out_lines = open(output_file_path).readlines()
    self.assertEqual(out_lines[0].strip(), u"C B C A C B")
    self.assertEqual(out_lines[1].strip(), u"C B C A C B")
    self.assertEqual(out_lines[2].strip(), u"A")

  def test_evaluate(self):
    model_dir = os.path.abspath("tests/models/decode")
    gt_path = os.path.abspath("tests/data/toydict.test")
    params = Params(model_dir, gt_path)
    g2p_model = G2PModel(params, file_path=gt_path, is_training=False)
    estimator, decode_hp, g2p_gt_map = g2p_model.evaluate()
    correct, errors = g2p.calc_errors(g2p_gt_map, estimator, gt_path, decode_hp)
    self.assertAlmostEqual(float(errors)/(correct+errors), 1.000, places=3)
