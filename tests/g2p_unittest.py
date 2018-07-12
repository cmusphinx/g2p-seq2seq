
import unittest
import shutil
import sys
import os
sys.path.insert(0, '/Users/zhanwenchen/projects/g2p-seq2seq')
from g2p_seq2seq import g2p
import g2p_seq2seq.g2p_trainer_utils as g2p_trainer_utils
from g2p_seq2seq.g2p import G2PModel
from g2p_seq2seq.params import Params

class TestG2P(unittest.TestCase):

  def test_train(self):
    model_dir = os.path.abspath("tests/models/train")
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    train_path = os.path.abspath("tests/data/toydict.train")
    dev_path = os.path.abspath("tests/data/toydict.test")
    params = Params(model_dir, train_path)
    g2p_trainer_utils.save_params(model_dir, params.hparams)
    g2p_model = G2PModel(params, train_path=train_path, dev_path=dev_path,
                         test_path=dev_path)
    g2p_model.train()

    shutil.rmtree(model_dir, ignore_errors=True)

  def test_decode(self):
    model_dir = os.path.abspath("tests/models/decode")
    decode_file_path = os.path.abspath("tests/data/toydict.graphemes")
    output_file_path = os.path.abspath("tests/models/decode/decode_output.txt")
    params = Params(model_dir, decode_file_path)
    params.hparams = g2p_trainer_utils.load_params(model_dir)
    g2p_model = G2PModel(params, test_path=decode_file_path)
    g2p_model.decode(output_file_path=output_file_path)
    out_lines = open(output_file_path).readlines()
    self.assertEqual(out_lines[0].strip(), "cb")
    self.assertEqual(out_lines[1].strip(), "abcabac")
    self.assertEqual(out_lines[2].strip(), "a")
    os.remove(output_file_path)

  def test_evaluate(self):
    model_dir = os.path.abspath("tests/models/decode")
    gt_path = os.path.abspath("tests/data/toydict.test")
    params = Params(model_dir, gt_path)
    params.hparams = g2p_trainer_utils.load_params(model_dir)
    g2p_model = G2PModel(params, test_path=gt_path)
    g2p_model.evaluate()
    correct, errors = g2p_model.calc_errors(gt_path)
    self.assertAlmostEqual(float(errors)/(correct+errors), 1.000, places=3)
