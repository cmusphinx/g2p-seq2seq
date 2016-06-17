import os, sys, inspect
# realpath() will make your script run, even if you symlink it :)
#cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
#if cmd_folder not in sys.path:
#    sys.path.insert(0, cmd_folder)

sys.path.insert(0, './g2p_seq2seq')

import unittest
import g2p
import data_utils

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_init(self):
        #train_path = "./unittest_model/toydict.train"
        #model_path = "./unittest_model/model"
        #valid_path = "./unittest_model/toydict.test"
        #test_path = "./unittest_model/toydict.test"
        #train_file, valid_file, test_file =\
        #    data_utils.split_dictionary(train_path, valid_path, test_path)
        model_dir = "./unittest_model/model"
        g2p_model = g2p.G2PModel(model_dir)
        train_path = "./unittest_model/toydict.train"
        valid_path = "./unittest_model/toydict.test"
        test_path = "./unittest_model/toydict.test"
        g2p_params = g2p.G2P_Params()
        g2p_params.max_steps = 200
        g2p_model.train(g2p_params, train_path, valid_path, test_path)
 

#if __name__ == '__main__':
#    unittest.main()
suite = unittest.TestLoader().loadTestsFromTestCase(TestStringMethods)
unittest.TextTestRunner(verbosity=2).run(suite)
