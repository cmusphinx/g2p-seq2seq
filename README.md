# Sequence-to-Sequence G2P toolkit

The tool does Grapheme-to-Phoneme (G2P) conversion using recurrent neural network (RNN) with long short-term memory units (LSTM).
LSTM sequence-to-sequence models were successfully applied in various tasks, including machine translation [1] and grapheme-to-phoneme [2].

This implementation is based on python [TensorFlow](https://www.tensorflow.org/versions/r0.8/tutorials/seq2seq/index.html), which allows an efficient training on both CPU and GPU.

## Requirements

The tool requires TensorFlow. Please see the installation [guide](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md) for details

## Running G2P

A 2-layer LSTM with 64 hidden units is already included in the package.
It is trained on [CMU English dictionary](http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b)

The easiest way to check how the tool works is to run it the interactive mode
```
  python g2p.py --interactive_mode --model_dir PATH_TO_G2P/mdl/cmu_en_2l64

  then type the words
```


To generate pronunciations for an English word list with a trained model, run

```
  python g2p.py --decode_file [your_wordlist] --model_dir [path_to_g2p]/mdl/cmu_en_2l64

```
The wordlist is a text file: one word per line



## Training G2P system

To train G2P you need a dictionary (word and phone sequence per line). See an [example dictionary](http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b)

```
  python g2p.py --train_dic [train_dictionary.dic] --model_dir [output_model_path]
```

It is a good idea to play with the following parameters:
```
  "--size" - Size of each model layer (Default: 64).
     We observed much better results with 512 units, but the training becomes slow

  "--num_layers" - Number of layers in the model (Default: 2). 
     For example, you can try 1 if the train set is not large enough, 
     or 3 to hopefully get better results

  "--learning_rate" - Initial Learning rate (Default: 0.5). 

  "--learning_rate_decay_factor" - Learning rate decays by this much (Default: 0.95)

```


#### Word error rate on 12K test set of CMU dictionary

System | WER,%
--- | --- 
Baseline WFST (Phonetisaurus) | 28.0
LSTM num_layers=2, size=64    | 31.9 
LSTM num_layers=2, size=512   | **24.7**



## References
---------------------------------------
[1] Ilya Sutskever, Vinyals Oriol and V. Le Quoc. "Sequence to sequence learning with neural networks." In Advances in neural information processing systems, pp. 3104-3112. 2014.

[2] Yao, Kaisheng, and Geoffrey Zweig. "Sequence-to-sequence neural net models for grapheme-to-phoneme conversion." arXiv preprint arXiv:1506.00196, 2015.

