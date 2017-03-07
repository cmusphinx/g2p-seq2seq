[![Build Status](https://travis-ci.org/cmusphinx/g2p-seq2seq.svg?branch=master)](https://travis-ci.org/cmusphinx/g2p-seq2seq)

# Sequence-to-Sequence G2P toolkit

The tool does Grapheme-to-Phoneme (G2P) conversion using recurrent
neural network (RNN) with long short-term memory units (LSTM). LSTM
sequence-to-sequence models were successfully applied in various tasks,
including machine translation [1] and grapheme-to-phoneme [2].

This implementation is based on python
[TensorFlow](https://www.tensorflow.org/tutorials/seq2seq/),
which allows an efficient training on both CPU and GPU.

## Installation

The tool requires TensorFlow at least version 1.0.0. Please see the installation
[guide](https://www.tensorflow.org/install/)
for details

You can install tensorflow with the following command:

```
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0-cp27-none-linux_x86_64.whl
```

The package itself uses setuptools, so you can follow standard installation process:

```
sudo python setup.py install
```

You can also run the tests

```
python setup.py test
```

The runnable script `g2p-seq2seq` is installed in  `/usr/local/bin` folder by default (you can adjust it with `setup.py` options if needed) . You need to make sure you have this folder included in your `PATH` so you can run this script from command line.

## Running G2P

A pretrained model 2-layer LSTM with 512 hidden units is [available for download on cmusphinx website](https://sourceforge.net/projects/cmusphinx/files/G2P%20Models/g2p-seq2seq-cmudict.tar.gz/download).
Unpack the model after download. The model is trained on [CMU English dictionary](http://github.com/cmusphinx/cmudict)

```
wget -O g2p-seq2seq-cmudict.tar.gz https://sourceforge.net/projects/cmusphinx/files/G2P%20Models/g2p-seq2seq-cmudict.tar.gz/download 
tar xf g2p-seq2seq-cmudict.tar.gz
```

The easiest way to check how the tool works is to run it the interactive mode and type the words

```
$ g2p-seq2seq --interactive --model g2p-seq2seq-cmudict
Creating 2 layers of 512 units.
Reading model parameters from g2p-seq2seq-cmudict
> hello
HH EH L OW
>
```

To generate pronunciations for an English word list with a trained model, run

```
  g2p-seq2seq --decode your_wordlist --model model_folder_path

```
The wordlist is a text file with one word per line


To evaluate Word Error Rate of the trained model, run

```
  g2p-seq2seq --evaluate your_test_dictionary --model model_folder_path

```
The test dictionary should be a dictionary in standard format.


## Training G2P system

To train G2P you need a dictionary (word and phone sequence per line).
See an [example dictionary](http://github.com/cmusphinx/cmudict)

```
  g2p-seq2seq --train train_dictionary.dic --model model_folder_path
```

You can set up maximum training steps:
```
  "--max_steps" - Maximum number of training steps (Default: 0).
     If 0 train until no improvement is observed
```

It is a good idea to play with the following parameters:
```
  "--size" - Size of each model layer (Default: 64).
     We observed much better results with 512 units, but the training becomes slow

  "--num_layers" - Number of layers in the model (Default: 2). 
     For example, you can try 1 if the train set is not large enough, 
     or 3 to hopefully get better results

  "--learning_rate" - Initial Learning rate (Default: 0.5) 

  "--learning_rate_decay_factor" - Learning rate decays by this much (Default: 0.8)
```

You can manually point out Development and Test datasets:
```
  "--valid" - Development dictionary (Default: created from train_dictionary.dic)
  "--test" - Test dictionary (Default: created from train_dictionary.dic)
```

If you need to continue train saved model just launch the following code:
```
  g2p-seq2seq --train train_dictionary.dic --model model_folder_path
```

And, if you want to start training from scratch:
```
  "--reinit" - Rewrite model in model_folder_path
```

#### Word error rate on CMU dictionary data sets

System | WER ([CMUdict PRONALSYL 2007](https://sourceforge.net/projects/cmusphinx/files/G2P%20Models/phonetisaurus-cmudict-split.tar.gz)), % | WER ([CMUdict latest\*](https://github.com/cmusphinx/cmudict)), %
--- | --- | ---
Baseline WFST (Phonetisaurus) | 24.4 | 33.89
LSTM num_layers=2, size=64    | 31.3 | ~39
LSTM num_layers=2, size=512   | 23.3 | ~31
\* These results pointed out for dictionary without stress.

## References
---------------------------------------

[1] Ilya Sutskever, Vinyals Oriol and V. Le Quoc. "Sequence to sequence
learning with neural networks." In Advances in neural information
processing systems, pp. 3104-3112. 2014.

[2] Yao, Kaisheng, and Geoffrey Zweig. "Sequence-to-sequence neural net
models for grapheme-to-phoneme conversion." arXiv preprint
arXiv:1506.00196, 2015.

