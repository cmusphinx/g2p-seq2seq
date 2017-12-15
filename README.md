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

The tool requires TensorFlow at least version 1.3.0 and Tensor2Tensor with version 1.3.0 or higher. Please see the installation
[guide](https://www.tensorflow.org/install/)
for TensorFlow installation details.

You can install tensorflow with the following command:

```
sudo pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp27-none-linux_x86_64.whl
```

And for installing Tensor2Tensor pass the steps from it's installation [guide](https://github.com/tensorflow/tensor2tensor#installation)

The g2p_seq2seq package itself uses setuptools, so you can follow standard installation process:

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
$ g2p-seq2seq --interactive --model_dir model_folder_path
...
> hello
...
INFO:tensorflow:HH EH L OW
...
>
```

To generate pronunciations for an English word list with a trained model, run

```
  g2p-seq2seq --decode your_wordlist --model_dir model_folder_path [--output decode_output_file_path]
```

The wordlist is a text file with one word per line

If you wish to list top N variants of decoding, set return_beams flag and specify beam_size:

```
$ g2p-seq2seq --decode your_wordlist --model_dir model_folder_path --return_beams --beam_size number_returned_beams [--output decode_output_file_path]
```

To evaluate Word Error Rate of the trained model, run

```
  g2p-seq2seq --evaluate your_test_dictionary --model_dir model_folder_path

```
The test dictionary should be a dictionary in standard format:
HELLO\tHH EH L OW
BYE\tB AY

You may also calculate Word Error Rate considering all top N beams.
In this case we consider word decoding as error only if none of the decoded beams will match with the ground true pronunciation of the word.

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
     We observed much better results with 256 units, but the training becomes slower

  "--num_layers" - Number of layers in the model (Default: 2). 
     For example, you can try 1 if the train set is not large enough, 
     or 3 to hopefully get better results

  "--filter_size" - The size of the filter layer in a convolutional layer (Default: 256)

  "--dropout" - The proportion of dropping out units in hidden layers (Default: 0.5)

  "--attention_dropout" - The proportion of dropping out units in an attention layer (Default: 0.5)

  "--num_heads" - Number of applied heads in Multi-attention mechanism (Default: 2)
```

You can manually point out Development and Test datasets:
```
  "--valid" - Development dictionary (Default: created from train_dictionary.dic)
  "--test" - Test dictionary (Default: created from train_dictionary.dic)
```

If you need to continue train a saved model just point out the directory with the existing model:
```
  g2p-seq2seq --train train_dictionary.dic --model model_folder_path
```

And, if you want to start training from scratch:
```
  "--reinit" - Rewrite model in model_folder_path
```

The differences in pronunciations between short and long words can be significant. So, seq2seq models applies bucketing technique to take account of such problems. On the other hand, splitting initial data into too many buckets can worse the final results. Because in this case there will be not enough amount of examples in each particular bucket. To get a better results, you may tune following three parameters that change number and size of the buckets:
```
  "--min_length_bucket" - the size of the minimal bucket (Default: 5)
  "--max_length" - maximal possible length of words or maximal number of phonemes in pronunciations (Default: 40)
  "--length_bucket_step" - multiplier that controls the number of length buckets in the data. The buckets have maximum lengths from min_bucket_length to max_length, increasing by factors of length_bucket_step (Default: 2.0)
```

To reproduce the following results, train the model on CMUdict dictionaries during 50 epochs:
```
--max_epochs 50
```

#### Word error rate on CMU dictionary data sets

System | WER ([CMUdict PRONALSYL 2007](https://sourceforge.net/projects/cmusphinx/files/G2P%20Models/phonetisaurus-cmudict-split.tar.gz)), % | WER ([CMUdict latest\*](https://github.com/cmusphinx/cmudict)), %
--- | --- | ---
Baseline WFST (Phonetisaurus) | 24.4 | 33.89
LSTM num_layers=2, size=256   | 22.2 | ~31
\* These results pointed out for dictionary without stress.

## References
---------------------------------------

[1] Ilya Sutskever, Vinyals Oriol and V. Le Quoc. "Sequence to sequence
learning with neural networks." In Advances in neural information
processing systems, pp. 3104-3112. 2014.

[2] Yao, Kaisheng, and Geoffrey Zweig. "Sequence-to-sequence neural net
models for grapheme-to-phoneme conversion." arXiv preprint
arXiv:1506.00196, 2015.

