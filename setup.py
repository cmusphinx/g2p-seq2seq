"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='g2p_seq2seq',
    version='6.2.2a0',
    packages=['g2p_seq2seq'],
    description='Grapheme to phoneme module based on Seq2Seq',
    long_description=long_description,
    url='https://github.com/cmusphinx/g2p-seq2seq',
    author='Nurtas Makhazhanov',
    author_email='makhazhanovn@gmail.com',
    license='Apache Software License',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='g2p seq2seq tensor2tensor rnnlm',

    install_requires=['tensor2tensor>=1.6.6,<=1.7.0'],

    extras_require={
        'tensorflow': ['tensorflow>=1.8.0'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.8.0']
    },

    entry_points={
        'console_scripts': [
            'g2p-seq2seq=g2p_seq2seq.app:main',
        ],
    },
    test_suite = 'tests'
)
