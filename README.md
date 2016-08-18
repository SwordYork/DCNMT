Deep Character-Level  Neural Machine Translation
============
We implement a **Deep Character-Level Neural Machine Translation** based on [Theano](https://github.com/Theano/Theano) and [Blocks](https://github.com/mila-udem/blocks). Please intall relative packages according to [Blocks](http://blocks.readthedocs.io/en/latest/setup.html) before testing our program. Note that, please use Python3 instead of Python2. There will be some problem with Python2. 

The architecture of DCNMT is shown in the following figure which is a single, large neural network.
![DCNMT](/dcnmt.png?raw=true "The architecture of DCNMT")




Training
-----------------------
If you want to train you own model, please prepare a parallel linguistics corpus, like corpus in [WMT](http://www.statmt.org/wmt15/translation-task.html). A GPU with 12GB memory will be helpful. You could just run `bash train.sh` or 
follow these steps.
 1. Tokenize and shuffle the cropus.
 2. Create the character list for both language using `create_vocab.py` in `preprocess` folder. Don't forget to pass the language setting, vocabulary size and file name to this script.
 3. Create a `data` folder, and put the `vocab.*.*.pkl` and `*.shuf` in the `data` folder.
 4. Edit the `configurations.py`, and run `python training.py`.


Testing
-----------------------
To perform testing, just run `python testing.py`. Note that, Python3 is required. It takes about an hour to do translation on 3000 sentences if you have a moderate GPU.


Embedding
-----------------------
Please prepare a wordlist to calculate embedding, then just run `python embedding.py` to view the results.


This program have been tested under the latest Theano and Blocks, it may fail to run because of different version. If you failed to run these scripts, please make sure that you can run the examples of [Blocks](https://github.com/mila-udem/blocks-examples).


References:
----------------------
1. [An Efficient Character-Level Neural Machine Translation](https://arxiv.org/abs/1608.04738)
2. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
