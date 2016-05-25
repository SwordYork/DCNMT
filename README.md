Deep Character-Level  Neural Machine Translation
============

We implement a **Deep Character-Level  Neural Machine Translation** based on [Theano](https://github.com/Theano/Theano) and [Blocks](https://github.com/mila-udem/blocks), and implementation is based on the example in [block-examples](https://github.com/mila-udem/blocks-examples).

Training
============

 1. Prepare a parallel linguistics corpus, like corpus in [WMT](http://www.statmt.org/wmt15/translation-task.html).
 2. Tokenize and Shuffle the cropus.
 3. Create the character list for both language using `create_vocab.py`. Don't forget to modify the language setting, vocabulary size and file name in both file.
 4. Create a `data` folder, and put the `vocab.*.*.pkl` and `*.shuf` in the `data` folder.
 5. Edit the `configurations.py`, and run `python run.py`
