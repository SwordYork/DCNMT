"""Encoder-Decoder with search for machine translation.

In this demo, encoder-decoder architecture with attention mechanism is used for
machine translation. The attention mechanism is implemented according to
[BCB]_. The training data used is WMT15 Czech to English corpus, which you have
to download, preprocess and put to your 'datadir' in the config file. Note
that, you can use `prepare_data.py` script to download and apply all the
preprocessing steps needed automatically.  Please see `prepare_data.py` for
further options of preprocessing.

.. [BCB] Dzmitry Bahdanau, Kyunghyun Cho and Yoshua Bengio. Neural
   Machine Translation by Jointly Learning to Align and Translate.
"""

import argparse
import logging
import pprint

import configurations
from stream import get_tr_stream, get_dev_stream
from training import main

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--proto", default="get_config_en2fr",
                    help="Prototype config to use for config")
args = parser.parse_args()

if __name__ == "__main__":
    # Get configurations for model
    configuration = getattr(configurations, args.proto)()
    logger.info("Model options:\n{}".format(pprint.pformat(configuration)))
    # Get data streams and call main
    main(configuration, get_tr_stream(**configuration),
         get_dev_stream(**configuration))
