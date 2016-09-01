def get_config():
    config = {}

    # prepare data
    config['source_language'] = '--src_lang--'
    config['target_language'] = '--trg_lang--'


    # Where to save model, this corresponds to 'prefix' in groundhog
    config['saveto'] = 'dcnmt_{}2{}'.format(config['source_language'], config['target_language'])

    # Model related -----------------------------------------------------------

    # Sequences longer than this will be discarded
    config['max_src_seq_char_len'] = 300
    config['max_src_seq_word_len'] = 50
    config['max_trg_seq_char_len'] = 300
    config['max_trg_seq_word_len'] = 50

    # Number of hidden units in encoder/decoder GRU
    config['src_dgru_nhids'] = 512
    config['enc_nhids'] = 1024
    config['dec_nhids'] = 1024
    config['trg_dgru_nhids'] = 512
    config['trg_igru_nhids'] = 1024


    # Dimension of the word embedding matrix in encoder/decoder
    config['enc_embed'] = 64
    config['dec_embed'] = 64
    config['src_dgru_depth'] = 2
    config['bidir_encoder_depth'] = 2
    config['transition_depth'] = 1
    config['trg_dgru_depth'] = 1
    config['trg_igru_depth'] = 1

    # Optimization related ----------------------------------------------------

    # Batch size
    config['batch_size'] = 80

    # This many batches will be read ahead and sorted
    config['sort_k_batches'] = 12

    # Optimization step rule
    config['step_rule'] = 'AdaDelta'

    # Gradient clipping threshold
    config['step_clipping'] = 1.

    # Std of weight initialization
    config['weight_scale'] = 0.01


    # Vocabulary/dataset related ----------------------------------------------

    # Root directory for dataset
    datadir = './data/'

    # Module name of the stream that will be used
    config['stream'] = 'stream'

    # Source and target vocabularies
    config['src_vocab'] = datadir + 'vocab.{}-{}.{}.pkl'.format(config['source_language'], config['target_language'],
                                                                config['source_language'])
    config['trg_vocab'] = datadir + 'vocab.{}-{}.{}.pkl'.format(config['source_language'], config['target_language'],
                                                                config['target_language'])

    # Source and target datasets
    config['src_data'] = datadir + 'all.{}-{}.{}.tok.shuf'.format(config['source_language'], config['target_language'],
                                                                config['source_language'])
    config['trg_data'] = datadir + 'all.{}-{}.{}.tok.shuf'.format(config['source_language'], config['target_language'],
                                                                config['target_language'])

    # Source and target vocabulary sizes, should include bos, eos, unk tokens
    config['src_vocab_size'] = --src_vocab_size--
    config['trg_vocab_size'] = --trg_vocab_size--

    # Special tokens and indexes
    config['unk_id'] = 1
    config['bos_token'] = '<S>'
    config['eos_token'] = '</S>'
    config['unk_token'] = '<UNK>'

    # Early stopping based on val related ------------------------------------

    # Normalize cost according to sequence length after beam-search
    config['normalized_val'] = True

    # Normalize cost according to sequence length after beam-search
    config['normalized_bleu'] = True

    # Bleu script that will be used (moses multi-perl in this case)
    config['bleu_script'] = datadir + 'multi-bleu.perl'

    # Validation set source file
    config['val_set'] = datadir + '--src_val--.tok'

    # Validation set gold file
    config['val_set_grndtruth'] = datadir + '--trg_val--.tok'

    # Test set source file
    config['test_set'] = datadir + '--src_test--.tok'

    # Test set gold file
    config['test_set_grndtruth'] = datadir + '--trg_test--.tok'

    config['validate'] = True

    # Print validation output to file
    config['output_val_set'] = True

    # Validation output file
    config['val_set_out'] = config['saveto'] + '/validation_out.txt'

    # Validation output file
    config['test_set_out'] = config['saveto'] + '/test_out.txt'

    # Beam-size
    config['beam_size'] = 12

    # Timing/monitoring related -----------------------------------------------

    # Maximum number of updates
    config['finish_after'] = 800000

    # Reload model from files if exist
    config['reload'] = True

    # Save model after this many updates
    config['save_freq'] = 500

    # Print training status after this many updates
    config['print_freq'] = 10

    # Show samples from model after this many updates
    config['sampling_freq'] = 30

    # Show this many samples at each sampling
    config['hook_samples'] = 2

    config['bleu_val_freq'] = 18000
    # Start validation after this many updates
    config['val_burn_in'] = 70000

    return config
