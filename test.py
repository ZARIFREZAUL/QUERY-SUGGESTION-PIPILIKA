from querysuggestion import querysuggestion

textgen = querysuggestion(weights_path='querysuggestion_weights.hdf5',
                       vocab_path='querysuggestion_vocab.json',
                       config_path='querysuggestion_config.json')
textgen.generate(interactive=True, top_n=15)
