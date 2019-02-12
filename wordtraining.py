from querysuggestion import querysuggestion

suggestion = querysuggestion()

file_path = "data.txt"

suggestion.reset()
suggestion.train_from_largetext_file(file_path,
                        new_model=True,
                        max_length=5,
                        num_epochs=20,
                        gen_epochs=5,
                        word_level=True)
