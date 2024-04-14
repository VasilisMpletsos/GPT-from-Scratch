# Pathlib
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


# Iterating through dataset to extract the original sentence and its translation
def get_all_sentences(ds, lang):
    for pair in ds:
        yield pair["translation"][lang]


# Defining Tokenizer
def build_tokenizer(config, ds, lang):
    # Crating a file path for the tokenizer
    tokenizer_path = Path(config["tokenizer_file"].format(lang))

    # Checking if Tokenizer already exists
    if not Path.exists(tokenizer_path):
        # If it doesn't exist, we create a new one
        tokenizer = Tokenizer(
            WordLevel(unk_token="[UNK]")
        )  # Initializing a new world-level tokenizer
        tokenizer.pre_tokenizer = (
            Whitespace()
        )  # We will split the text into tokens based on whitespace

        # Creating a trainer for the new tokenizer
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )  # Defining Word Level strategy and special tokens

        all_sentences_iterator = get_all_sentences(ds, lang)

        # Training new tokenizer on sentences from the dataset and language specified
        tokenizer.train_from_iterator(all_sentences_iterator, trainer=trainer)
        tokenizer.save(
            str(tokenizer_path)
        )  # Saving trained tokenizer to the file path specified at the beginning of the function
    else:
        tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )  # If the tokenizer already exist, we load it
    return tokenizer  # Returns the loaded tokenizer or the trained tokenizer
