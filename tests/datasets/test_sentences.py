from datasets import load_dataset, Dataset
from transformer_ranker.utils import DatasetCleaner
from tokenizers.pre_tokenizers import Whitespace


def test_sentence_preprocessing():
    sentence_dataset = "trec"
    dataset_size = 5952
    first_sentence = 'How did serfdom develop in and then leave Russia ?'
    first_sentence_tokenized = ['How', 'did', 'serfdom', 'develop', 'in', 'and', 'then', 'leave', 'Russia', '?']

    dataset = load_dataset(sentence_dataset)
    handler = DatasetCleaner()
    preprocessed_dataset = handler.prepare_dataset(dataset)

    # Check if the dataset has only the relevant columns (text and label)
    assert isinstance(preprocessed_dataset, Dataset)
    assert handler.text_column in preprocessed_dataset.column_names
    assert handler.label_column in preprocessed_dataset.column_names
    assert len(preprocessed_dataset.column_names) == 2

    # Check if the size is same after preprocessing
    assert len(preprocessed_dataset) == dataset_size

    # Check if the first sentence in prepare sentences is still the same as original
    sentences = handler.prepare_sentences(preprocessed_dataset)
    assert sentences[0] == first_sentence

    handler = DatasetCleaner(pre_tokenizer=Whitespace())
    tokenized_dataset = handler.prepare_dataset(dataset)

    # Check if the size is same after preprocessing and tokenizing
    assert len(tokenized_dataset) == dataset_size

    # Tokenization should affect sentence tasks
    tokenized_sentences = handler.prepare_sentences(tokenized_dataset)
    assert tokenized_sentences[0] == first_sentence_tokenized


def test_word_datasets_datahandler():
    word_dataset = "conll2003"
    dataset_size = 20744
    first_sentence = ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']

    dataset = load_dataset(word_dataset)
    handler = DatasetCleaner()
    preprocessed_dataset = handler.prepare_dataset(dataset)

    # Check if the dataset has only the relevant columns (text and label)
    assert isinstance(preprocessed_dataset, Dataset)
    assert handler.text_column in preprocessed_dataset.column_names
    assert handler.label_column in preprocessed_dataset.column_names
    assert len(preprocessed_dataset.column_names) == 2

    # Check if the size is same after preprocessing
    assert len(preprocessed_dataset) == dataset_size

    # Check if the first sentence in prepare sentences is still the same as original
    sentences = handler.prepare_sentences(preprocessed_dataset)
    assert sentences[0] == first_sentence

    handler = DatasetCleaner(pre_tokenizer=Whitespace())
    tokenized_dataset = handler.prepare_dataset(dataset)

    # Check if the size is same after preprocessing and tokenizing
    assert len(tokenized_dataset) == dataset_size

    # Tokenization should not affect word classification tasks
    tokenized_sentences = handler.prepare_sentences(tokenized_dataset)
    assert tokenized_sentences[0] == first_sentence

# pytest test_sentences.py -v
