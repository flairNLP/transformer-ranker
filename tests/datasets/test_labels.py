from datasets import load_dataset
from transformer_ranker.datacleaner import DatasetCleaner
import torch


def test_sentence_labels():
    sentence_dataset = "trec"
    num_labels = 5952
    first_label = torch.tensor(2)

    dataset = load_dataset(sentence_dataset)

    handler = DatasetCleaner()
    dataset = handler.prepare_dataset(dataset)
    labels = handler.prepare_labels(dataset)

    # Check if labels were converted to a tensor
    assert isinstance(labels, torch.Tensor)

    # Check if number of labels match
    assert torch.Size([num_labels]) == labels.shape

    # Check if the first label is as expected
    print(labels[0])
    assert torch.equal(first_label, labels[0])


def test_word_labels():
    word_dataset = "conll2003"
    num_labels = 301418  # total number of tokens (words) in the dataset
    first_label = torch.Tensor([3, 0, 7, 0, 0, 0, 7, 0, 0])

    dataset = load_dataset(word_dataset )

    handler = DatasetCleaner()
    dataset = handler.prepare_dataset(dataset)
    labels = handler.prepare_labels(dataset)

    # Check if labels were converted to a tensor
    assert isinstance(labels, torch.Tensor)

    # Check if number of labels match
    assert torch.Size([num_labels]) == labels.shape

    # Check if the first subset of the label tensor matches
    assert torch.equal(labels[:len(first_label)], first_label)

# pytest test_labels.py -v
