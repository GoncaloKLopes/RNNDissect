import spacy
import torch


def tokenize_sentence(sentence):
    nlp = spacy.load("en")
    return nlp.tokenizer(sentence)


def sentence_to_tensor(sentence, vocab):
    """
    Transforms a sentence into an array of indices according
    to vocabulary vocab.

    Args:
        sentence (string) -> sentence to transform.
        vocab (dict) -> dictionary that maps words to indices.
    """
    tokenized = tokenize_sentence(sentence)
    indexed = [vocab[t] for t in tokenized]

    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1)
    return tensor
