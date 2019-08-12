import spacy
import torch
import pickle

from ..settings import *


def tokenize_sentence(sentence):
    nlp = spacy.load("en")
    return nlp.tokenizer(sentence)


def sentence_to_tensor(sentence):
    """
    Transforms a sentence into an array of indices according
    to vocabulary vocab.

    Args:
        sentence (string) -> sentence to transform.
        vocab (dict) -> dictionary that maps words to indices.
    """

    with open(VOCAB_PATH, "rb") as vocabf:
        vocab = pickle.load(vocabf)
    tokenized = tokenize_sentence(sentence.lower())

    specials = {"'s": "is"}
    for i in range(len(tokenized)):
        if tokenized[i] in specials.keys():
            tokenized[i] = specials[tokenized[i]]

    indexed = []
    for t in tokenized:
        if str(t) in specials.keys():
            indexed.append(vocab[specials[str(t)]].index)
        elif str(t) == "\n":
            continue
        else:
            indexed.append(vocab[str(t)].index)

    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1)
    return tensor
