import torch
import pickle
from .nlp_utils import sentence_to_tensor
from ..settings import *


def classify_sentence(model, sentence):
    """Classifies a sentence using a defined model.

    Arguments:
        model -- pytorch model object.
        sentence -- sentence string.
        vocab -- python dict that maps words to indices.
        device -- pytorch device.
    """

    model.eval()
    with open(VOCAB_PATH, "rb") as vocabf:
        vocab = pickle.load(vocabf)

    tensor = (sentence_to_tensor(sentence)).to(DEVICE)
    with torch.no_grad():
        prediction = torch.sigmoid(model(tensor))
    return prediction.squeeze(0)
