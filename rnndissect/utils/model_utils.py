import torch
from .nlp_utils import sentence_to_tensor


def classify_sentence(model, sentence, vocab, device):
    """Classifies a sentence using a defined model.

    Arguments:
        model -- pytorch model object.
        sentence -- sentence string.
        vocab -- python dict that maps words to indices.
        device -- pytorch device.
    """

    model.eval()
    tensor = (sentence_to_tensor(sentence, vocab)).to(device)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()
