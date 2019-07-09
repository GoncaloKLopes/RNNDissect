import os
import pickle
import sys
import torch

from globals import MODEL_DIR, VOCAB_PATH, DEVICE
from model.bisarnn import BinarySARNN
from model.configs import *
from rnndissect.utils.model_utils import classify_sentence


if __name__ == "__main__":

    if len(sys.argv) < 1:
        raise Exception("Invalid number of arguments.\
                        Please provide a sentence to classify.")

    sentence = sys.argv[1]
    state_dict = torch.load(os.path.join(MODEL_DIR, "lstm_2layers_bidir_adam.pt"))
    config = LSTM_CONFIG3

    model = BinarySARNN(config)
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    result = classify_sentence(model, sentence)

    print("Score ->", result)

else:
    raise Exception("This is a demo file, you should not be importing me!")
