import json
import os
import pickle
import sys
import torch

from globals import MODEL_DIR, VOCAB_PATH, DEVICE
from model.lstm import LSTM
from rnndissect.utils.model_utils import classify_sentence


if __name__ == "__main__":

    if len(sys.argv) < 1:
        raise Exception("Invalid number of arguments.\
                        Please provide a sentence to classify.")

    sentence = sys.argv[1]
    model_path = os.path.join(MODEL_DIR, "bidirlstm.pth")
    params_file = os.path.join(MODEL_DIR, "bidirlstm.json")

    with open(VOCAB_PATH, "rb") as vocabf:
        vocab = pickle.load(vocabf)

    with open(params_file) as jsonf:
        params = json.loads(jsonf.read())

    model = LSTM(len(vocab),
                 params["embedding_dim"],
                 params["hidden_dim"],
                 params["output_dim"],
                 params["n_layers"],
                 params["bidirectional"],
                 params["dropout"])

    trained = torch.load(model_path)
    model.load_state_dict(trained)
    model.to(DEVICE)

    result = classify_sentence(model, sentence, vocab, DEVICE)

    print("Score ->", round(result, 2))

else:
    raise Exception("This is a demo file, you should not be importing me!")
