import os
import sys
import torch
import pickle

sys.path.append("..")
from utils.model_utils import classify
import extractor as extr


sys.path.append("../../model")
from bisarnn import BinarySARNN
from configs import *

with open("/home/goncalo/Documents/rnndissect/assets/imdb_vocab.pickle", "rb") as vf:
    vocab = pickle.load(vf)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    input_file = sys.argv[1]
    config_str = sys.argv[2]

    if config_str == "rnn1":
        config = RNN_CONFIG1
        acts_file = "../../model/simple_rnn.pt"
    elif config_str == "rnn2":
        config = RNN_CONFIG2
        acts_file = "../../model/simple_rnn_2layers.pt"
    elif config_str == "rnn3":
        config = RNN_CONFIG3
        acts_file = "../../model/simple_rnn_2layers_bidir.pt"
    elif config_str == "lstm1":
        config = LSTM_CONFIG1
        acts_file = "../../model/lstm.pt"
    elif config_str == "lstm2":
        config = LSTM_CONFIG2
        acts_file = "../../model/lstm_2layers.pt"
    elif config_str == "lstm3":
        config = LSTM_CONFIG3
        acts_file = "../../model/lstm_2layers_bidir.pt"
    else:
        raise("NOT A VALID CONFIG.")

    config.output_dim = 2

    model = BinarySARNN(config)
    model.load_state_dict(torch.load(acts_file))
    model.to(DEVICE)

    with open(os.path.join("../../reviews", input_file)) as f:
        inpt = f.read()

    json_fname = config_str + "_" + input_file[:-4] + ".json"
    output = rnndissect.utils.model_utils.classify(model, inpt)

    ex = extr.Extractor(config, model)
    ex.activations_to_json(inpt, json_fname)
