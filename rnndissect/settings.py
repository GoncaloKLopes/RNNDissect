import os
import torch

MODEL_DIR = "/home/goncalo/Documents/rnndissect/model"
ASSETS_DIR = "/home/goncalo/Documents/rnndissect/assets"

VOCAB_PATH = os.path.join(ASSETS_DIR, "imdb_vocab.pickle")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

