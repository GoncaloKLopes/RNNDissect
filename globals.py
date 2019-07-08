import os
import torch

MODEL_DIR = os.path.join(str(os.path.dirname(os.path.abspath(__file__))),
                         "model")
ASSETS_DIR = os.path.join(str(os.path.dirname(os.path.abspath(__file__))),
                          "assets")
VOCAB_PATH = os.path.join(ASSETS_DIR, "imdb_vocab.pickle")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
