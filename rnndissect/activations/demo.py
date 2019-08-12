import torch
import pickle
import sys

from .extractor import Extractor
from models import BidirectionalLSTM
from utils.model_utils import classify

if __name__ == "__main__":
    input_file = sys.argv[1]
    fname = input_file + ".json"

    with open(os.path.join("../../reviews", input_file), "r") as f:
        input = f.read()

    with open(vocab_path, "rb") as vocabf:
        vocab = pickle.load(vocabf)
    params = {"input_dim": len(vocab), "embedding_dim": 100, "hidden_dim": 256,
              "output_dim": 1, "n_layers": 2, "bidirectional": True,
              "dropout": 0.5, "model": BidirectionalLSTM}
    ex = Extractor(model_path, params)

    ex.activations_to_json(input, fname)

    print(f"{input_file} -> {classify(input, ex.model, vocab)}")

    """
    #   SANITY CHECK
    encoded = (sentence_to_tensor(input, vocab)).to(ex.device)
    embeddings = ex.model.embedding(encoded)
    acts = ex.forward_pass(embeddings, ex.extract_params())
    out = torch.sigmoid(ex.model(acts["h291"]))
    print(out)
    """
