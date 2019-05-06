from extractor import Extractor
import torch
import pickle
import sys
sys.path.append("/home/goncalo/Documents/RNNDissect/models")
from models import BidirectionalLSTM
sys.path.append("..")
from utils.nlp_utils import sentence_to_tensor, classify

if __name__ == "__main__":
    input = sys.argv[1]

    model_path = "/home/goncalo/Documents/RNNDissect/models/Bidirectional.pth"
    vocab_path = "/home/goncalo/Documents/RNNDissect/assets/vocab.pickle"
    with open(vocab_path, "rb") as vocabf:
        vocab = pickle.load(vocabf)
    params = {"input_dim":len(vocab), "embedding_dim":100, "hidden_dim":256,
                "output_dim":1, "n_layers":2, "bidirectional":True, 
                "dropout":0.5, "model":BidirectionalLSTM}
    ex = Extractor(model_path, params)

    ex.activations_to_json(input, vocab)

    print(f"{input} -> {classify(input, ex.model, vocab)}")