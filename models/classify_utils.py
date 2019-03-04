import torch
from models import BidirectionalLSTM

import pickle
import spacy
import sys


def classify(sentence, model, device):
	tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
	indexed = [vocab[t] for t in tokenized]

	tensor = torch.LongTensor(indexed).to(device)
	tensor = tensor.unsqueeze(1)

	prediction = torch.sigmoid(model(tensor))
	return prediction.item()


if __name__ == "__main__":
	input = sys.argv[1]

	nlp = spacy.load("en")
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(device)
	with open("vocab.pickle", "rb") as vocabf:
		vocab = pickle.load(vocabf)

	INPUT_DIM = len(vocab)
	EMBEDDING_DIM = 100
	HIDDEN_DIM = 256
	OUTPUT_DIM = 1
	N_LAYERS = 2
	BIDIRECTIONAL = True
	DROPOUT = 0.5
	model = BidirectionalLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 
		  				      N_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)

	model.load_state_dict(torch.load("Bidirectional.pth"))

	#print(model.state_dict())
	print(f"{input} -> {classify(input, model, device)}")