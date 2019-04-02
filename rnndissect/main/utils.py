import spacy
import torch

def sentence_to_tensor(sentence, vocab):
	"""
	Represents a sentence using the indices in a vocabulary.
	Args:
		sentence (string) -> sentence to transform.
		vocab (dict) -> dictionary that maps words to indices.
	"""
	nlp = spacy.load("en")
	tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
	indexed = [vocab[t] for t in tokenized]
	
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	tensor = torch.LongTensor(indexed).to(device)
	tensor = tensor.unsqueeze(1)
	return tensor
