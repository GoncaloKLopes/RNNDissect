import spacy
import torch

def sentence_to_tensor(sentence, vocab):
	"""
	Transforms a sentence into an array of indices according to vocabulary vocab.
	Args:
		sentence (string) -> sentence to transform.
		vocab (dict) -> dictionary that maps words to indices.
	"""
	nlp = spacy.load("en")
	tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
	indexed = [vocab[t] for t in tokenized]
	
	tensor = torch.LongTensor(indexed)
	tensor = tensor.unsqueeze(1)
	return tensor

def classify(sentence, model, vocab):
		"""
		Performs the forward pass on a sentence, resulting in a score.
		Args:
			sentence (string) -> the sentence to classify.
		"""
		
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model.eval()
		tensor = (sentence_to_tensor(sentence, vocab)).to(device)
		prediction = torch.sigmoid(model(tensor))
		return prediction.item()
