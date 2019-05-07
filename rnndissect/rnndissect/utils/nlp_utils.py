import spacy
import torch

def tokenize_sentence(sentence, vocab):
	nlp = spacy.load("en")
	return nlp.tokenizer(sentence)

def sentence_to_tensor(sentence, vocab):
	"""
	Transforms a sentence into an array of indices according to vocabulary vocab.
	Args:
		sentence (string) -> sentence to transform.
		vocab (dict) -> dictionary that maps words to indices.
	"""
	tokenized = tokenize_sentence(sentence, vocab)
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
