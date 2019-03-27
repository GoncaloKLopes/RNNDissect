import torch
from models import BidirectionalLSTM
import pickle
import spacy
import sys
import numpy as np
import math
import random
import json

class RNNToolbox:
	"""
	Class that encapsulates rnn utilities.
	"""

	def __init__(self, model, n_layers,  vocab_path):
		"""
		Args:
			model (nn.Module) -> an instance of one of the model classes 
								 defined in models.py
			vocab_path (string) -> path to the serialized vocabulary file.
		"""
		self.model = model
		self.n_layers = n_layers
		with open(vocab.path, "rb") as vocabf:
			self.vocab = pickle.load(vocabf)	
				
	def classify(self, sentence):
		"""
		Performs the forward pass on a sentence, resulting in a score.
		Args:
			sentence (string) -> the sentence to classify.
		"""
		tensor = sentence_to_tensor(sentence, self.vocab)	
	    prediction = torch.sigmoid(model(tensor))
	    return prediction.item()

	def extract_params(n_layers, pytorch_dict):
		"""
		Given the model.state_dict() from an lstm from pytorch,
		unfold the pytorch dict's matrices into a new dict that facilitates
		accesses.

		Args:
			n_layers (int) -> denotes the number of layers of the model.
			pytorch_dict (dict) -> obtained from model.state_dict.
		"""
		result = {}
		gate_ids = set(zip(range(4), ["i", "f", "g", "o"]))
		tensor_ids = ["i", "h"]
		params = ["weight", "bias"]
		
		for rev in range(2):
			revstr = ("_reverse" * rev)
			for param in params:
				for l in range(n_layers):
					for tensor_id in tensor_ids:
						tensor = f"rnn.{param}_{tensor_id}h_l{l}" + revstr
						for idx, id in gate_ids:
							result[f"{param}_{tensor_id}{id}{l}"+revstr] = \
								pytorch_dict[tensor][(idx*HIDDEN_DIM):((idx+1)*HIDDEN_DIM)]
		return result

	def infer_gates(self, xt, ht, ct, params, l, t, rev=False):
		"""
		Assume the following dimension:
		- xt.shape = [EMBEDDING_DIM x 1]
		- ht.shape = [HIDDEN_DIM x 1]
		- ct.shape = [HIDDEN_DIM x 1]
		- weights[Wi*] = [HIDDEN_DIM x EMBEDDING_DIM]
		- weights[Wh*] = [HIDDEN_DIM x HIDDEN_DIM]
		- biases[*] = [HIDDEN_DIM]
		"""
		res = {}
		tanh = torch.nn.Tanh()
		
		wstr = "weight_"
		bstr = "bias_"
		revstr = "_reverse"
		
		#how do I loop this
		Wiil = wstr + f"ii{l}" + (revstr * rev)
		biil = bstr + f"ii{l}" + (revstr * rev)
		Whil = wstr + f"hi{l}" + (revstr * rev)
		bhil = bstr + f"hi{l}" + (revstr * rev)
		Wifl = wstr + f"if{l}" + (revstr * rev)
		bifl = bstr + f"if{l}" + (revstr * rev)
		Whfl = wstr + f"hf{l}" + (revstr * rev)
		bhfl = bstr + f"hf{l}" + (revstr * rev)
		Wigl = wstr + f"ig{l}" + (revstr * rev)
		bigl = bstr + f"ig{l}" + (revstr * rev)
		Whgl = wstr + f"hg{l}" + (revstr * rev)
		bhgl = bstr + f"hg{l}" + (revstr * rev)
		Wiol = wstr + f"io{l}" + (revstr * rev)
		biol = bstr + f"io{l}" + (revstr * rev)
		Whol = wstr + f"ho{l}" + (revstr * rev)
		bhol = bstr + f"ho{l}" + (revstr * rev)
		itl = f"i{t}{l}" + (revstr * rev)
		ftl = f"f{t}{l}" + (revstr * rev)
		gtl = f"g{t}{l}" + (revstr * rev)
		otl = f"o{t}{l}" + (revstr * rev)
		#tt is t+1 in variablenameland
		cttl = f"c{t+1}{l}" + (revstr * rev)
		httl = f"h{t+1}{l}" + (revstr * rev)
		
		res[itl] = infer_gate(params[Wiil], xt, params[biil],
						params[Whil], ht, params[bhil], torch.sigmoid)

		res[ftl] = infer_gate(params[Wifl], xt, params[bifl],
						params[Whfl], ht, params[bhfl], torch.sigmoid)

		res[gtl] = infer_gate(params[Wigl], xt, params[bigl],
						params[Whgl], ht, params[bhgl], tanh)       

		res[otl] = infer_gate(params[Wiol], xt, params[biol],
						params[Whol], ht, params[bhol], torch.sigmoid) 

		res[cttl] = torch.add(torch.mul(res[ftl], ct), torch.mul(res[itl], res[gtl]))
		res[httl] = torch.mul(res[otl], tanh(res[cttl]))
			
		return res

	def infer_gate(i_w, x, b_i, h_w, h, b_h, fun):
		input = torch.addmm(torch.unsqueeze(b_i, -1), i_w, x)
		hidden = torch.addmm(torch.unsqueeze(b_h, -1), h_w, h)
		
		return fun(torch.add(input, hidden))

	def forward_pass(self, embeddings, n_layers, params, hidden_dim):
		result = {} #saves values of states and gates
		revstr = "_reverse"
			
		for l in range(n_layers):
			#random initialization of both hidden and cell states
			result[f"h0{l}"] = torch.zeros(hidden_dim, 1).to(device)
			result[f"c0{l}"] = torch.zeros(hidden_dim, 1).to(device)
			#reverse aswell
			result[f"h0{l}{revstr}"] = torch.zeros(hidden_dim, 1).to(device)
			result[f"c0{l}{revstr}"] = torch.zeros(hidden_dim, 1).to(device)
			
			for t in range(len(embeddings)):
				#if layer > 1, then the input isn't the model input, but the hidden state from the previous
			#layer
				reverse_t = len(embeddings)-(t+1)
				if l == 0:
					input = torch.transpose(embeddings[t],0 , 1)
					input_rev = torch.transpose(embeddings[reverse_t], 0, 1)
				else:
					input = torch.cat((result[f"h{t+1}{l-1}"], result[f"h{reverse_t+1}{l-1}{revstr}"]))
					input_rev = torch.cat((result[f"h{reverse_t+1}{l-1}"],
		  								  result[f"h{t+1}{l-1}{revstr}"]))
						
				result.update(infer_gates(input, result[f"h{t}{l}"], 
											  result[f"c{t}{l}"], params, l, t))
				result.update(infer_gates(input_rev, result[f"h{t}{l}{revstr}"], result[f"c{t}{l}{revstr}"], 
											  params, l, t, rev=True))
			return result

	def activations_to_json(self, json_path, sentence):
		"""
		Writes the activation values of the model corrresponding
		to a specific input sentence into a .json file.
		Args: 
			json_path (string) -> path of the json file.	
			sentence (string) -> input sentence
		"""
		embeddings = model.dropout(model.embedding(input))
		acts = self.forward_pass(embeddings, self.n_layers, extract_params(
									self.n_layers, model.state_dict()), 256)
		return acts 
						
if __name__ == "__main__":
	input = sys.argv[1]


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
