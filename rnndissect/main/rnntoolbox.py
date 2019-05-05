import torch
from models import BidirectionalLSTM
from utils import sentence_to_tensor
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

	def __init__(self, model_path, vocab_path):
		"""
		Args:
			model (nn.Module) -> an instance of one of the model classes
								 defined in models.py
			vocab_path (string) -> path to the serialized vocabulary file.
		"""
		self.device = torch.device('cuda' if torch.cuda.is_available()
										  else 'cpu')

		#load vocab first to get the number of inputs
		with open(vocab_path, "rb") as vocabf:
			self.vocab = pickle.load(vocabf)

		#TODO READ PARAMETERS FROM A FILE
		INPUT_DIM = len(self.vocab)
		EMBEDDING_DIM = 100
		self.hidden_dim = 256
		self.output_dim = 1
		self.n_layers = 2
		BIDIRECTIONAL = True
		DROPOUT = 0.5

		self.model = BidirectionalLSTM(INPUT_DIM, EMBEDDING_DIM, self.hidden_dim,
								  self.output_dim, self.n_layers, BIDIRECTIONAL,
								  DROPOUT).to(self.device)

		self.model.load_state_dict(torch.load(model_path))



	def classify(self, sentence):
		"""
		Performs the forward pass on a sentence, resulting in a score.
		Args:
			sentence (string) -> the sentence to classify.
		"""
		tensor = sentence_to_tensor(sentence, self.vocab)
		prediction = torch.sigmoid(self.model(tensor))
		return prediction.item()

	def extract_params(self):
		"""
		Given the model.state_dict() from an lstm from pytorch,
		unfold the pytorch dict's matrices into a new dict that facilitates
		accesses.

		"""
		result = {}
		gate_ids = set(zip(range(4), ["i", "f", "g", "o"]))
		tensor_ids = ["i", "h"]
		params = ["weight", "bias"]
		pytorch_dict = self.model.state_dict()
		for rev in range(2):
			revstr = ("_reverse" * rev)
			for param in params:
				for l in range(self.n_layers):
					for tensor_id in tensor_ids:
						tensor = f"rnn.{param}_{tensor_id}h_l{l}" + revstr
						for idx, id in gate_ids:
							result[f"{param}_{tensor_id}{id}{l}"+revstr] = \
								pytorch_dict[tensor][(idx*self.hidden_dim):((idx+1)*self.hidden_dim)]
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

		res[itl] = self.infer_gate(params[Wiil], xt, params[biil],
						params[Whil], ht, params[bhil], torch.sigmoid)

		res[ftl] = self.infer_gate(params[Wifl], xt, params[bifl],
						params[Whfl], ht, params[bhfl], torch.sigmoid)

		res[gtl] = self.infer_gate(params[Wigl], xt, params[bigl],
						params[Whgl], ht, params[bhgl], tanh)

		res[otl] = self.infer_gate(params[Wiol], xt, params[biol],
						params[Whol], ht, params[bhol], torch.sigmoid)

		res[cttl] = torch.add(torch.mul(res[ftl], ct), torch.mul(res[itl], res[gtl]))
		res[httl] = torch.mul(res[otl], tanh(res[cttl]))

		return res

	def infer_gate(self, i_w, x, b_i, h_w, h, b_h, fun):
		input = torch.addmm(torch.unsqueeze(b_i, -1), i_w, x)
		hidden = torch.addmm(torch.unsqueeze(b_h, -1), h_w, h)

		return fun(torch.add(input, hidden))

	def forward_pass(self, embeddings, params):
		result = {} #saves values of states and gates
		revstr = "_reverse"

		for l in range(self.n_layers):
			#random initialization of both hidden and cell states
			result[f"h0{l}"] = torch.zeros(self.hidden_dim, 1).to(self.device)
			result[f"c0{l}"] = torch.zeros(self.hidden_dim, 1).to(self.device)
			#reverse aswell
			result[f"h0{l}{revstr}"] = torch.zeros(self.hidden_dim, 1).to(self.device)
			result[f"c0{l}{revstr}"] = torch.zeros(self.hidden_dim, 1).to(self.device)

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

				result.update(self.infer_gates(input, result[f"h{t}{l}"],
											  result[f"c{t}{l}"], params, l, t))
				result.update(self.infer_gates(input_rev, result[f"h{t}{l}{revstr}"], result[f"c{t}{l}{revstr}"],
											  params, l, t, rev=True))
			return result

	def activations_to_json(self, sentence):
		"""
		Writes the activation values of the model corrresponding
		to a specific input sentence into a .json file.
		Args:
			sentence (string) -> input sentence
		"""
		json_path = "assets/data/"
		encoded = sentence_to_tensor(sentence, self.vocab)
		embeddings = self.model.dropout(self.model.embedding(encoded))
		acts = self.forward_pass(embeddings, self.extract_params())
		filename = "_".join([json_path, "acts", sentence.replace(" ", "_"), ".json"])
		acts_lists = {}
		for (key,value) in acts.items():
 			acts_lists[key] = value.tolist()
		with open(filename, "w+") as actsf:
			json.dump(acts_lists, actsf)
		print("Activations saved to", filename)

if __name__ == "__main__":
	input = sys.argv[1]

	model_path = "assets/models/Bidirectional.pth"
	vocab_path = "assets/data/vocab.pickle"
	toolbox = RNNToolbox(model_path, vocab_path)

	toolbox.activations_to_json(input)

	print(f"{input} -> {toolbox.classify(input)}")
