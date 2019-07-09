import torch
import os
import json
import sys

from ..utils.nlp_utils import sentence_to_tensor
from ..settings import *


class Extractor:
    """
    Class calculates activation values for different RNN architectures.
    """
    def __init__(self, config, model):

        self.arch = config.arch
        self.output_dim = config.output_dim
        self.embedding_dim = config.d_embed
        self.hidden_dim = config.d_hidden
        self.n_layers = config.n_layers
        self.bidir = config.bidir
        self.model = model

    def extract_params(self):
        """
        Given the model.state_dict() from an lstm from pytorch,
        unfold the pytorch dict's matrices into a new dict that facilitates
        accesses.
        """
        result = {}
        if self.arch == "RNN":
            tensor_ids = ["i", "h"] 
            params = ["weight", "bias"]
            pytorch_dict = self.model.state_dict()
            for rev in range(2 - (1 - self.bidir)):
                revstr = ("_reverse" * rev)
                for param in params:
                    for l in range(self.n_layers):
                        for tensor_id in tensor_ids:
                            tensor = f"rnn.{param}_{tensor_id}h_l{l}" + revstr
                            result[f"{param}_{tensor_id}h{l}"+revstr] = pytorch_dict[tensor]

        if self.arch == "LSTM":
            gate_ids = set(zip(range(4), ["i", "f", "g", "o"]))
            tensor_ids = ["i", "h"]
            params = ["weight", "bias"]
            pytorch_dict = self.model.state_dict()
            for rev in range(2 - (1 - self.bidir)):
                revstr = ("_reverse" * rev)
                for param in params:
                    for l in range(self.n_layers):
                        for tensor_id in tensor_ids:
                            tensor = f"rnn.{param}_{tensor_id}h_l{l}" + revstr
                            for idx, id in gate_ids:
                                result[f"{param}_{tensor_id}{id}{l}"+revstr] = \
                                    pytorch_dict[tensor][(idx*self.hidden_dim):((idx+1)*self.hidden_dim)]
        return result

    def infer_rnn_h(self, xt, ht, params, l, t, rev=False):
        if self.arch == "LSTM":
            raise Exception("Wrong function, use infer_lstm_gates().")
        res = {}
        tanh = torch.nn.Tanh()

        wstr = "weight_"
        bstr = "bias_"
        revstr = "_reverse"

        wihl = wstr + f"ih{l}" + (revstr * rev)
        whhl = wstr + f"hh{l}" + (revstr * rev)
        bihl = bstr + f"ih{l}" + (revstr * rev)
        bhhl = bstr + f"hh{l}" + (revstr * rev)

        # tt is t+1 in variablenameland
        httl = f"h{t+1}{l}" + (revstr * rev)

        inpt = torch.addmm(torch.unsqueeze(params[bihl], -1),
                           params[wihl],
                           xt)
        hidden = torch.addmm(torch.unsqueeze(params[bhhl], -1),
                             params[whhl],
                             ht)

        res[httl] = tanh(torch.add(inpt, hidden))
        return res

    def infer_lstm_gates(self, xt, ht, ct, params, l, t, rev=False):
        """
        Assume the following dimension:
        - xt.shape = [EMBEDDING_DIM x 1]
        - ht.shape = [HIDDEN_DIM x 1]
        - ct.shape = [HIDDEN_DIM x 1]
        - weights[Wi*] = [HIDDEN_DIM x EMBEDDING_DIM]
        - weights[Wh*] = [HIDDEN_DIM x HIDDEN_DIM]
        - biases[*] = [HIDDEN_DIM]
        """

        if self.arch == "RNN":
            raise Exception("Wrong function, use infer_rnn_h().")

        res = {}
        tanh = torch.nn.Tanh()

        wstr = "weight_"
        bstr = "bias_"
        revstr = "_reverse"

        # how do I loop this
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
        # tt is t+1 in variablenameland
        cttl = f"c{t+1}{l}" + (revstr * rev)
        httl = f"h{t+1}{l}" + (revstr * rev)

        res[itl] = self.infer_lstm_gate(params[Wiil], xt, params[biil],
                                        params[Whil], ht, params[bhil],
                                        torch.sigmoid)

        res[ftl] = self.infer_lstm_gate(params[Wifl], xt, params[bifl],
                                        params[Whfl], ht, params[bhfl],
                                        torch.sigmoid)

        res[gtl] = self.infer_lstm_gate(params[Wigl], xt, params[bigl],
                                        params[Whgl], ht, params[bhgl],
                                        tanh)

        res[otl] = self.infer_lstm_gate(params[Wiol], xt, params[biol],
                                        params[Whol], ht, params[bhol],
                                        torch.sigmoid)

        res[cttl] = torch.add(torch.mul(res[ftl], ct),
                              torch.mul(res[itl],
                              res[gtl]))
        res[httl] = torch.mul(res[otl], tanh(res[cttl]))

        return res

    def infer_lstm_gate(self, i_w, x, b_i, h_w, h, b_h, fun):
        if self.arch == "RNN":
            raise Exception("Wrong function, use infer_rnn_h().")

        inpt = torch.addmm(torch.unsqueeze(b_i, -1), i_w, x)
        hidden = torch.addmm(torch.unsqueeze(b_h, -1), h_w, h)

        return fun(torch.add(inpt, hidden))

    def forward_pass(self, embeddings, params):
        result = {}  # saves values of states and gates
        revstr = "_reverse"
        for l in range(self.n_layers):
            # random initialization of both hidden and cell states
            result[f"h0{l}"] = torch.zeros(self.hidden_dim, 1).to(DEVICE)
            result[f"c0{l}"] = torch.zeros(self.hidden_dim, 1).to(DEVICE)
            # reverse aswell
            if self.bidir:
                result[f"h0{l}{revstr}"] = torch.zeros(self.hidden_dim, 1).to(DEVICE)
                result[f"c0{l}{revstr}"] = torch.zeros(self.hidden_dim, 1).to(DEVICE)

            for t in range(len(embeddings)):
                # if layer > 1, then the input isn't the model input,
                # but the hidden state from the previous
            # layer
                reverse_t = len(embeddings)-(t+1)
                if l == 0:
                    inpt = torch.transpose(embeddings[t], 0 , 1)
                    if self.bidir:
                        inpt_rev = torch.transpose(embeddings[reverse_t], 0, 1)
                else:
                    if self.bidir:
                        inpt = torch.cat((result[f"h{t+1}{l-1}"], result[f"h{reverse_t+1}{l-1}{revstr}"]))
                        inpt_rev = torch.cat((result[f"h{reverse_t+1}{l-1}"],
                                              result[f"h{t+1}{l-1}{revstr}"]))
                    else:
                        inpt = result[f"h{t+1}{l-1}"]
                if self.arch == "RNN":
                    result. update(self.infer_rnn_h(inpt, result[f"h{t}{l}"], params, l, t))

                    if self.bidir:
                        result.update(self.infer_rnn_h(inpt_rev,
                                                       result[f"h{t}{l}{revstr}"],
                                                       params, l, t, rev=True))
                else:
                    result.update(self.infer_lstm_gates(inpt, result[f"h{t}{l}"],
                                                        result[f"c{t}{l}"], params, l, t))
                    if self.bidir:
                        result.update(self.infer_lstm_gates(inpt_rev, 
                                                            result[f"h{t}{l}{revstr}"],
                                                            result[f"c{t}{l}{revstr}"],
                                                            params, l, t, rev=True))

        return result

    def activations_to_json(self, sentence, fname):
        """
        Writes the activation values of the model corrresponding
        to a specific input sentence into a .json file.
        Args:
            sentence (string) -> input sentence
        """
        dir_path = os.path.join(os.path.expanduser("~"), ".rnndissect", "activations")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        encoded = (sentence_to_tensor(sentence)).to(DEVICE)
        embeddings = self.model.embed(encoded)
        acts = self.forward_pass(embeddings, self.extract_params())
        filename = os.path.join(dir_path,
                                ".".join(["_".join(["acts", fname]), "json"]))
        acts_lists = {}
        for (key, value) in acts.items():
            acts_lists[key] = value.tolist()
        with open(filename, "w+") as actsf:
            json.dump(acts_lists, actsf)
        print("Activations saved to", filename)
