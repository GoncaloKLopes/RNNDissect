import torch
import os
import json
import sys
sys.path.append("..")
from utils.nlp_utils import sentence_to_tensor


class Extractor:
    """
    Class calculates activation values for different RNN architectures.
    """
    def __init__(self, model_path, model_params, arch="LSTM"):

        self.arch = arch
        self.input_dim = model_params["input_dim"]
        self.output_dim = model_params["output_dim"]
        self.embedding_dim = model_params["embedding_dim"]
        self.hidden_dim = model_params["hidden_dim"]
        self.n_layers = model_params["n_layers"]
        self.bidirectional = model_params["bidirectional"]
        self.dropout = model_params["dropout"]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model_params["model"](self.input_dim, self.embedding_dim,
                                        self.hidden_dim, self.output_dim,
                                        self.n_layers, self.bidirectional, 
                                        self.dropout).to(self.device)

        self.model.load_state_dict(torch.load(model_path))


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

    def activations_to_json(self, sentence, vocab):
        """
        Writes the activation values of the model corrresponding
        to a specific input sentence into a .json file.
        Args:
            sentence (string) -> input sentence
        """
        dir_path = os.path.join(os.path.expanduser("~"), ".rnndissect", "activations")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        encoded = (sentence_to_tensor(sentence, vocab)).to(self.device)
        embeddings = self.model.embedding(encoded)
        acts = self.forward_pass(embeddings, self.extract_params())
        filename = os.path.join(dir_path, 
                                ".".join(["_".join(["acts", sentence.replace(" ", "_")]), "json"]))
        acts_lists = {}
        for (key,value) in acts.items():
            acts_lists[key] = value.tolist()
        with open(filename, "w+") as actsf:
            json.dump(acts_lists, actsf)
        print("Activations saved to", filename)

