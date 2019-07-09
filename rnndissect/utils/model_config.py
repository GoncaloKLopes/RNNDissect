class ModelConfig:
    """Encapsulates model configuration."""
    def __init__(self, id, d_hidden, vocab_size, d_embed,
                 batch_size, n_layers, bidir,
                 arch, output_dim, nonlin=""):
        self.id = id
        self.d_hidden = d_hidden
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.nonlin = nonlin
        self.bidir = bidir
        self.arch = arch
        self.output_dim = output_dim

    def __str__(self):
        return "Model Config\n" + 12 * ("=") + "\n" +\
               "d_hidden = " + str(self.d_hidden) + "\n" +\
               "vocab_size = " + str(self.vocab_size) + "\n" +\
               "d_embed = " + str(self.d_embed) + "\n" +\
               "batch_size = " + str(self.batch_size) + "\n" +\
               "n_layers = " + str(self.n_layers) + "\n" +\
               "nonlin = " + str(self.nonlin) + "\n" +\
               "bidir = " + str(self.bidir) + "\n" +\
               "arch = " + self.arch + "\n" +\
               "output_dim = " + self.output_dim
