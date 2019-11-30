from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import Model
from delfos import helper
import numpy

class weight(layers.Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def build(self, inshape):
		self.w = self.add_weight(name = "w", shape = None, initializer = initializers.Constant(1e-2))
		super().build(inshape)

	def call(self, x):
		return x * self.w

	def compute_output_shape(self, inshape):
		return inshape

class rnnregressor():
	def __init__(self, tab):
		# Embedding Lookup Table
		self.tab = tab
		# Vocabulary Size
		self.voc = tab.shape[0]
		# Embedding Dimension
		self.dim = tab.shape[1]

	def projector(self, tkn):
		return layers.Embedding(self.voc, self.dim, weights = [self.tab], mask_zero = True, trainable = True)(tkn)

	def rnnblock(self, seq):
		# Forward RNN
		fwd = layers.LSTM(self.dim, return_sequences = True, go_backwards = False)(seq[0])
		# Backward RNN
		bwd = layers.LSTM(self.dim, return_sequences = True, go_backwards = True)(seq[1])
		bwd = layers.Lambda(lambda x: backend.reverse(x, axes = 1))(bwd)
		bwd = layers.Masking(mask_value = 0.0)(bwd)
		return fwd, bwd

	def interaction(self, vec):
		# Dot-product Interaction Map
		atn = layers.dot([vec[0], vec[1]], axes = [-1, -1], normalize = False, name = "interaction_map")
		# Atomwise Summation
		atn = layers.Lambda(lambda x: backend.sum(x, axis = -2, keepdims = False), name = "solvent_sum")(atn)
		atn = layers.Lambda(lambda x: backend.sum(x, axis = -1, keepdims = True), name = "solute_sum")(atn)
		return atn

	def buildnn(self, depth):
		tkn_solv = layers.Input((None, ))
		tkn_solu = layers.Input((None, ))
		emb_solv = self.projector(tkn_solv)
		emb_solu = self.projector(tkn_solu)
		# Solvent RNN
		fwd_solv = [None] * depth
		bwd_solv = [None] * depth
		fwd_solu = [None] * depth
		bwd_solu = [None] * depth
		# Stacked
		fwd_solv[0], bwd_solv[0] = self.rnnblock([emb_solv, emb_solv])
		fwd_solu[0], bwd_solu[0] = self.rnnblock([emb_solu, emb_solu])
		for i in range(1, depth):
			fwd_solv[i], bwd_solv[i] = self.rnnblock([fwd_solv[i - 1], bwd_solv[i - 1]])
			fwd_solu[i], bwd_solu[i] = self.rnnblock([fwd_solu[i - 1], bwd_solu[i - 1]])
		rnn_solv = [layers.concatenate([fwd_solv[i], bwd_solv[i]]) for i in range(depth)]
		rnn_solu = [layers.concatenate([fwd_solu[i], bwd_solu[i]]) for i in range(depth)]
		rnn_solv = [weight()(l) for l in rnn_solv]
		rnn_solu = [weight()(l) for l in rnn_solu]
		# Merge
		rnn_solv = layers.add(rnn_solv, name = "solvent_vec")
		rnn_solu = layers.add(rnn_solu, name = "solute_vec")
		# Output
		target = self.interaction([rnn_solv, rnn_solu])
		return Model([tkn_solv, tkn_solu], target)

class gcnregressor():
	def __init__(self, tab):
		self.tab = tab
		self.voc = tab.shape[0]
		self.dim = tab.shape[1]

		self.mtb = numpy.ones(tab.shape)
		self.mtb[0] = numpy.zeros(self.dim)

	def projector(self, tkn):
		emb = layers.Embedding(self.voc, self.dim, weights = [self.tab], trainable = True)(tkn)
		msk = layers.Embedding(self.voc, self.dim, weights = [self.mtb], trainable = False)(tkn)
		return layers.multiply([emb, msk])

	def gcnblock(self, seq, adj):
		reg = regularizers.l2(1e-5)
		gcn = helper.GraphConv(self.dim, use_bias = False, kernel_regularizer = reg, activation = "tanh")([seq, adj])
		return gcn

	def interaction(self, vec):
		# Dot-product Interaction Map
		atn = layers.dot([vec[0], vec[1]], axes = [-1, -1], normalize = False, name = "interaction_map")
		# Atomwise Summation
		atn = layers.Lambda(lambda x: backend.sum(x, axis = -2, keepdims = False), name = "solvent_sum")(atn)
		atn = layers.Lambda(lambda x: backend.sum(x, axis = -1, keepdims = True), name = "solute_sum")(atn)
		return atn

	def buildnn(self, depth):
		tkn_solv = layers.Input((None, ))
		tkn_solu = layers.Input((None, ))
		adj_solv = layers.Input((None, None))
		adj_solu = layers.Input((None, None))
		emb_solv = self.projector(tkn_solv)
		emb_solu = self.projector(tkn_solu)
		# Solvent RNN
		gcn_solv = [None] * depth
		gcn_solu = [None] * depth
		# Stacked
		gcn_solv[0] = self.gcnblock(emb_solv, adj_solv)
		gcn_solu[0] = self.gcnblock(emb_solu, adj_solu)
		for i in range(1, depth):
			gcn_solv[i] = self.gcnblock(gcn_solv[i - 1], adj_solv)
			gcn_solu[i] = self.gcnblock(gcn_solu[i - 1], adj_solu)
		gcn_solv = [weight()(l) for l in gcn_solv]
		gcn_solu = [weight()(l) for l in gcn_solu]
		# Merge
		gcn_solv = layers.add(gcn_solv, name = "solvent_vec")
		gcn_solu = layers.add(gcn_solu, name = "solute_vec")
		# Output
		target = self.interaction([gcn_solv, gcn_solu])
		return Model([tkn_solv, adj_solv, tkn_solu, adj_solu], target)
