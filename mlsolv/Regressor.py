from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import Model
import spektral
import numpy

def ApplyWeight(seq_solv, seq_solu):
	return TrainableWeight()(seq_solv), TrainableWeight()(seq_solu)

def Interaction(vec_solv, vec_solu):
	atn = layers.dot([vec_solv, vec_solu], axes = [-1, -1], normalize = False, name = "interaction_map")
	atn = layers.Lambda(lambda x: backend.sum(x, axis = -2, keepdims = False), name = "solvent_sum")(atn)
	atn = layers.Lambda(lambda x: backend.sum(x, axis = -1, keepdims = True), name = "solute_sum")(atn)
	return atn

class Regressor():
	def __init__(self, lookup, mask):
		assert lookup.shape == mask.shape

		self.lookup = lookup
		self.mask = mask

		self.voc_num = lookup.shape[0]
		self.emb_dim = lookup.shape[1]

	def Embedder(self, encoder, seq_solv, seq_solu, reg):
		emb_solv = layers.Embedding(self.voc_num, self.emb_dim, weights = [self.lookup], trainable = True, mask_zero = True, embeddings_regularizer = reg)(seq_solv)
		emb_solu = layers.Embedding(self.voc_num, self.emb_dim, weights = [self.lookup], trainable = True, mask_zero = True, embeddings_regularizer = reg)(seq_solu)

		if encoder == "gcn":
			self.msk_solv = layers.Embedding(self.voc_num, self.emb_dim, weights = [self.mask], trainable = False)(seq_solv)
			self.msk_solu = layers.Embedding(self.voc_num, self.emb_dim, weights = [self.mask], trainable = False)(seq_solu)
			emb_solv = layers.multiply([emb_solv, self.msk_solv])
			emb_solu = layers.multiply([emb_solu, self.msk_solu])

		return emb_solv, emb_solu

	def GCNBlock(self, gph_solv, adj_solv, gph_solu, adj_solu, reg):
		gcn_solv = spektral.layers.GCSConv(self.emb_dim, use_bias = True, activation = "tanh", kernel_regularizer = reg, bias_regularizer = reg)([gph_solv, adj_solv])
		gcn_solu = spektral.layers.GCSConv(self.emb_dim, use_bias = True, activation = "tanh", kernel_regularizer = reg, bias_regularizer = reg)([gph_solu, adj_solu])
		gcn_solv = layers.multiply([gcn_solv, self.msk_solv])
		gcn_solu = layers.multiply([gcn_solu, self.msk_solu])
		return gcn_solv, gcn_solu

	def RNNBlock(self, seq_solv, seq_solu, reg, backward):
		rnn_solv = layers.LSTM(self.emb_dim, return_sequences = True, kernel_regularizer = reg, recurrent_regularizer = reg, bias_regularizer = reg, go_backwards = backward)(seq_solv)
		rnn_solu = layers.LSTM(self.emb_dim, return_sequences = True, kernel_regularizer = reg, recurrent_regularizer = reg, bias_regularizer = reg, go_backwards = backward)(seq_solu)
		if backward == True:
			rnn_solv = layers.Lambda(lambda x: backend.reverse(x, axes = 1))(rnn_solv)
			rnn_solu = layers.Lambda(lambda x: backend.reverse(x, axes = 1))(rnn_solu)
			rnn_solv = layers.Masking(mask_value = 0.0)(rnn_solv)
			rnn_solu = layers.Masking(mask_value = 0.0)(rnn_solu)
		return rnn_solv, rnn_solu

	def BuildNN(self, encoder = "gcn", depth = 3, l2 = 1e-6):
		reg = regularizers.l2(l2)

		# Input
		seq_solv, seq_solu = layers.Input((None,)), layers.Input((None, ))

		if encoder == "gcn":
			adj_solv, adj_solu = layers.Input((None, None)), layers.Input((None, None))

		# Embedding
		emb_solv, emb_solu = self.Embedder(encoder, seq_solv, seq_solu, reg)

		# GCN stack
		if encoder == "gcn":
			enc_solv, enc_solu = [None] * depth, [None] * depth
			enc_solv[0], enc_solu[0] = self.GCNBlock(emb_solv, adj_solv, emb_solu, adj_solu, reg)
			for i in range(1, depth):
				enc_solv[i], enc_solu[i] = self.GCNBlock(enc_solv[i - 1], adj_solv, enc_solu[i - 1], adj_solu, reg)

		# RNN stack
		if encoder == "rnn":
			fwd_solv, fwd_solu = [None] * depth, [None] * depth
			bwd_solv, bwd_solu = [None] * depth, [None] * depth
			fwd_solv[0], fwd_solu[0] = self.RNNBlock(emb_solv, emb_solu, reg, backward = False)
			bwd_solv[0], bwd_solu[0] = self.RNNBlock(emb_solv, emb_solu, reg, backward = True)
			for i in range(1, depth):
				fwd_solv[i], fwd_solu[i] = self.RNNBlock(fwd_solv[i - 1], fwd_solu[i - 1], reg, backward = False)
				bwd_solv[i], bwd_solu[i] = self.RNNBlock(bwd_solv[i - 1], bwd_solu[i - 1], reg, backward = True)
			enc_solv = [layers.concatenate([fwd_solv[i], bwd_solv[i]]) for i in range(depth)]
			enc_solu = [layers.concatenate([fwd_solu[i], bwd_solu[i]]) for i in range(depth)]
				
		# Weighted sum
		for i in range(0, depth):
			enc_solv[i], enc_solu[i] = ApplyWeight(enc_solv[i], enc_solu[i])

		vec_solv = layers.add(enc_solv)
		vec_solu = layers.add(enc_solu)

		# Feature vector
		#vec_solv = spektral.layers.GlobalSumPool(name = "solvent_feature")(vec_solv)
		#vec_solu = spektral.layers.GlobalSumPool(name = "solute_feature")(vec_solu)
		#target = layers.dot([vec_solv, vec_solu], axes = [-1, -1], normalize = False)

		target = Interaction(vec_solv, vec_solu)
		
		if encoder == "gcn":
			return Model([seq_solv, adj_solv, seq_solu, adj_solu], target)
		
		if encoder == "rnn":
			return Model([seq_solv, seq_solu], target)

class TrainableWeight(layers.Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def build(self, inshape):
		self.w = self.add_weight(name = "w", shape = None, initializer = initializers.Constant(1e-2))
		super().build(inshape)

	def call(self, x):
		return x * self.w

	def compute_output_shape(self, inshape):
		return inshape