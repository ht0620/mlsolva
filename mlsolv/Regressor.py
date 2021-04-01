from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import Model
import spektral
import numpy

class GCNRegressor():
	def __init__(self, lookup, mask):
		assert lookup.shape == mask.shape

		self.lookup = lookup
		self.mask = mask

		self.voc_num = lookup.shape[0]
		self.emb_dim = lookup.shape[1]

	def Embedder(self, seq_solv, seq_solu, reg):
		emb_solv = layers.Embedding(self.voc_num, self.emb_dim, weights = [self.lookup], trainable = True, embeddings_regularizer = reg)(seq_solv)
		emb_solu = layers.Embedding(self.voc_num, self.emb_dim, weights = [self.lookup], trainable = True, embeddings_regularizer = reg)(seq_solu)
		self.msk_solv = layers.Embedding(self.voc_num, self.emb_dim, weights = [self.mask], trainable = False)(seq_solv)
		self.msk_solu = layers.Embedding(self.voc_num, self.emb_dim, weights = [self.mask], trainable = False)(seq_solu)
		emb_solv = layers.multiply([emb_solv, self.msk_solv])
		emb_solu = layers.multiply([emb_solu, self.msk_solu])
		return emb_solv, emb_solu

	def GCNBlock(self, gph_solv, adj_solv, gph_solu, adj_solu, reg):
		gcn_solv = spektral.layers.GCSConv(self.emb_dim, use_bias = False, activation = "tanh", kernel_regularizer = reg)([gph_solv, adj_solv])
		gcn_solu = spektral.layers.GCSConv(self.emb_dim, use_bias = False, activation = "tanh", kernel_regularizer = reg)([gph_solu, adj_solu])
		gcn_solv = layers.multiply([gcn_solv, self.msk_solv])
		gcn_solu = layers.multiply([gcn_solu, self.msk_solu])
		return gcn_solv, gcn_solu

	def ApplyWeight(self, gph_solv, gph_solu):
		return TrainableWeight()(gph_solv), TrainableWeight()(gph_solu)

	def Interaction(self, vec_solv, vec_solu):
		atn = layers.dot([vec_solv, vec_solu], axes = [-1, -1], normalize = False, name = "interaction_map")
		atn = layers.Lambda(lambda x: backend.sum(x, axis = -2, keepdims = False), name = "solvent_sum")
		atn = layers.Lambda(lambda x: backend.sum(x, axis = -1, keepdims = True), name = "solute_sum")
		return atn

	def BuildNN(self, depth, l2):
		reg = regularizers.l2(l2)

		# Input
		seq_solv, seq_solu = layers.Input((None,)), layers.Input((None, ))
		adj_solv, adj_solu = layers.Input((None, None)), layers.Input((None, None))

		# Embedding
		emb_solv, emb_solu = self.Embedder(seq_solv, seq_solu, reg)

		# GCN stack
		gcn_solv, gcn_solu = [None] * depth, [None] * depth
		gcn_solv[0], gcn_solu[0] = self.GCNBlock(emb_solv, adj_solv, emb_solu, adj_solu, reg)
		for i in range(1, depth):
			gcn_solv[i], gcn_solu[i] = self.GCNBlock(gcn_solv[i - 1], adj_solv, gcn_solu[i - 1], adj_solu, reg)

		# Weighted sum
		for i in range(0, depth):
			gcn_solv[i], gcn_solu[i] = self.ApplyWeight(gcn_solv[i], gcn_solu[i])

		vec_solv = layers.add(gcn_solv)
		vec_solu = layers.add(gcn_solu)

		# Feature vector
		vec_solv = spektral.layers.GlobalSumPool(name = "solvent_feature")(vec_solv)
		vec_solu = spektral.layers.GlobalSumPool(name = "solute_feature")(vec_solu)
		target = layers.dot([vec_solv, vec_solu], axes = [-1, -1], normalize = False)

		#target = self.Interaction(vec_solv, vec_solu)
		
		return Model([seq_solv, adj_solv, seq_solu, adj_solu], target)

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