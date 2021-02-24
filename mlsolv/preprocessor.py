from rdkit.Chem.PandasTools import AddMoleculeColumnToFrame
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from spektral.utils import normalized_adjacency
from spektral.utils import localpooling_filter
from mol2vec.features import mol2alt_sentence
from keras.preprocessing import sequence
import pandas
import pickle
import numpy

class preprocessor():
	def __init__(self, db):
		self.db = db

	def tokenize(self, token):
		# Load Tokenizer
		with open(token, "rb") as tk:
			tokenizer = pickle.load(tk)

		for smi in ["solvent", "solute"]:
			# Generate RDKit Mols
			AddMoleculeColumnToFrame(self.db, smilesCol = smi, molCol = "%s_rdmol" %smi)
			# Generate Morgan IDs
			self.db["%s_morgan" %smi] = self.db.apply(lambda x:
					mol2alt_sentence(x["%s_rdmol" %smi], radius = 0), axis = 1)
			# Generate Tokens
			self.db["%s_token" %smi] = tokenizer.texts_to_sequences(self.db["%s_morgan" %smi])

	def getadjacency(self):
		for smi in ["solvent", "solute"]:
			self.db["%s_adj" %smi] = self.db.apply(lambda x:
					GetAdjacencyMatrix(x["%s_rdmol" %smi]), axis = 1)
			self.db["%s_adj" %smi] = self.db.apply(lambda x:
					localpooling_filter(x["%s_adj" %smi]), axis = 1)
#					normalized_adjacency(x["%s_adj" %smi]), axis = 1)

	def importdata(self):
		x_solv = sequence.pad_sequences(self.db["solvent_token"], padding = "post")
		x_solu = sequence.pad_sequences(self.db["solute_token"], padding = "post")

		adj_solv = numpy.zeros((x_solv.shape[0], x_solv.shape[1], x_solv.shape[1]))
		adj_solu = numpy.zeros((x_solu.shape[0], x_solu.shape[1], x_solu.shape[1]))

		for i in range(adj_solv.shape[0]):
			offset = self.db["solvent_adj"][i].shape[0]
			adj_solv[i, :offset, :offset] = self.db["solvent_adj"][i]

		for i in range(adj_solu.shape[0]):
			offset = self.db["solute_adj"][i].shape[0]
			adj_solu[i, :offset, :offset] = self.db["solute_adj"][i]

		y = self.db["deltaG"].values

		return x_solv, adj_solv, x_solu, adj_solu, y
