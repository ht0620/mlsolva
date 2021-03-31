from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
from spektral.utils import convolution
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from rdkit import Chem
import pandas
import pickle
import numpy

class InputPreparation():
    def __init__(self, db):
        self.db = db
        for col in ["solvent", "solute"]:
            self.db["%s_rdmol" %col] = self.db.apply(lambda x: Chem.MolFromSmiles(x[col]), axis = 1)


    def GenSequence(self, token_path):
        tokenizer = pickle.load(open(token_path, "rb"))
        for col in ["solvent", "solute"]:
            # Morgan IDs
            self.db["%s_morgan" %col] = self.db.apply(lambda x: MolToSentence(x["%s_rdmol" %col]), axis = 1)
            # Tokenize
            self.db["%s_token" %col] = tokenizer.texts_to_sequences(self.db["%s_morgan" %col])
    
    def GetAdjacency(self):
        for col in ["solvent", "solute"]:
            # Adjacency matrix (default)
            self.db["%s_adj" %col] = self.db.apply(lambda x: Chem.GetAdjacencyMatrix(x["%s_rdmol" %col]), axis = 1)
            # Normalized adjacency (for GCSconv)
            self.db["%s_gcs" %col] = self.db.apply(lambda x: convolution.normalized_adjacency(x["%s_adj" %col]), axis = 1)
            # GCN filter (for GCNconv)
            self.db["%s_gcn" %col] = self.db.apply(lambda x: convolution.gcn_filter(x["%s_adj" %col]), axis = 1)

    def ImportData(self):
        seq_solv = sequence.pad_sequences(self.db["solvent_token"], padding = "pre")
        seq_solu = sequence.pad_sequences(self.db["solute_token" ], padding = "pre")

        assert seq_solv.shape[0] == seq_solu.shape[0]

        len_data = seq_solv.shape[0]

        dim_solv = seq_solv.shape[1]
        dim_solu = seq_solu.shape[1]

        adj_solv = numpy.zeros((len_data, dim_solv, dim_solv))
        adj_solu = numpy.zeros((len_data, dim_solu, dim_solu))

        for idx in self.db.index:
            adj_solv[idx] = ZeroPadding2D(self.db.at[idx, "solvent_gcs"], dim_solv)
            adj_solu[idx] = ZeroPadding2D(self.db.at[idx, "solute_gcs" ], dim_solu)

        target = self.db["deltaG"].values

        return (seq_solv, adj_solv, seq_solu, adj_solu), target


def ZeroPadding2D(adj, dim):
    padder = ((dim - adj.shape[0], 0), (dim - adj.shape[0], 0))
    return numpy.pad(adj, padder, mode = "constant", constant_values = 0)

## This function is taken from mol2vec, https://github.com/samoturk/mol2vec
def MolToSentence(mol, radius = 0):
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element

    identifiers_alt = []
    for atom in dict_atoms:
        for r in radii:
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)