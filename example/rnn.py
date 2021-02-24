from tensorflow.keras import optimizers
from tensorflow.keras import backend
from tensorflow.keras import Model
from mlsolv.preprocessor import preprocessor
from mlsolv.predictor import gcnregressor
from mlsolv.predictor import rnnregressor
import pandas
import numpy

edim = 128

# Pretrained Data
db = pandas.read_csv("db/merged_db.csv")
tbl = numpy.load("emb/emat.npy")
tkn = "emb/token.pkl"

# Input Preparation
pre = preprocessor(db)
pre.tokenize(tkn)
pre.getadjacency()

x_solv, adj_solv, x_solu, adj_solu, y = pre.importdata()

# BiLM Regression
optim = optimizers.RMSprop(lr = 1e-3, rho = 0.9)
model = rnnregressor(tbl).buildnn(depth = 3)
model.compile(loss = "mse", optimizer = optim)

model.fit((x_solv, x_solu), y, batch_size = 32, epochs = 100, shuffle = True)
