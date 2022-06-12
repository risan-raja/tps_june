import pandas as pd
import pickle
from trials import gen_sparse_data
with open("data.pkl", "rb") as fp:
    dataset: pd.DataFrame = pickle.load(fp)
    sp = gen_sparse_data(dataset)
    sp.to_csv('sparse_data.csv')