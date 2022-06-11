import pickle
import pandas as pd
cdef sv(df: pd.DataFrame, name: str):
    with open(f'{name}.pkl', 'wb+') as fp:
        pickle.dump(df, fp)
cdef load():
    with open("data.pkl", "rb") as fp:
        dpkl = pickle.load(fp)
    return dpkl
