import pickle

import dpctl
import numpy as np
import pandas as pd
from numba import njit
from numba import jit


def conn():
    with open("data.pkl", "rb") as fp:
        dpkl = pickle.load(fp)
    return dpkl


@njit
def bring_slice(_ndf, missing):
    return _ndf[missing, :]


class DB:
    def __init__(self):
        self.df: pd.DataFrame = conn()
        self.ndf: np.ndarray = self.df.to_numpy()
        self.missing_data = self.pkload('missing_data')
        # self.missing_data: np.array = self.missing_data.to_numpy()
        # removing the missing_cols

    @staticmethod
    def sv(data: pd.DataFrame, name: str):
        with open(f"{name}.pkl", "wb+") as fp:
            pickle.dump(data, fp)

    @staticmethod
    def pkload(fn):
        with open(fn + ".pkl", "rb") as fp:
            data = pickle.load(fp)
        return data

    def n_cols_missing_idx(self, n):
        return self.missing_data[self.missing_data == n].index.to_numpy()

    def data_missing_cols(self, n):
        _missing: np.array = self.n_cols_missing_idx(n)
        n_slice = None
        # with dpctl.device_context("cpu"):
        n_slice = bring_slice(self.ndf, _missing)
        return n_slice


if __name__ == '__main__':
    pass
