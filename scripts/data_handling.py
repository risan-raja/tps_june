# import pandas
import pickle
import ray
ray.init()
import modin.pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


data = pd.read_csv("../data/data.csv", index_col=0)

int64_cols = data.select_dtypes("int64")
for c in int64_cols:
    data[c] = data[c].astype("category")

fl16_cols = data.select_dtypes(include="float64")

for c in fl16_cols:
    data[c] = data[c].astype(np.float32)

na_mat = data.isna()

miss_row = na_mat.sum(axis=1)

data["missing_cols"] = miss_row
data.to_pickle("cooked.pkl")



def gen_sparse_data_modin(dpkl) -> pd.DataFrame:
    cats = [x for x in dpkl.columns if "F_2" in x]
    # for c in cats:
    #     dpkl = feature_one_hot(c)
    spar = pd.DataFrame()
    cat_pd = dpkl.loc[:, cats].copy()

    for fi in cats:
        new_n = fi.replace("_", "") + "_"
        f_one_hot = pd.get_dummies(dpkl[fi], prefix=new_n)
        spar_ = f_one_hot.astype(pd.SparseDtype(np.uint8, fill_value=0))
        for c in spar_.columns:
            spar[c] = spar_[c]
    dpkl = dpkl.drop(cats, axis=1)
    for c in spar.columns:
        dpkl[c] = spar[c]
    return dpkl.copy()
sparsed = gen_sparse_data_modin(data) 
sparsed.to_pickle("cooked_sparsely.pkl")


def get_info(given_data):
    print(f"dense_type: {given_data.memory_usage(index=True, deep=True).sum() / 10 ** 6}MB")
    sparse_data = gen_sparse_data_modin(given_data)
    print(sparse_data.dtypes.value_counts())
    print(f"sparse_type: {sparse_data.memory_usage(index=True, deep=True).sum() / 10 ** 6}MB")
    sparse_data_np: np.ndarray = sparse_data.to_numpy()
    print(f"numpy_type: {sparse_data_np.nbytes / 10 ** 6}MB")


# create dense copy
# if __name__ == '__main__':
#     with open("data.pkl", "rb") as fp:
#         dataset = pickle.load(fp)
#         # run(dataset)
#         modin_dense_df = pd.DataFrame(dataset)
#         # modin_sparse_df = pf.DataFrame()

#         # sparse_dataset = gen_sparse_data(dataset)
#         # for c in tqdm(dataset.columns):
#         #     modin_dense_df[c] = dataset[c]
#         # for c in sparse_dataset.columns:
#         #    modin_sparse_df[c] = sparse_dataset[c]
#         modin_dense_df.to_pickle('modin_dense.pkl')
#         # modin_sparse_df.to_pickle('modin_sparse.pkl')

# #
# import modin.config as cfg
# # cfg.Engine.put('native')
# # cfg.Backend.put('omnisci')
# # cfg.IsExperimental.put(True)
# import modin.pandas as pd
#
# warnings.filterwarnings("ignore")
#
# data = pd.read_csv('data.csv', index_col=0)
# data.info()
