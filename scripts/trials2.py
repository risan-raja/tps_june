#!/usr/bin/env python
# coding: utf-8

import gc
import pickle
import warnings
from collections import defaultdict

from dask.distributed import Client
from tqdm import tqdm

# try:
#     ray.init()
# except:
#     pass
# import modin.pandas as pd
# from data_handling import get_info
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn import set_config
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    Lasso,
    LassoLars,
    LassoLarsIC,
    RANSACRegressor,
    RidgeCV,
    TweedieRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearnex import patch_sklearn

# import ray


pd.options.display.max_columns = 90
pd.options.display.max_rows = 90
warnings.filterwarnings("ignore")
set_config(display="diagram")
dirty_results = defaultdict(dict)
reference_metadata = defaultdict(dict)
clean_results = defaultdict(dict)

patch_sklearn()
iter_ = 10000
tol = 0.000001


def gen_stack():
    # Category Selector
    # cat_selector = make_column_selector(dtype_exclude=np.float32)
    # Number Selector
    # cat_scaler = OneHotEncoder(sparse=True)
    # linear_prep = ColumnTransformer(
    #     transformers=[("num", numeric_scaler, numerical_selector)]
    # )
    # category_transformer
    # Feature Selector
    numerical_selector = make_column_selector(dtype_exclude=np.uint8)
    sel = SelectFromModel(estimator=ElasticNet(precompute=True), threshold="median")
    numeric_scaler = StandardScaler()
    tree_prep = ColumnTransformer(transformers=[("num", numeric_scaler, numerical_selector)])
    lasso_linear_prep = ColumnTransformer(transformers=[("num", numeric_scaler, numerical_selector)])
    modis = [
        make_pipeline(lasso_linear_prep, sel, LassoLarsIC(normalize=False, precompute=True, criterion="bic")),
        make_pipeline(lasso_linear_prep, sel, ARDRegression(n_iter=1000, compute_score=True, tol=tol)),
        make_pipeline(
            lasso_linear_prep, sel, BayesianRidge(lambda_init=0.001, n_iter=iter_, tol=tol, compute_score=True)
        ),
        make_pipeline(lasso_linear_prep, sel, Lasso(precompute=True, max_iter=iter_, tol=tol, selection="cyclic")),
        make_pipeline(lasso_linear_prep, sel, LassoLars(precompute=True, max_iter=iter_)),
        make_pipeline(lasso_linear_prep, sel, TweedieRegressor(power=0)),
        make_pipeline(
            lasso_linear_prep,
            sel,
            RANSACRegressor(
                min_samples=500,
                base_estimator=LassoLarsIC(normalize=False, precompute=True, criterion="aic"),
                max_trials=10000,
            ),
        ),
        make_pipeline(
            lasso_linear_prep,
            sel,
            ElasticNet(
                precompute=True,
            ),
        ),
        make_pipeline(tree_prep, sel, HistGradientBoostingRegressor(max_iter=1000, max_depth=500)),
        make_pipeline(tree_prep, sel, GradientBoostingRegressor(random_state=0, max_depth=30)),
        make_pipeline(tree_prep, sel, DecisionTreeRegressor()),
        make_pipeline(
            tree_prep,
            sel,
            ExtraTreesRegressor(n_jobs=-1),
        ),
        make_pipeline(tree_prep, sel, AdaBoostRegressor(base_estimator=Lasso(precompute=True))),
    ]
    stacked_estimators = []
    for q in modis:
        estimator_name = q[2].__class__.__name__
        stacked_estimators.append((estimator_name, q))
    learning_stack = StackingRegressor(estimators=stacked_estimators, cv=3, n_jobs=-1, final_estimator=RidgeCV())
    return learning_stack


def save_pipeline(c, p):
    with open(f"stacking_models/stack_{c}.pkl", "wb+") as file_output:
        pickle.dump(p, file_output)


def get_data_feed(c, x_y):
    training_features = x_y.drop([c], axis=1)
    y = x_y[c]
    x_temp_train, x_temp_test, y_temp_train, y_temp_train = train_test_split(
        training_features, y, test_size=0.2, random_state=0
    )
    return x_temp_train, x_temp_test, y_temp_train, y_temp_train


training_targets = [
    "F_1_0",
    "F_1_1",
    "F_1_2",
    "F_1_3",
    "F_1_4",
    "F_1_5",
    "F_1_6",
    "F_1_7",
    "F_1_8",
    "F_1_9",
    "F_1_10",
    "F_1_11",
    "F_1_12",
    "F_1_13",
    "F_1_14",
    "F_3_0",
    "F_3_1",
    "F_3_2",
    "F_3_3",
    "F_3_4",
    "F_3_5",
    "F_3_6",
    "F_3_7",
    "F_3_8",
    "F_3_9",
    "F_3_10",
    "F_3_11",
    "F_3_12",
    "F_3_13",
    "F_3_14",
    "F_3_15",
    "F_3_16",
    "F_3_17",
    "F_3_18",
    "F_3_19",
    "F_3_20",
    "F_3_21",
    "F_3_22",
    "F_3_23",
    "F_3_24",
    "F_4_0",
    "F_4_1",
    "F_4_2",
    "F_4_3",
    "F_4_4",
    "F_4_5",
    "F_4_6",
    "F_4_7",
    "F_4_8",
    "F_4_9",
    "F_4_10",
    "F_4_11",
    "F_4_12",
    "F_4_13",
    "F_4_14",
]


def gen_sparse_data(dpkl) -> pd.DataFrame:
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
    return dpkl



if __name__ == "__main__":
    cli = Client(processes=False)
    def run(dataset_db):
        trd = dataset_db[dataset_db.missing_cols == 0].copy()
        x_y = trd.drop(["missing_cols"], axis=1)
        for cl in tqdm(training_targets):
            training_features = x_y.drop([cl], axis=1)
            y = x_y[cl]
            x_train, x_test, y_train, y_test = train_test_split(training_features, y, test_size=0.2, random_state=0)
            gc.collect()
            gc.collect()
            with parallel_backend("dask"):
                new_stack = gen_stack()
                new_stack.fit(x_train, y_train)
            yp = new_stack.predict(x_test)
            with open("results_mod", "a") as fp:
                fp.write(f"{cl} : {mean_squared_error(yp, y_test)}\n\n")
            print(mean_squared_error(yp, y_test))
            save_pipeline(cl, new_stack)

    # with parallel_backend("multiprocessing", n_jobs=-1):
    dataset = pd.read_pickle("cooked.pkl")
    run(dataset)
