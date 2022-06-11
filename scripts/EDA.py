# import pickle

# from xgboost import XGBRegressor

# with open('data.pkl', 'rb') as fp:
#     data = pickle.load(fp)
# xgb = XGBRegressor()
# pristine = data[data.missing_cols == 0]


# def get_single_missing_feature():
#     global data
#     pristine = data[data.missing_cols == 0]
#     single_missing = data[data.missing_cols == 1]
#     single_missing_agg = single_missing.isna().sum()
#     single_missing_cols = single_missing_agg[single_missing_agg > 0]
#     print(single_missing_cols.index)
#     return single_missing_cols


# single_missing_cols = ['F_1_0', 'F_1_1', 'F_1_2', 'F_1_3', 'F_1_4', 'F_1_5', 'F_1_6', 'F_1_7',
#                        'F_1_8', 'F_1_9', 'F_1_10', 'F_1_11', 'F_1_12', 'F_1_13', 'F_1_14',
#                        'F_3_0', 'F_3_1', 'F_3_2', 'F_3_3', 'F_3_4', 'F_3_5', 'F_3_6', 'F_3_7',
#                        'F_3_8', 'F_3_9', 'F_3_10', 'F_3_11', 'F_3_12', 'F_3_13', 'F_3_14',
#                        'F_3_15', 'F_3_16', 'F_3_17', 'F_3_18', 'F_3_19', 'F_3_20', 'F_3_21',
#                        'F_3_22', 'F_3_23', 'F_3_24', 'F_4_0', 'F_4_1', 'F_4_2', 'F_4_3',
#                        'F_4_4', 'F_4_5', 'F_4_6', 'F_4_7', 'F_4_8', 'F_4_9', 'F_4_10',
#                        'F_4_11', 'F_4_12', 'F_4_13', 'F_4_14']

# def learn_single_feature():
#     global data, pristine, single_missing_cols

#     for c in single_missing_cols:

from sklearnex import patch_sklearn, config_context
patch_sklearn()
import numpy  as np
from sklearn.cluster import DBSCAN
import dpctl


X = np.array([[1., 2.], [2., 2.], [2., 3.],
            [8., 7.], [8., 8.], [25., 80.]], dtype=np.uint8)
with dpctl.device_context("gpu"):
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)