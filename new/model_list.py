import numpy as np
import pandas as pd
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    Lars,
    LassoLars,
    OrthogonalMatchingPursuit,
    BayesianRidge,
    ARDRegression,
    SGDRegressor,
    PassiveAggressiveRegressor,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor,
)
from sklearn.svm import LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_ITER = 1000
ALPHA = 0.1
LEARNING_RATE = 0.01

SINGLE_TARGET_MODELS = [
    ("Linear Regression", LinearRegression()),
    ("Ridge", Ridge(alpha=ALPHA, random_state=RANDOM_STATE)),
    ("Lasso", Lasso(alpha=ALPHA, random_state=RANDOM_STATE)),
    ("ElasticNet", ElasticNet(alpha=ALPHA, l1_ratio=0.5, random_state=RANDOM_STATE)),
    ("Lars", Lars(random_state=RANDOM_STATE)),
    ("LassoLars", LassoLars(alpha=ALPHA, random_state=RANDOM_STATE)),
    ("OrthogonalMatchingPursuit", OrthogonalMatchingPursuit()),
    ("BayesianRidge", BayesianRidge(max_iter=MAX_ITER)),
    ("ARDRegression", ARDRegression(max_iter=MAX_ITER)),
    ("SGDRegressor", SGDRegressor(max_iter=MAX_ITER, random_state=RANDOM_STATE)),
    (
        "PassiveAggressiveRegressor",
        PassiveAggressiveRegressor(max_iter=MAX_ITER, random_state=RANDOM_STATE),
    ),
    (
        "HuberRegressor",
        HuberRegressor(max_iter=MAX_ITER),
    ),
    ("RANSACRegressor", RANSACRegressor(random_state=RANDOM_STATE)),
    ("TheilSenRegressor", TheilSenRegressor(random_state=RANDOM_STATE)),
    ("LinearSVR", LinearSVR(random_state=RANDOM_STATE)),
    # ("KernelRidge", KernelRidge(alpha=ALPHA)), # 180GiB of RAM required!?!?!
    ("KNeighborsRegressor", KNeighborsRegressor()),
    ("DecisionTreeRegressor", DecisionTreeRegressor(random_state=RANDOM_STATE)),
    (
        "RandomForestRegressor",
        RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
    ),
    (
        "ExtraTreesRegressor",
        ExtraTreesRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
    ),
    (
        "GradientBoostingRegressor",
        GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "AdaBoostRegressor",
        AdaBoostRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE,
        ),
    ),
    ("MLPRegressor", MLPRegressor(max_iter=MAX_ITER, random_state=RANDOM_STATE)),
    (
        "GaussianProcessRegressor",
        GaussianProcessRegressor(
            kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), random_state=RANDOM_STATE
        ),
    ),
    ("PLSRegression", PLSRegression(n_components=2, max_iter=MAX_ITER)),
    (
        "XGBoost",
        XGBRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "LightGBM",
        LGBMRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE,
        ),
    ),
    (
        "CatBoost",
        CatBoostRegressor(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            random_state=RANDOM_STATE,
            verbose=False,
        ),
    ),
]


MULTI_TARGET_MODELS = [
    ("Multi-target Linear Regression", LinearRegression()),
    ("Multi-target Ridge", Ridge(alpha=ALPHA, random_state=RANDOM_STATE)),
    (
        "Multi-target Lasso",
        MultiOutputRegressor(Lasso(alpha=ALPHA, random_state=RANDOM_STATE)),
    ),
    (
        "Multi-target ElasticNet",
        MultiOutputRegressor(
            ElasticNet(alpha=ALPHA, l1_ratio=0.5, random_state=RANDOM_STATE)
        ),
    ),
    (
        "Multi-target RandomForest",
        RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
    ),
    (
        "Multi-target ExtraTrees",
        ExtraTreesRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE),
    ),
    (
        "Multi-target GradientBoosting",
        MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                random_state=RANDOM_STATE,
            )
        ),
    ),
    (
        "Multi-target MLPRegressor",
        MLPRegressor(max_iter=MAX_ITER, random_state=RANDOM_STATE),
    ),
    (
        "Multi-target XGBoost",
        MultiOutputRegressor(
            XGBRegressor(
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                random_state=RANDOM_STATE,
            )
        ),
    ),
    (
        "Multi-target LightGBM",
        MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                random_state=RANDOM_STATE,
            )
        ),
    ),
    (
        "Multi-target CatBoost",
        MultiOutputRegressor(
            CatBoostRegressor(
                n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE,
                random_state=RANDOM_STATE,
                verbose=False,
            )
        ),
    ),
]
