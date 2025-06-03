##### Front Matter #####
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import openpyxl
from sklearn.exceptions import ConvergenceWarning
import warnings
import sys
import os

script_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.append(root_dir)

from DPCTGAN_utils import *

warnings.filterwarnings("ignore", category = ConvergenceWarning)
warnings.filterwarnings(action='ignore', category = FutureWarning)


if __name__ == "__main__":
    print("##### Loading data.....")
    target_variable = 'PE'
    air_quality = fetch_ucirepo(id=294)
    X = air_quality.data.features
    Y = air_quality.data.targets
    df = pd.concat([X, Y], axis=1)
    row_cnt = df.shape[0]
    base_output_dir = "Combined_Cycle_Power_Plant/DPCTGAN"
    file_name = "CCPP_DPCTGAN_Results.xlsx"
    print(f"Original dataset loaded with {row_cnt} rows.")

    ##### 2. Defining columns and bounds #####
    continuous_cols = list(col for col in df.columns if col != target_variable)
    categorical_cols = []

    # Define the models to be used
    models_to_use = {
        "DecisionTree": DecisionTreeRegressor(random_state=63),
        "ElasticNet": ElasticNet(random_state=63),
        # "GradientBoosting": GradientBoostingRegressor(random_state=63),
        # "Huber": HuberRegressor(max_iter = 1000),
        # "Lasso": Lasso(random_state=63),
        # "LinearRegression": LinearRegression(),
        # "MLP": MLPRegressor(random_state=63),
        # "RandomForest": RandomForestRegressor(random_state=63),
        # "Ridge": Ridge(random_state=63),
        # "SVR": SVR(),
        # "XGBoost": XGBRegressor(random_state=63)
    }

    # Running the evaluation pipeline
    final_metrics_df, final_synthetic_data = dpctgan_pipeline(
        df = df,
        target_variable = target_variable,
        models = models_to_use,
        base_output_dir = base_output_dir,
        excel_file_name = file_name,
        row_cnt = row_cnt,
        categorical_cols = categorical_cols,
        continuous_cols = continuous_cols,
        num_synthetic_datasets = 2
    )

    print("\n##### Final Metrics DataFrame")
    print(final_metrics_df)




