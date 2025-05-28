##### 0. Front Matter #####
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
from snsynth.transform import *
from snsynth.pytorch.nn import PATECTGAN
from snsynth.pytorch import PytorchDPSynthesizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import time
from tqdm import tqdm
import openpyxl
from sklearn.exceptions import ConvergenceWarning
import sys
import os
import warnings
warnings.filterwarnings("ignore", category = ConvergenceWarning)
warnings.filterwarnings(action='ignore', category = FutureWarning)


def block_print():
    sys.stdout = open(os.devnull, 'w')

def restore_print():
    sys.stdout = sys.__stdout__


##### 1. Loading and Preparing Original Data #####
print("##### Loading data.....")
target_variable = 'Number of Barriers'
df = pd.read_csv('Intrusion_Detection/data.csv')
row_cnt = df.shape[0]
print(f"Original dataset loaded with {row_cnt} rows.")
##### 2. Defining columns and bounds #####
continuous_cols = list(col for col in df.columns if col != target_variable)
categorical_cols = []  # No categorical columns in this dataset


def synthetic_pipeline(model, X_test, y_test, continuous_cols, num_synthetic_datasets=10):

    synth_mae_b, synth_mape_b, synth_rmse_b = [], [], []
    synth_mae_c, synth_mape_c, synth_rmse_c = [], [], []

    print(f"\n##### Generating {num_synthetic_datasets} Synthetic Datasets and Evaluating #####")
    start_time = time.time()

    for i in tqdm(range(num_synthetic_datasets), desc="Generating synthetic datasets", unit="dataset"):

        tt = TableTransformer([
            StandardScaler(lower=df[col].min(), upper=df[col].max()) for col in continuous_cols
        ])
        
        iteration_seed = 63 + i
        np.random.seed(iteration_seed)

        epsilon_val = 1.0  # Privacy budget (decrease for more privacy)
        epochs_val = 300   # Training epochs (decrease for more privacy)
        delta_val = 1 / (row_cnt ** 1.1)
        synth = PytorchDPSynthesizer(
        epsilon_val,
        PATECTGAN(regularization='dragan', epochs=epochs_val, delta=delta_val, cuda = True),
        None
        )

        df_no_target = df.drop(columns=[target_variable])
        block_print()
        synth.fit(df_no_target, preprocessor_eps = 0.1, transformer = tt, categorical_columns = categorical_cols, continuous_columns = continuous_cols)
        restore_print()

        synth_data_no_target = synth.sample(len(df_no_target)) # This 'len(df_no_target)' ensures the synthetic dataset has the same size as the original dataset
        synth_data = synth_data_no_target.copy()
        synth_data[target_variable] = df[target_variable].values[:len(synth_data_no_target)]
        
        # Preparing synthetic data for modeling
        Xs = synth_data.drop(columns = [target_variable])
        ys = synth_data[target_variable]
        Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size = 0.2, random_state = 63)

        # Setting C
        model.fit(Xs_train, ys_train)
        ys_pred_c = model.predict(Xs_test)
        synth_mae_c.append(mean_absolute_error(ys_test, ys_pred_c))
        synth_mape_c.append(mean_absolute_percentage_error(ys_test, ys_pred_c))
        synth_rmse_c.append(np.sqrt(mean_squared_error(ys_test, ys_pred_c)))

        # Setting B
        model.fit(Xs_train, ys_train)
        y_pred_b = model.predict(X_test)
        synth_mae_b.append(mean_absolute_error(y_test, y_pred_b))
        synth_mape_b.append(mean_absolute_percentage_error(y_test, y_pred_b))
        synth_rmse_b.append(np.sqrt(mean_squared_error(y_test, y_pred_b)))

    end_time = time.time()
    print(f"\nTotal time taken for generation and testing: {end_time - start_time:.2f} seconds.")

    return {
        'Setting B': {
            'MAE': (np.mean(synth_mae_b), np.std(synth_mae_b)),
            'MAPE': (np.mean(synth_mape_b), np.std(synth_mape_b)),
            'RMSE': (np.mean(synth_rmse_b), np.std(synth_rmse_b)),
        },
        'Setting C': {
            'MAE': (np.mean(synth_mae_c), np.std(synth_mae_c)),
            'MAPE': (np.mean(synth_mape_c), np.std(synth_mape_c)),
            'RMSE': (np.mean(synth_rmse_c), np.std(synth_rmse_c)),
        }
    }



##### 3. Evaluating the Model on Non-Synthetic Data (Setting A) #####
X_vals = df.drop(columns = [target_variable]).values
Y_vals = df[target_variable].values
X_train, X_test, y_train, y_test = train_test_split(X_vals, Y_vals, test_size=0.2, random_state=63)


"""
Models to use:
    - Decision Tree Regressor: DecisionTreeRegressor(random_state = 63)
    - ElasticNet: ElasticNet(random_state = 63)
    - GradientBoostingRegressor: GradientBoostingRegressor(random_state = 63)
    - HuberRegressor: HuberRegressor()
    - Lasso: Lasso(random_state = 63)
    - LinearRegression: LinearRegression()
    - MLP: MLPRegressor(random_state = 63)
    - RandomForest: RandomForestRegressor(random_state = 63)
    - Ridge: Ridge(random_state = 63)
    - SVR: svr = SVR()
    - XgBoost: XGBRegressor(random_state = 63)
"""

models = {
    "DecisionTree": DecisionTreeRegressor(random_state=63),
    "ElasticNet": ElasticNet(random_state=63),
    "GradientBoosting": GradientBoostingRegressor(random_state=63),
    "Huber": HuberRegressor(max_iter = 1000),
    "Lasso": Lasso(random_state=63),
    "LinearRegression": LinearRegression(),
    "MLP": MLPRegressor(random_state=63),
    "RandomForest": RandomForestRegressor(random_state=63),
    "Ridge": Ridge(random_state=63),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(random_state=63)
}

metrics_df = pd.DataFrame(columns=["Setting", "Model", "MAE", "MAPE", "RMSE"])

for model_name, model in models.items():
    print(f"\n\n############### Model Running: {model_name} ###############")
    

    # Setting A
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    a_mae = mean_absolute_error(y_test, y_pred)
    a_mape = mean_absolute_percentage_error(y_test, y_pred)
    a_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Setting: Setting A \nMAE: {a_mae:.4f} \nMAPE: {a_mape:.4f} \nRMSE: {a_rmse:.4f}")

    metrics_df = pd.concat([metrics_df, pd.DataFrame([{
        "Setting": "Setting A", 
        "Model": model_name, 
        "MAE": a_mae, 
        "MAPE": a_mape, 
        "RMSE": a_rmse
        }])], ignore_index=True)

    


    
    # Setting B & C using the function
    results = synthetic_pipeline(model, X_test, y_test, continuous_cols, num_synthetic_datasets = 10)

    for setting in ['Setting B', 'Setting C']:
        print(f"\nSetting: {setting}")
        for metric in ['MAE', 'MAPE', 'RMSE']:
            avg, std = results[setting][metric]
            print(f"{metric}: {avg:.4f} (± {std:.4f})")
        
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{
        "Setting": setting, 
        "Model": model_name, 
        "MAE": results[setting]['MAE'][0], 
        "MAPE": results[setting]['MAPE'][0], 
        "RMSE": results[setting]['RMSE'][0]
        }])], ignore_index=True)

excel_name = "Intrusion_Detection/PATECTGAN/ID_PATECTGAN_Results.xlsx"
metrics_df.to_excel(excel_name, index = False)
print(f"\nResults saved to {excel_name}")



