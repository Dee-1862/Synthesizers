##### 0. Front Matter #####
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import time
from utils import *
from tqdm import tqdm
import openpyxl
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


##### 1. Loading and Preparing Original Data #####
print("##### Loading data.....")
target_variable = 'Rings'
abalnone = fetch_ucirepo(id=1)
X = abalnone.data.features
Y = abalnone.data.targets
df = pd.concat([X, Y], axis=1)
row_cnt = df.shape[0]
print(f"Original dataset loaded with {row_cnt} rows.")

##### 2. Defining columns and bounds #####
categorical_cols = ['Sex']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
continuous_cols = list(col for col in df.columns if col != target_variable)
x_bounds = {col: (df[col].min(), df[col].max()) for col in continuous_cols}


y_lb, y_ub = df[target_variable].min(), df[target_variable].max()


def synthetic_pipeline(model, X_vals, Y_vals, X_test, y_test, continuous_cols, x_bounds, y_bounds, num_synthetic_datasets=10):

    y_lb, y_ub = y_bounds

    synth_mae_b, synth_mape_b, synth_rmse_b = [], [], []
    synth_mae_c, synth_mape_c, synth_rmse_c = [], [], []

    print(f"\n##### Generating {num_synthetic_datasets} Synthetic Datasets and Evaluating #####")
    start_time = time.time()

    for i in tqdm(range(num_synthetic_datasets), desc="Generating synthetic datasets", unit="dataset"):
        iteration_seed = 63 + i
        np.random.seed(iteration_seed)

        epsilon_val = 1.0
        delta_val = 1 / (len(X_vals) ** 1.1)
        theta = 200
        mu = mu_from_eps_delta(epsilon_val, delta_val)

        priv_budget = 9 / np.sqrt(len(continuous_cols)) * mu

        x_bins_list, sum_x, sum_y, counts, _ = privtree_binning_XY_with_theta(
            X_vals, Y_vals, list(x_bounds.values()), mu, theta
        )
        X_syn, Y_syn, *_ = DP_fy_sum_x_y_counts_syn_data(
            sum_x, sum_y, counts, x_bins_list, y_ub, priv_budget
        )

        for idx, col in enumerate(continuous_cols):
            X_syn[:, idx] = np.clip(X_syn[:, idx], x_bounds[col][0], x_bounds[col][1])
        Y_syn = np.clip(Y_syn, y_lb, y_ub)

        Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_syn, Y_syn, test_size=0.2, random_state=63)

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
    
    print(f"Setting A - \nMAE: {a_mae:.4f}, \nMAPE: {a_mape:.4f}, \nRMSE: {a_rmse:.4f}")

    metrics_df = pd.concat([metrics_df, pd.DataFrame([{
        "Setting": "Setting A", 
        "Model": model_name, 
        "MAE": a_mae, 
        "MAPE": a_mape, 
        "RMSE": a_rmse
        }])], ignore_index=True)

    


    
    # Setting B & C using the function
    results = synthetic_pipeline(model, X_vals, Y_vals, X_test, y_test, continuous_cols, x_bounds, (y_lb, y_ub), num_synthetic_datasets = 10)

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

excel_name = "Abalone/VER 11/AD_VER11_Results.xlsx"
metrics_df.to_excel(excel_name, index = False)
print(f"\nResults saved to {excel_name}")



