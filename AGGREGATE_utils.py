from utils import *
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from snsynth.aggregate_seeded import AggregateSeededSynthesizer
from snsynth.transform import *
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
import sys
import os

def block_print():
    sys.stdout = open(os.devnull, 'w')

def restore_print():
    sys.stdout = sys.__stdout__

def synthetic_pipeline(df, target_variable, row_cnt, model, X_test, y_test, epsi, pre_epsi, num_synthetic_datasets = 10):

    synth_mae_b, synth_mape_b, synth_rmse_b = [], [], []
    synth_mae_c, synth_mape_c, synth_rmse_c = [], [], []
    all_synth_data_dfs = []

    print(f"\n##### Generating {num_synthetic_datasets} Synthetic Datasets and Evaluating #####")
    start_time = time.time()

    imputer = SimpleImputer(strategy='mean')

    for i in tqdm(range(num_synthetic_datasets), desc="Generating synthetic datasets", unit="dataset"):
        iteration_seed = 63 + i
        np.random.seed(iteration_seed)

        epsilon_val = epsi  # Privacy budget (decrease for more privacy)
        delta_val = 1 / (row_cnt ** 1.1)
        synth = AggregateSeededSynthesizer(epsilon=epsilon_val, delta = delta_val, reporting_length = 2)

        df_no_target = df.drop(columns=[target_variable])
        block_print()
        synth_data_no_target = synth.fit_sample(df_no_target, preprocessor_eps = pre_epsi)
        restore_print()
        
        
        synth_data_imputed_array = imputer.fit_transform(synth_data_no_target)
        synth_data = pd.DataFrame(synth_data_imputed_array, columns=synth_data_no_target.columns)

        synth_data[target_variable] = df[target_variable].values[:len(synth_data)]
        all_synth_data_dfs.append(synth_data.copy())
        
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
        },
        'all_synth_data_dfs': all_synth_data_dfs
    }


def aggregate_pipeline(df, target_variable, models, base_output_dir, excel_file_name, row_cnt, epsi, pre_epsi, continuous_cols, num_synthetic_datasets = 2):
    """
    Runs the complete evaluation pipeline for synthetic data generation and model performance.

    Inputs:
        df: The original dataset.
        target_variable: The name of the target variable column.
        models: A dictionary of scikit-learn compatible regression models.
        base_output_dir: The base directory for saving all results.
        excel_file_name: The filename for the main metrics Excel sheet.
        num_synthetic_datasets: The number of synthetic datasets to generate per model.
    
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

    os.makedirs(base_output_dir, exist_ok=True)
    ##### Evaluating the Model on Non-Synthetic Data (Setting A) #####
    X_vals = df.drop(columns = [target_variable]).values
    Y_vals = df[target_variable].values
    X_train, X_test, y_train, y_test = train_test_split(X_vals, Y_vals, test_size = 0.2, random_state = 63)

    ##### Defining columns and bounds #####
    all_numeric_cols = continuous_cols + [target_variable]

    metrics_df = pd.DataFrame(columns=["Setting", "Model", "MAE", "MAPE", "RMSE"])
    all_synthetic_data_per_model = {} 

    print("\n##### Generating Correlation Heatmaps for Original Data")
    # Plot Correlation heatmap for the original dataset
    plot_correlation_heatmap(df, "Original Dataset Pearson Correlation", basepath=base_output_dir, correlation_method='pearson')
    plot_correlation_heatmap(df, "Original Dataset Spearman Correlation", basepath=base_output_dir, correlation_method='spearman')

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
        results = synthetic_pipeline(df, target_variable, row_cnt, model, X_test, y_test, epsi, pre_epsi, num_synthetic_datasets = num_synthetic_datasets)
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
        all_synthetic_data_per_model[model_name] = results['all_synth_data_dfs']

        # Prints head of the first synthetic dataset for the current model
        if model_name in all_synthetic_data_per_model and len(all_synthetic_data_per_model[model_name]) > 0:
            print(f"\nSample of first synthetic dataset for {model_name}:")
            print(all_synthetic_data_per_model[model_name][0].head())
        else:
            print(f"No synthetic data found for {model_name}.")

    # Construct the full Excel path
    excel_name = os.path.join(base_output_dir, excel_file_name)
    #metrics_df.to_excel(excel_name, index = False)
    print(f"\nResults saved to {excel_name}")

    # Plot heatmaps for all synthetic datasets generated by each model
    for model_name, synth_data_list in all_synthetic_data_per_model.items():
        if synth_data_list:
            for i, synth_data in enumerate(synth_data_list):
                if synth_data is not None:
                    # Plot correlation heatmap for synthetic data
                    plot_correlation_heatmap(synth_data, f"{model_name} Synthetic Data Pearson Correlation Iteration {i+1}", basepath=base_output_dir, correlation_method='pearson')
                    plot_correlation_heatmap(synth_data, f"{model_name} Synthetic Data Spearman Correlation Iteration {i+1}", basepath=base_output_dir, correlation_method='spearman')
                    
                    # Perform WS test for each continuous column
                    compare_distributions_ws(df, synth_data, all_numeric_cols, base_path=base_output_dir, title= f"{model_name} Synthetic Data Iteration {i+1}")
                    
                    # # Save synthetic data
                    # synthetic_data_output_dir = os.path.join(base_output_dir, "synthetic_datasets")
                    # os.makedirs(synthetic_data_output_dir, exist_ok=True)
                    # synth_filename = os.path.join(synthetic_data_output_dir, f"{model_name}_synthetic_data_iter_{i+1}.csv")
                    # synth_data.to_csv(synth_filename, index=False)
                    # print(f"Saved synthetic data: {synth_filename}")

                else:
                    print(f"No valid synthetic data found for {model_name}.")
        else:
            print(f"No synthetic data available for model: {model_name}.")

    print("\n##### Evaluation Pipeline Completed")
    return metrics_df, all_synthetic_data_per_model
