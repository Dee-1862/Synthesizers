
def aggregate_pipeline(model, r_cnt, X_test, y_test, num_synthetic_datasets = 10):

    synth_mae_b, synth_mape_b, synth_rmse_b = [], [], []
    synth_mae_c, synth_mape_c, synth_rmse_c = [], [], []

    print(f"\n##### Generating {num_synthetic_datasets} Synthetic Datasets and Evaluating #####")
    start_time = time.time()

    for i in tqdm(range(num_synthetic_datasets), desc="Generating synthetic datasets", unit="dataset"):


        iteration_seed = 63 + i
        np.random.seed(iteration_seed)

        epsilon_val = 1.0  # Privacy budget (decrease for more privacy)
        delta_val = 1 / (r_cnt ** 1.1)
        synth = AggregateSeededSynthesizer(epsilon=epsilon_val, delta = delta_val, reporting_length = 2)

        df_no_target = df.drop(columns=[target_variable])
        # df_no_target = df_no_target.fillna(df_no_target.mean())
        block_print()
        synth_data_no_target = synth.fit_sample(df_no_target, preprocessor_eps = 0.2)
        restore_print()

        synth_data_no_target.fillna(synth_data_no_target.mean(), inplace=True)
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