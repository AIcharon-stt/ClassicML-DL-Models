# ------------------------------------------------------------------------------
# -*- coding: utf-8 -*-            
# @Author : Code_charon
# @Time : 2023/12/9 20:37
# ------------------------------------------------------------------------------
import time
from Function import load_data, preprocess_data, calculate_metrics
from Model import historical_average, train_and_predict
import pandas as pd

params = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 100,  
    'model_type': 'HA'  # model: 'HA', 'ARIMA', 'SVR'
}


def main():
    demand_data, feature_data = load_data()
    train_data, test_data, train_features, test_features = preprocess_data(demand_data, feature_data)

    metrics_sum = {'MAE': 0, 'RMSE': 0, 'MSE': 0, 'MAPE': 0, 'MPE': 0, 'R2': 0}
    num_columns = demand_data.shape[1]
    mpe_values_list = []
    mpe_values = []

    for column in range(num_columns):
        train_labels = train_data.iloc[:, column].values
        test_labels = test_data.iloc[:, column].values

        start_time = time.time()
        test_predictions = train_and_predict(params['model_type'], train_features, train_labels, test_features)
        duration = time.time() - start_time

        metrics, mpe = calculate_metrics(test_labels, test_predictions)
        mpe_values.append(mpe)
        for key in metrics_sum:
            metrics_sum[key] += metrics[key]

        mpe_values_list.append(metrics['MPE_values'])

        print(f"地区 {column + 1}: Duration {duration} seconds")

    avg_metrics = {key: value / num_columns for key, value in metrics_sum.items()}
    print(f"所有地区指标平均: {avg_metrics}")

    # Write mpe values for each iteration to a CSV file with transposition and column index starting from 0
    pd.DataFrame(mpe_values, columns=['MPE']).transpose().reset_index(drop=True).to_csv('Results/mpe_values_average_HA.csv', index=False)

    # Write MPE values to a CSV file with transposition and column index starting from 0
    mpe_values_df = pd.DataFrame(mpe_values_list).transpose().reset_index(drop=True)
    mpe_values_df.columns = range(num_columns)  # Set column index starting from 0
    mpe_values_df.to_csv('Results/mpe_values_HA.csv', index=False)


if __name__ == "__main__":
    main()
