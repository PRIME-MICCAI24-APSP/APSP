import os
import sys
import re
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

from src.preprocessor import CPreprocessor
from src.lasso_trainer import LassoTrainer
from src.tabnet_trainer import TabNetTrainer
from src.catboost_trainer import CatBoostTrainer
from src.file_operator import FileOperator
from src.error_visualizer import ErrorVisualizer
from src.displacement_saver import DisplacementSaver


if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.abspath(__file__))
    print('base_dir: ', base_dir)

    config_paths = [
        os.path.join(base_dir, 'config', 'lasso_temporal.json'),
        os.path.join(base_dir, 'config', 'catboost_test.json'),
        os.path.join(base_dir, 'config', 'tabnet_train_all.json'),
    ]

    data_file_path = os.path.join(base_dir, 'data', 'pneumo_dataset.xlsx')
    base_save_dir = os.path.join(base_dir, 'results')

    all_configs_results_file_paths = {}

    for config_path in config_paths:
        config_filename = os.path.basename(config_path).split('.')[0]
        config_ops = FileOperator(config_path)
        config = config_ops.read_json()
        
        config_save_dir = os.path.join(base_save_dir, config_filename)
        save_dir = os.path.join(config_save_dir, 'specific_test' if config['specific_test'] else 'default')
        
        preprocessor = CPreprocessor()
        preprocessor.ConfigPath = config_path
        preprocessor.DataFilePath = data_file_path
        preprocessor.SaveDir = save_dir
        preprocessor.process()

        preprocessed_df = preprocessor.DF
        X_variables = preprocessor.IndependentVariables
        target_variables = preprocessor.TargetVariables
        
        if 'lasso' in config_filename.lower():
            trainer = LassoTrainer()
        elif 'tabnet' in config_filename.lower():
            trainer = TabNetTrainer()
        elif 'catboost' in config_filename.lower():
            trainer = CatBoostTrainer()
        else:
            print('Please include the model name in config file name')
            sys.exit(0)
        trainer.ConfigPath = config_path
        trainer.DF = preprocessed_df
        trainer.XVariables = X_variables
        trainer.SaveDir = save_dir

        all_results = pd.DataFrame()
        all_internal_results = pd.DataFrame()
        all_internal_results_detail = pd.DataFrame()

        for y_variable in tqdm(target_variables):
            trainer.YVariable = y_variable
            if config["specific_test"]:
                if config['train_set_eval']:
                    internal_result_df, result_df, internal_results_detail_df = trainer.run() # train data 내부적으로 성능 검증 (CV or LOOCV)
                    all_internal_results_detail = pd.concat([all_internal_results_detail, internal_results_detail_df], ignore_index=True)
                else: 
                    internal_result_df, result_df = trainer.run()
                all_internal_results = pd.concat([all_internal_results, internal_result_df], ignore_index=True) # train data cross validation results
                all_results = pd.concat([all_results, result_df], ignore_index=True) # valid data eval results
            else:
                result_df = trainer.run()
                all_results = pd.concat([all_results, result_df], ignore_index=True)
        print('### MAE mean: ', all_results['MAE'].mean())

        os.makedirs(save_dir, exist_ok=True)
        if config["specific_test"]:
            if config['train_set_eval']:
                internal_detail_results_file_name = os.path.join(save_dir, 'train_set_evaluation_results.csv')
                all_internal_results_detail.to_csv(internal_detail_results_file_name)
            internal_results_file_name = os.path.join(save_dir, 'internal_results.csv')
            results_file_name = os.path.join(save_dir, 'results.csv')
            all_internal_results.to_csv(internal_results_file_name)
            all_results.to_csv(results_file_name)
            all_configs_results_file_paths[config_filename] = {
                "internal_rst_file_path": internal_results_file_name,
                "results_file_path": results_file_name,
                "is_specific_test": True,
            }
        else:
            results_file_name = os.path.join(save_dir, 'all_results.csv')
            all_results.to_csv(results_file_name)
            all_configs_results_file_paths[config_filename] = {
                "rst_file_path": results_file_name,
                "is_specific_test": False,
            }
        
        visualizer = ErrorVisualizer()
        visualizer.ResultDF = all_results
        visualizer.SaveDir = save_dir
        visualizer.Title = "Cross-validation Performance Across All Data (Target-wise)" if not config['specific_test'] else "Cross-validation Performance Across Specific Data (Target-wise)"
        visualizer.FileName = 'cv_performance_across_all_data(target-wise)' if not config['specific_test'] else 'cv_performance_across_specific_data(target-wise)'
        visualizer.plot_target_wise_mae()
        visualizer.clear()
        
        if config["specific_test"]:
            if config["train_set_eval"]:
                visualizer.ResultDF = all_internal_results_detail
                visualizer.Title = "Cross-validation Prediction Error Across Internal Data (ID-wise)"
                visualizer.FileName = "cv_performance_across_internal_data (id-wise)"
                visualizer.IDWiseFigSize = (50, 6)
                visualizer.plot_id_wise_mae()
                visualizer.clear()

            visualizer.ResultDF = all_results
            visualizer.Title = "Cross-validation Prediction Error Across Specific Data (ID-wise)"
            visualizer.FileName = "cv_performance_across_specific_data(id-wise)"
            visualizer.plot_id_wise_mae()
            visualizer.clear()

            visualizer.ResultDF = all_internal_results
            visualizer.Title = "Cross-validation Prediction Error Across Specific Data (Target-wise)"
            visualizer.FileName = "cv_performance_across_specific_data(target-wise)"
            visualizer.plot_target_wise_mae()
            visualizer.clear()

            dp_saver = DisplacementSaver()
            dp_saver.SaveDir = save_dir
            dp_saver.InputCSVPath = results_file_name
            dp_saver.run()
        else:
            pass
    
    # All Configs' Results Visualization
    custom_sort_key = lambda item: (re.search(r'([a-zA-Z]+)', item).group(), int(re.search(r'(\d+)', item).group()))

    plt.figure(figsize=(30, 6))

    for config_name, config_info in all_configs_results_file_paths.items():
        is_specific_test = config_info["is_specific_test"]

        if is_specific_test:
            internal_df = pd.read_csv(config_info["internal_rst_file_path"])
            results_df = pd.read_csv(config_info["results_file_path"])

            if internal_df.empty and results_df.empty:
                print(f'Skipping {config_name} due to empty dataframes.')
                continue

            if not internal_df.empty:
                internal_df = internal_df.groupby('target')['MAE'].mean().reset_index()
                internal_df['sort_key'] = internal_df['target'].apply(custom_sort_key)
                internal_df = internal_df.sort_values(by='sort_key').drop('sort_key', axis=1)
                plt.plot(internal_df['target'], internal_df['MAE'], label=config_name + " - Internal-Cross-Validation", marker='o')

            if not results_df.empty:
                results_df = results_df.groupby('target')['MAE'].mean().reset_index()
                results_df['sort_key'] = results_df['target'].apply(custom_sort_key)
                results_df = results_df.sort_values(by='sort_key').drop('sort_key', axis=1)
                plt.plot(results_df['target'], results_df['MAE'], label=config_name + " - Temporal-Validation", marker='o')
        else:
            df = pd.read_csv(config_info["rst_file_path"])
            df = df.groupby('target')['MAE'].mean().reset_index()
            df['sort_key'] = df['target'].apply(custom_sort_key)
            df = df.sort_values(by='sort_key').drop('sort_key', axis=1)
            plt.plot(df['target'], df['MAE'], label=config_name, marker='o')

    plt.xlabel('Target')
    plt.ylabel('MAE')
    plt.title('MAE Comparison Across Configurations and Types', size=20)
    plt.legend()
    plt.grid(True, linestyle='--', color='lightgray')
    save_file_name = os.path.join(base_save_dir, 'compare_all_experiments.png')
    plt.savefig(save_file_name)
    print(f'Saved to {save_file_name}')
    plt.close()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=1)

    for config_name, config_info in all_configs_results_file_paths.items():
        is_specific_test = config_info["is_specific_test"]

        if is_specific_test:
            internal_df = pd.read_csv(config_info["internal_rst_file_path"])
            results_df = pd.read_csv(config_info["results_file_path"])

            if internal_df.empty and results_df.empty:
                print(f'Skipping {config_name} due to empty dataframes.')
                continue

            if not internal_df.empty:
                internal_df = internal_df.groupby('target')['MAE'].mean().reset_index()
                internal_df['sort_key'] = internal_df['target'].apply(custom_sort_key)
                internal_df = internal_df.sort_values(by='sort_key').drop('sort_key', axis=1)
                fig.add_trace(go.Scatter(x=internal_df['target'], y=internal_df['MAE'], mode='lines+markers', name=config_name + " - Internal-Cross-Validation"))

            if not results_df.empty:
                results_df = results_df.groupby('target')['MAE'].mean().reset_index()
                results_df['sort_key'] = results_df['target'].apply(custom_sort_key)
                results_df = results_df.sort_values(by='sort_key').drop('sort_key', axis=1)
                fig.add_trace(go.Scatter(x=results_df['target'], y=results_df['MAE'], mode='lines+markers', name=config_name + " - Temporal-Validation"))
        else:
            df = pd.read_csv(config_info["rst_file_path"])
            df = df.groupby('target')['MAE'].mean().reset_index()
            df['sort_key'] = df['target'].apply(custom_sort_key)
            df = df.sort_values(by='sort_key').drop('sort_key', axis=1)
            fig.add_trace(go.Scatter(x=df['target'], y=df['MAE'], mode='lines+markers', name=config_name))

    fig.update_layout(
        title='MAE Comparison Across Configurations and Types',
        xaxis_title='Target',
        yaxis_title='MAE',
        template='plotly_white',
        legend_title_text='Configuration',
        legend=dict(
            orientation="v", 
            x=1,  
            y=1,  
            xanchor='left', 
            yanchor='auto'  
        ),
        margin=dict(l=5, r=5, t=30, b=5)
    )

    save_file_name = os.path.join(base_save_dir, 'compare_all_experiments.html')  # HTML로 저장
    fig.write_html(save_file_name)
    print(f'Saved to {save_file_name}')
