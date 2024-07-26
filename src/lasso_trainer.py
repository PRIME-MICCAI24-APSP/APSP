import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
import json
import warnings
from sklearn.exceptions import ConvergenceWarning

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.file_operator import FileOperator

warnings.filterwarnings('ignore', category=ConvergenceWarning)

lasso_temporal_df = pd.DataFrame()

class LassoTrainer:
    def __init__(self) -> None:
        self.config_path = ""
        self.model = None
        self.X_variables = []
        self.y_variable = ""
        self.save_dir = ""
        self.df = None
        self.scaler = None
        self.scaler_saved = False

    def load_config(self):
        try:
            config_ops = FileOperator(self.config_path)
            self.config = config_ops.read_json()

            self.should_save_model = self.config.get('save_model', False)
            self.specific_test = self.config.get('specific_test', False)
            self.specific_test_ids = self.config.get('valid_cases', [])
            self.id = self.config.get('id', "p_num")
            self.valid_cases = self.config.get('valid_cases', [])
            self.n_splits = self.config.get('n_splits', 5)
            self.metric = self.config.get('metric', "neg_mean_absolute_error")
            self.lasso_params = self.config.get('lasso_params', {})
            alpha_range = self.lasso_params['alpha']
            self.lasso_params['alpha'] = np.logspace(alpha_range[0], alpha_range[1], alpha_range[2])
            self.train_set_eval = self.config.get('train_set_eval', False)
            self.train_set_eval_cv_type = self.config.get('train_set_eval_cv_type', "5fold")
            self.internal_to_external_test = self.config.get('internal_to_external_test', False)

        except FileNotFoundError:
            print(f"Error: The config file was not found at {self.config_path}.")
            exit(0)
        except json.JSONDecodeError:
            print(f"Error: The config file at {self.config_path} is not a valid JSON.")
            exit(0)
        except KeyError as e:
            print(f"Error: Missing key in the config file: {e}")
            exit(0)
    
    def train_model(self, X_train, y_train):
        lasso = Lasso()
        grid_search = GridSearchCV(estimator=lasso, param_grid=self.lasso_params, cv=self.n_splits, scoring=self.metric)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_
    
    def save_model(self):
        save_dir = os.path.join(self.save_dir, 'model')
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{self.y_variable}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model for {self.y_variable} saved at {model_path}")
        self.save_scaler()
        
    def save_scaler(self):
        if not self.scaler_saved:
            scaler_dir = os.path.join(self.save_dir, 'scaler')
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f'Scaler saved at {scaler_path}')
            self.scaler_saved = True

    def fit_scale_data(self):
        if self.scaler is None:
            scaler_name = self.config.get('scaler', 'StandardScaler')
            scaler_class = getattr(preprocessing, scaler_name, StandardScaler)
            self.scaler = scaler_class()
            scaled_columns = [col for col in self.df.columns if col not in [self.y_variable] and col in self.X_variables]
            self.df[scaled_columns] = self.scaler.fit_transform(self.df[scaled_columns])

    def evaluate_model(self, X_test, y_test, p_nums):
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        return pd.DataFrame({
            'p_num': p_nums,
            'target': [self.y_variable] * len(y_test),
            'actual': y_test,
            'pred': predictions,
            'MAE': abs(predictions - y_test)
        })

    def perform_kfold_cv(self, X, y, p_nums, X_external_test=None, y_external_test=None, external_p_nums=None):
        if self.train_set_eval_cv_type.lower() == "loo":
            cv = LeaveOneOut()
        elif self.train_set_eval_cv_type[-4:].lower() == "fold":
            cv = KFold(n_splits=int(self.train_set_eval_cv_type[0]), shuffle=True, random_state=42)
        
        results = []
        fold_num = 0  
        min_mae = float('inf') # for external test
        self.temporal_test_df = pd.DataFrame()

        for train_index, test_index in cv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            p_nums_test = p_nums.iloc[test_index]

            lasso = Lasso(**self.best_params)  # Use best_params from GridSearchCV
            lasso.fit(X_train, y_train)
            predictions = lasso.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)

            for id_val, actual, pred in zip(p_nums_test, y_test, predictions):
                results.append({
                    'fold_num': fold_num,
                    f'{self.id}': id_val,
                    'target': self.y_variable,
                    'actual': actual,
                    'pred': pred,
                    'MAE': abs(actual - pred)
                })

            if self.internal_to_external_test:
                external_predictions = lasso.predict(X_external_test)
                external_mae = mean_absolute_error(y_external_test, external_predictions)
                if external_mae < min_mae:
                    min_mae = external_mae  
                    self.temporal_test_df = pd.DataFrame({
                        'p_num': external_p_nums,
                        'target': [self.y_variable] * len(external_p_nums),
                        'actual': y_external_test,
                        'pred': external_predictions,
                        'MAE': abs(external_predictions - y_external_test)
                    })

            fold_num += 1  # Fold 번호 증가
        
        global lasso_temporal_df
        lasso_temporal_df = lasso_temporal_df.append(self.temporal_test_df)
        if not self.temporal_test_df.empty:
            save_path = "/home/newspring/RP_temp_code/temp_folder"
            file_name = "temporal_test_results.csv"
            full_path = os.path.join(save_path, file_name)
            lasso_temporal_df.to_csv(full_path, index=False)
            print(f"Saved temporal test results to {full_path}")

        return pd.DataFrame(results)
    
    def run(self):
        self.load_config()
        self.fit_scale_data()
        X = self.df[self.X_variables]
        y = self.df[self.y_variable]

        if self.specific_test:
            test_ids = self.specific_test_ids
            test_df = self.df[self.df[self.id].isin(test_ids)]
            X_test = test_df[self.XVariables]
            y_test = test_df[self.y_variable]
            p_nums = self.df[self.id]
            p_nums_test = test_df[self.id] 

            train_df = self.df[~self.df[self.id].isin(test_ids)]
            X_train = train_df[self.XVariables]
            y_train = train_df[self.y_variable]
        else:
            X_train, y_train = X, y
            X_test, y_test, p_nums_test = None, None, None

        best_params, best_score = self.train_model(X_train, y_train)
        self.best_params = best_params
        print(f"{self.y_variable} Best params: {best_params}")

        if self.should_save_model:
            self.save_model()
        
        if self.specific_test:
            internal_results = pd.DataFrame({'target': [self.y_variable], 'MAE': [-best_score]})
            results = self.evaluate_model(X_test, y_test, p_nums_test)
            if self.train_set_eval:
                internal_results_detail = self.perform_kfold_cv(X_train, y_train, p_nums, X_test, y_test, p_nums_test)
                return internal_results, results, internal_results_detail
            else:
                return internal_results, results
        else:
            return pd.DataFrame({'target': [self.y_variable], 'MAE': [-best_score]})
    
    def clear(self):
        self.model = None
        self.y_variable = ""


    @property
    def ConfigPath(self):
        return self.config_path
    @ConfigPath.setter
    def ConfigPath(self, value):
        self.config_path = value
    @property
    def DF(self):
        return self.df
    @DF.setter
    def DF(self, value):
        self.df = value
    @property
    def XVariables(self):
        return self.X_variables
    @XVariables.setter
    def XVariables(self, value):
        self.X_variables = value
    @property
    def YVariable(self):
        return self.y_variable
    @YVariable.setter
    def YVariable(self, value):
        self.y_variable = value
    @property
    def SaveDir(self):
        return self.save_dir
    @SaveDir.setter
    def SaveDir(self, value):
        self.save_dir = value
