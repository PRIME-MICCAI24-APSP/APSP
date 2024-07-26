import os
import json
import random
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

class TabNetTrainer:
    def __init__(self):
        self.config_path = ""
        self.model = None
        self.X_variables = []
        self.y_variable = ""
        self.save_dir = ""
        self.df = None
        self.device = 'cpu'

    def set_seed(self):
        """ Set the seed for reproducibility. """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_config(self):
        with open(self.config_path, 'r') as file:
            self.config = json.load(file)
        self.tabnet_params = self.config.get('tabnet_params', {})
        self.fit_params = self.config.get('fit_params', {})
        self.n_splits = self.config.get('n_splits', 5)
        self.seed = self.config.get('seed', 42)
        self.specific_test = self.config.get('specific_test', False)
        self.train_set_eval = self.config.get('train_set_eval', False)
        self.train_set_eval_cv_type = self.config.get('train_set_eval_cv_type', "5fold")
        self.cv_type = self.config.get('cv_type', '5fold')
        self.valid_cases = self.config.get('valid_cases', [])
        self.id = self.config.get('id', "p_num")
        self.gpu_ids = self.config.get('gpu_ids', [6])
        self.should_save_model = self.config.get('save_model', False)
        self.seed = self.config.get('seed', 42)
        self.set_seed()

        self.optimize_hyperparameters = self.config.get('optimize_hyperparameters', False)
        self.hyperparameter_grid = self.config.get('hyperparameter_grid', {})

        self.tabnet_params['optimizer_fn'] = getattr(optim, self.tabnet_params.get('optimizer_fn', 'Adam'))
        self.tabnet_params['scheduler_fn'] = getattr(lr_scheduler, self.tabnet_params.get('scheduler_fn', 'StepLR'))
        self.fit_params['loss_fn'] = getattr(torch.nn.functional, self.fit_params.get('loss_fn', 'mse_loss'))

    def setup_device(self):
        if torch.cuda.is_available():
            gpu_ids = self.gpu_ids  
            gpu_device = f'cuda:{gpu_ids[0]}'  # 첫 번째 GPU ID 사용
            self.device = torch.device(gpu_device)
        else:
            self.device = torch.device('cpu')
        self.tabnet_params['device_name'] = self.device

    def save_model(self, model):
        save_dir = os.path.join(self.save_dir, 'model')
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{self.y_variable}.pkl')
        torch.save(model, model_path)
        print(f"TabNet model for {self.y_variable} saved at {model_path}")

    def train_model(self, X_train, y_train, X_val=None, y_val=None):
        if self.optimize_hyperparameters:
            # 하이퍼파라미터 최적화를 위한 GridSearchCV 설정
            tabnet_wrapper = TabNetSklearnWrapper()
            param_grid = self.hyperparameter_grid  # self.load_config()에서 로드된 설정
            print('Start GridSearchCV...')
            grid_search = GridSearchCV(estimator=tabnet_wrapper, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
            
            grid_search.fit(X_train, y_train)
            print("Best parameters:", grid_search.best_params_)
            best_model = grid_search.best_estimator_.model
        else:
            # 기존 학습 방식 유지
            self.model = TabNetRegressor(**self.tabnet_params, verbose=0)
            eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
            self.model.fit(X_train=X_train, y_train=y_train, eval_set=eval_set, **self.fit_params)
            best_model = self.model

        return best_model
        
    def run(self):
        print(f'-------------- {self.y_variable} --------------')
        self.load_config()
        self.setup_device()
        X = self.df[self.X_variables].values
        y = self.df[self.y_variable].values.reshape(-1, 1)

        val_df = self.df[self.df[self.id].isin(self.valid_cases)]
        X_val = val_df[self.X_variables].values
        y_val = val_df[self.y_variable].values.reshape(-1, 1)
        p_num_val = val_df[self.id].values

        train_df = self.df[~self.df[self.id].isin(self.valid_cases)]
        X_train = train_df[self.X_variables].values
        y_train = train_df[self.y_variable].values.reshape(-1, 1)

        model = None  
        if self.specific_test:
            if self.cv_type.lower() == 'loo':
                cv_results_detail, model = self.cross_validation(X=X_train, y=y_train, cv_type='loo')
            elif self.cv_type.lower() == 'ncv':
                cv_results_detail, model = self.nested_cross_validation(X=X_train, y=y_train, train_df=train_df)
            elif self.cv_type.lower()[-4:] == 'fold':
                n_splits = int(self.cv_type[0]) if self.cv_type[0].isdigit() else self.n_splits
                cv_results_detail, model = self.cross_validation(X=X_train, y=y_train, cv_type='kfold', n_splits=n_splits)
            elif self.cv_type.lower() == 'train_all':
                cv_results_detail, model = self.train_all_data(X, y)
            else:
                raise ValueError(f"Unsupported cross-validation type: {self.cv_type}. Supported types are 'loo', 'ncv', 'kfold'.")

            if model and self.should_save_model:
                self.save_model(model)  

            # 결과 계산 및 저장
            if self.cv_type.lower() == 'train_all':
                cv_internal_results = cv_results_detail # 빈 데이터프레임
            else:
                avg_mae = cv_results_detail['MAE'].mean() 
                cv_internal_results = pd.DataFrame({'target': [self.y_variable], 'MAE': [avg_mae]})
            valid_results_detail = self.valid_test_cases(model, X_val, y_val, p_num_val) if model else None

            if valid_results_detail is not None:
                valid_avg_mae = valid_results_detail['MAE'].mean()
                valid_results = pd.DataFrame({'target': [self.y_variable], 'MAE': [valid_avg_mae]})

            if self.train_set_eval:
                return cv_internal_results, valid_results_detail, cv_results_detail
            else:
                return cv_internal_results, valid_results_detail

        else:
            return cv_internal_results

    def valid_test_cases(self, model, X_val, y_val, p_num_val):
        test_predictions = model.predict(X_val).ravel()
        test_error = np.abs(test_predictions - y_val.ravel())
        mae = np.mean(test_error)
        print(f"MAE for {self.y_variable}: {mae:.4f}")

        test_results_detail = [{
            'p_num': p_num,  
            'target': self.y_variable,  
            'actual': actual, 
            'pred': pred,  
            'MAE': error  
        } for p_num, actual, pred, error in zip(p_num_val, y_val.ravel(), test_predictions, test_error)]
        
        test_results_details = pd.DataFrame(test_results_detail)

        return test_results_details

    def train_all_data(self, X, y):
        model = self.train_model(X, y, None, None)
        results = pd.DataFrame()
        return results, model

    def cross_validation(self, X, y, cv_type='kfold', n_splits=5):
        if cv_type == 'kfold':
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
            print(f"Starting {cv_type.upper()} cross-validation with {n_splits} splits.")
        elif cv_type == 'loo':
            n_splits = X.shape[0]  
            print(f"Starting LOO (Leave-One-Out) cross-validation with {n_splits} splits.")

        results_detail = []
        fold_num = 0
        best_mae = float('inf')  
        best_model = None

        for train_idx, test_idx in cv.split(X):
            fold_num += 1
            print(f"Training fold {fold_num}/{n_splits}...")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = self.train_model(X_train, y_train, X_test, y_test)
            predictions = model.predict(X_test).ravel()
            error = np.abs(predictions - y_test.ravel())
            test_mae = np.mean(error)
            for i in range(len(test_idx)):
                results_detail.append({
                    'fold_num': fold_num,
                    'p_num': self.df.iloc[test_idx][self.id].values[i],
                    'target': self.y_variable,
                    'actual': y_test.ravel()[i],
                    'pred': predictions[i],
                    'MAE': error[i]
                })
        
            if test_mae < best_mae:
                best_mae = test_mae
                best_model = model
                print(f"New best model with MAE: {best_mae} at fold {fold_num}/{n_splits}")
        
        print(f"Completed {cv_type.upper()} cross-validation. Best MAE: {best_mae}")
        results_details = pd.DataFrame(results_detail)

        return results_details, best_model

    def nested_cross_validation(self, X, y, train_df, outer_splits=10, inner_splits=9): 
        print(f"Starting nested cross-validation with {outer_splits} outer splits and {inner_splits} inner splits.")
        outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=self.seed)
        inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=self.seed)

        outer_internal_results_detail = []
        outer_fold_num = 0

        for train_idx, test_idx in outer_cv.split(X): # test
            outer_fold_num += 1
            print(f"Outer fold {outer_fold_num}/{outer_splits}...")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
        
            inner_fold_num = 0
            best_model = None
            best_score = float('inf')

            for inner_train_idx, inner_val_idx in inner_cv.split(X_train):
                inner_fold_num += 1
                print(f"  Inner fold {inner_fold_num}/{inner_splits} of outer fold {outer_fold_num}/{outer_splits}...")
                X_fold_train, X_fold_val = X_train[inner_train_idx], X_train[inner_val_idx]
                y_fold_train, y_fold_val = y_train[inner_train_idx], y_train[inner_val_idx]

                p_num_fold_val = train_df.iloc[inner_val_idx][self.id].values

                model = self.train_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
                fold_predictions = model.predict(X_fold_val).ravel()
                fold_error = np.abs(fold_predictions - y_fold_val.ravel())

                fold_mae = np.mean(fold_error)
                if fold_mae < best_score:
                    best_score = fold_mae
                    best_model = model
                    print(f"    New best model with MAE: {best_score} in inner fold {inner_fold_num}/{inner_splits}")

            test_predictions = best_model.predict(X_test).ravel()
            test_error = np.abs(test_predictions - y_test.ravel())
            for i in range(len(test_idx)):
                outer_internal_results_detail.append({
                    'outer_fold_num': outer_fold_num,
                    'p_num': train_df.iloc[test_idx][self.id].values[i],
                    'target': self.y_variable,
                    'actual': y_test.ravel()[i],
                    'pred': test_predictions[i],
                    'MAE': test_error[i]
                })
            
        outer_internal_results_details = pd.DataFrame(outer_internal_results_detail)

        return outer_internal_results_details, best_model
        
        
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
