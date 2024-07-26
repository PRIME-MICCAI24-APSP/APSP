import pandas as pd
import numpy as np
import pickle
import os
import json
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings('ignore')

catboost_temporal_df = pd.DataFrame()

class CatBoostTrainer:
    def __init__(self) -> None:
        self.config_path = ""
        self.model = None
        self.X_variables = []
        self.y_variable = ""
        self.save_dir = ""
        self.df = None

    def load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                self.config = json.load(file)

            self.n_splits = self.config.get('n_splits', 5)
            self.specific_test = self.config.get('specific_test', False)
            self.should_save_model = self.config.get('save_model', False)
            self.id = self.config.get('id', "p_num")
            self.valid_cases = self.config.get('valid_cases', [])
            self.metric = self.config.get('metric', "MAE")
            self.early_stopping_rounds = self.config.get('early_stopping_rounds', 50)
            self.iterations = self.config.get('iterations', 1000)
            # JSON에서 리스트를 읽어와서 튜플로 변환
            self.catboost_params = {k: tuple(v) for k, v in self.config['catboost_params'].items()}
            self.internal_to_external_test = self.config.get('internal_to_external_test', False)

        except Exception as e:
            print(f"Error loading config: {e}")
            exit(0)

    def train_model(self, X_train, y_train):
        def bo_params_to_catboost(params):
            params['depth'] = int(params['depth'])
            params['border_count'] = int(params['border_count'])
            return params

        def catboost_hyper_param(**params):
            cb_params = bo_params_to_catboost(params)
            model = CatBoostRegressor(
                iterations=self.iterations,
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metric=self.metric,
                **cb_params,
                verbose=False
            )
            model.fit(X_train, y_train, logging_level='Silent')
            predictions = model.predict(X_train)
            return -mean_absolute_error(y_train, predictions)  # 최대화할 목표(음수 MAE)

        optimizer = BayesianOptimization(
            f=catboost_hyper_param,
            pbounds=self.catboost_params,  # 직접 self.catboost_params 사용
            random_state=1,
        )
        optimizer.maximize(init_points=2, n_iter=3)

        best_params = bo_params_to_catboost(optimizer.max['params'])
        self.model = CatBoostRegressor(**best_params, iterations=1000, eval_metric=self.metric, verbose=False)
        self.model.fit(X_train, y_train)

        return best_params, -optimizer.max['target']
    
    def save_model(self):
        if not self.should_save_model:
            return
        save_dir = os.path.join(self.save_dir, 'model')
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'{self.y_variable}.pkl')
        self.model.save_model(model_path)
        print(f"Model saved at {model_path}")

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        results = pd.DataFrame({
            'target': self.y_variable,
            self.id: X_test.index, # self.id: X_test[self.id]
            'actual': y_test,
            'pred': predictions,
            'MAE': abs(predictions - y_test)
        })
        return results, mae
        
    def run(self):
        self.load_config()
        X = self.df[self.X_variables]
        y = self.df[self.y_variable]

        # 교차 검증을 위한 설정
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        mae_scores = []
        internal_results_details = [] 
        min_mae = float('inf')

        self.temporal_test_df = pd.DataFrame()

        if self.specific_test:
            test_ids = self.valid_cases
            train_df = self.df[~self.df[self.id].isin(test_ids)]
            test_df = self.df[self.df[self.id].isin(test_ids)]
            X_train, y_train = train_df[self.X_variables], train_df[self.y_variable]
            X_test, y_test = test_df[self.X_variables], test_df[self.y_variable]
            p_nums_test = test_df[self.id] 

            # 교차 검증 수행 (valid cases 제외)
            fold_num = 1
            for train_index, test_index in kf.split(X_train):
                X_train_cv, X_test_cv = X_train.iloc[train_index], X_train.iloc[test_index]
                y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
                p_nums_cv = train_df.iloc[test_index][self.id]

                self.train_model(X_train_cv, y_train_cv)
                predictions = self.model.predict(X_test_cv)
                mae = mean_absolute_error(y_test_cv, predictions)
                mae_scores.append(mae)

                fold_results = pd.DataFrame({
                    "fold_num": fold_num, 
                    "p_num": p_nums_cv.values,
                    "target": [self.y_variable] * len(test_index),
                    "actual": y_test_cv.values,
                    "pred": predictions,
                    "MAE": np.abs(y_test_cv.values - predictions)
                })
                internal_results_details.append(fold_results)

                if self.internal_to_external_test:
                    external_predictions = self.model.predict(X_test)
                    external_mae = mean_absolute_error(y_test, external_predictions)
                    if external_mae < min_mae:
                        min_mae = external_mae
                        self.temporal_test_df = pd.DataFrame({
                            'target': self.y_variable,
                            self.id: p_nums_test, # self.id: X_test[self.id]
                            'actual': y_test,
                            'pred': external_predictions,
                            'MAE': abs(external_predictions - y_test)
                        })

                fold_num += 1

            internal_results_detail = pd.concat(internal_results_details, ignore_index=True)

            global catboost_temporal_df
            catboost_temporal_df = catboost_temporal_df.append(self.temporal_test_df)
            if not self.temporal_test_df.empty:
                save_path = '/home/newspring/RP_temp_code/temp_folder/catboost_temporal_test_results.csv'
                catboost_temporal_df.to_csv(save_path, index=False)
                print(f'Temporal test results saved to {save_path}')

            # 교차 검증 MAE 평균 계산
            cv_results_df = pd.DataFrame({'target': [self.y_variable], 'MAE': [np.mean(mae_scores)]})

            # valid cases에 대한 평가
            best_params, best_score = self.train_model(X_train, y_train)
            valid_results, valid_mae = self.evaluate_model(X_test, y_test)
            
            return cv_results_df, valid_results, internal_results_detail

        else:
            # 교차 검증만 수행
            for train_index, test_index in kf.split(X):
                X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
                y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

                self.train_model(X_train_cv, y_train_cv)
                predictions = self.model.predict(X_test_cv)
                mae = mean_absolute_error(y_test_cv, predictions)
                mae_scores.append(mae)

            # 모든 fold의 MAE 결과 반환
            cv_results_df = pd.DataFrame({'target': [self.y_variable], 'MAE': [np.mean(mae_scores)]})
            return cv_results_df

        
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
