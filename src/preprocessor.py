import pandas as pd
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.file_operator import FileOperator

class CPreprocessor:
    def __init__(self):
        self.config_path = ""
        self.data_file_path = ""
        self.save_dir = ""
        self.set_Xy()
    
    def set_Xy(self):
        self.independent_variables = ['Gender', 'AGE', 'Weight', 'Height', 'BMI', 
                                      '5SFA', '4SFA', '3SFA', '2SFA', '1SFA', 
                                      '5MA', '4MA', '3MA', '2MA', '1MA', 
                                      '5Peri', '4Peri', '3Peri', '2Peri', '1Peri', 
                                      '5Ratio', '4Ratio', '3Ratio', '2Ratio', '1Ratio']
        self.target_variables = [f"{i}{axis}" for axis in ['x', 'y', 'z'] for i in range(1,26)]
    
    def load_data(self):
        try:
            self.df = pd.read_excel(self.data_file_path)
        except FileNotFoundError:
            print(f"Error: The excel file was not found at {self.data_file_path}.")
            exit(0)
    
    def load_config(self):
        try:
            config_ops = FileOperator(self.config_path)
            config = config_ops.read_json()
            
            self.id = config.get('id', 'p_num')
            self.remove_outliers = config.get('remove_outliers', False)  
            self.failed_umbilicus = config.get('failed_umbilicus', [])
            self.specific_test_ids = config.get('specific_test_ids', [])
            self.save_model = config.get('save_model', False)
        except FileNotFoundError:
            print(f"Error: The config file was not found at {self.config_path}.")
            exit(0)
        except json.JSONDecodeError:
            print(f"Error: The config file at {self.config_path} is not a valid JSON.")
            exit(0)
        except KeyError as e:
            print(f"Error: Missing key in the config file: {e}")
            exit(0)

    def remove_outlier(self):
        if self.remove_outliers:
            outlier_ids = self.failed_umbilicus
            self.df = self.df[~self.df[self.id].isin(outlier_ids)]
    
    def encode_data(self):
        if 'Gender' in self.df.columns:
            self.df['Gender'] = self.df['Gender'].map({'M': 0, 'F': 1})
        
    def process(self):
        self.load_data()
        self.load_config()
        self.remove_outlier()
        self.encode_data()


    @property
    def ConfigPath(self):
        return self.config_path
    @ConfigPath.setter
    def ConfigPath(self, value):
        self.config_path = value
    @property
    def DataFilePath(self):
        return self.data_file_path
    @DataFilePath.setter
    def DataFilePath(self, value):
        self.data_file_path = value
    @property
    def SaveDir(self):
        return self.save_dir
    @SaveDir.setter
    def SaveDir(self, value):
        self.save_dir = value
    @property
    def DF(self):
        return self.df
    @DF.setter
    def DF(self, value):
        self.df = value
    @property
    def IndependentVariables(self):
        return self.independent_variables
    @IndependentVariables.setter
    def IndependentVariables(self, value):
        self.independent_variables = value
    @property
    def TargetVariables(self):
        return self.target_variables
    @TargetVariables.setter
    def TargetVariables(self, value):
        self.target_variables = value
