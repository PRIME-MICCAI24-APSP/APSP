import matplotlib.pyplot as plt
import pandas as pd
import re
import os

class ErrorVisualizer:
    def __init__(self):
        self.results_df = None
        self.save_dir = ""
        self.title = ""
        self.file_name = ""
        self.id_wise_figsize = (10, 6)
        self.target_wise_figsize = (20, 6)

    @staticmethod
    def sort_validation_list(target_list):
        def custom_sort_key(item):
            # Extracting numeric and alphabetic parts
            num_part = int(re.search(r'(\d+)', item).group())
            alpha_part = re.search(r'([a-zA-Z]+)', item).group()
            return alpha_part, num_part

        return sorted(target_list, key=custom_sort_key)

    def plot_target_wise_mae(self):
        if self.results_df is None or self.results_df.empty:
            print('No data in results_df(train_all_data). Operation aborted.')
            return
        target_wise_mae = self.results_df.groupby('target')['MAE'].mean().reset_index()
        sorted_targets = self.sort_validation_list(target_wise_mae['target'].tolist())
        target_wise_mae['target'] = pd.Categorical(target_wise_mae['target'], categories=sorted_targets, ordered=True)
        target_wise_mae.sort_values('target', inplace=True)
        title = self.title if self.title != "" else 'Target-wise Mean Absolute Error'
        file_name = f'{self.file_name}.png' if self.file_name != "" else 'target-wise_error.png'

        plt.figure(figsize=self.target_wise_figsize)
        plt.plot(target_wise_mae['target'], target_wise_mae['MAE'], marker='o')
        plt.xlabel('Target')
        plt.ylabel('Mean Absolute Error')
        plt.title(title)
        plt.grid(True, linestyle='--', color='lightgray') 
        plt.xticks(rotation=90)
        save_file_name = os.path.join(self.save_dir, f'{self.file_name}')
        plt.savefig(save_file_name)
        print(f'Saved to {save_file_name}')
        plt.close()

    def plot_id_wise_mae(self):
        if self.results_df is None or self.results_df.empty:
            print('No data in results_df(train_all_data). Operation aborted.')
            return
        id_wise_mae = self.results_df.groupby('p_num')['MAE'].mean().reset_index()
        title = self.title if self.title != "" else 'ID-wise Mean Absolute Error'
        file_name = f'{self.file_name}.png' if self.file_name != "" else 'id-wise_error.png'

        plt.figure(figsize=self.id_wise_figsize)
        id_wise_mae['p_num'] = id_wise_mae['p_num'].astype(str)  
        plt.plot(id_wise_mae['p_num'], id_wise_mae['MAE'], marker='o')
        plt.xlabel('ID (p_num)')
        plt.ylabel('Mean Absolute Error')
        plt.title(title)
        plt.grid(True, linestyle='--', color='lightgray') 
        plt.xticks(rotation=90)
        save_file_name = os.path.join(self.save_dir, f'{self.file_name}')
        plt.savefig(save_file_name)
        print(f'Saved to {save_file_name}')
        plt.close()
    
    def clear(self):
        self.results_df = None
        self.title = ""
        self.file_name = ""
        self.id_wise_figsize = (10, 6)
        self.target_wise_figsize = (20, 6)
    
    @property
    def ResultDF(self):
        return self.results_df
    @ResultDF.setter
    def ResultDF(self, value):
        self.results_df = value
    @property
    def SaveDir(self):
        return self.save_dir
    @SaveDir.setter
    def SaveDir(self, value):
        self.save_dir = value
    @property
    def Title(self):
        return self.title
    @Title.setter
    def Title(self, value):
        self.title = value
    @property
    def FileName(self):
        return self.file_name
    @FileName.setter
    def FileName(self, value):
        self.file_name = value
    @property
    def IDWiseFigSize(self):
        return self.id_wise_figsize
    @IDWiseFigSize.setter
    def IDWiseFigSize(self, value): 
        self.id_wise_figsize = value
    @property
    def TargetWiseFigSize(self):
        return self.target_wise_figsize
    @TargetWiseFigSize.setter
    def TargetWiseFigSize(self, value): 
        self.target_wise_figsize = value
