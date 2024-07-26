import pandas as pd
import os
import re

class DisplacementSaver:
    def __init__(self):
        self.input_csv_path = ""
        self.save_dir = ""
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.input_csv_path)
        
    def sort_predictions(self, group):
        group['num_part'] = group['target'].apply(lambda x: int(re.search(r'(\d+)', x).group()))
        group['alpha_part'] = group['target'].apply(lambda x: re.search(r'([a-zA-Z]+)', x).group())
        group_sorted = group.sort_values(by=['num_part', 'alpha_part'])
        return group_sorted

    def save_displacements(self):
        if self.data is not None:
            grouped = self.data.groupby('p_num')
            for p_num, group in grouped:
                group_sorted = self.sort_predictions(group)
                predictions = group_sorted['pred'].values
                formatted_predictions = "\n".join([" ".join(map(str, predictions[i:i+3])) for i in range(0, len(predictions), 3)])
                filename = os.path.join(self.save_dir, 'displacements', f'{p_num}_displacements.txt')
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w') as file:
                    file.write(formatted_predictions)
            print(f'All specific cases\' displacements saved to {os.path.dirname(filename)}')

    def run(self):
        self.load_data()
        self.save_displacements()
        
    @property
    def SaveDir(self):
        return self.save_dir
    @SaveDir.setter
    def SaveDir(self, dir):
        self.save_dir = dir
    @property
    def InputCSVPath(self):
        return self.input_csv_path
    @InputCSVPath.setter
    def InputCSVPath(self, path):
        self.input_csv_path = path
