import os
import pandas as pd
import numpy as np
import pickle
import json

class FileOperator:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def _ensure_directory_exists(self):
        directory = os.path.dirname(self.file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    def read_csv(self):
        try:
            data = pd.read_csv(self.file_path)
            return data
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {e}")
            return None
    
    def write_csv(self, data):
        self._ensure_directory_exists()
        try:
            data.to_csv(self.file_path, index=False)
            print(f"Data successfully written to {self.file_path}")
        except Exception as e:
            print(f"An error occurred while writing the CSV file: {e}")

    def read_excel(self):
        try:
            data = pd.read_excel(self.file_path)
            return data
        except Exception as e:
            print(f"An error occurred while reading the Excel file: {e}")
            return None
    
    def write_excel(self, data):
        self._ensure_directory_exists()
        try:
            data.to_excel(self.file_path, index=False)
            print(f"Data successfully written to {self.file_path}")
        except Exception as e:
            print(f"An error occurred while writing the Excel file: {e}")

    def read_txt(self):
        try:
            with open(self.file_path, "r") as file:
                data = file.readlines()
            return data
        except Exception as e:
            print(f"An error occurred while reading the TXT file: {e}")
            return None

    def write_txt(self, data):
        self._ensure_directory_exists()
        try:
            with open(self.file_path, "w") as file:
                file.writelines(data)
            print(f"Data successfully written to {self.file_path}")
        except Exception as e:
            print(f"An error occurred while writing the TXT file: {e}")

    def read_json(self):
        try:
            with open(self.file_path, "r") as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"An error occurred while reading the JSON file: {e}")
            return None
    
    def write_json(self, data):
        self._ensure_directory_exists()
        try:
            with open(self.file_path, "w") as file:
                json.dump(data, file, indent=4, default=self._json_serializable)
            print(f"Data successfully written to {self.file_path}")
        except Exception as e:
            print(f"An error occurred while writing the JSON file: {e}")

    def _json_serializable(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
                            np.float64, np.complex_, np.complex64, 
                            np.complex128)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # Handle numpy arrays
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif pd.isna(obj):  # Handle NaN values
            return None
        else:
            raise TypeError(f"Type not serializable: {type(obj)}")

    def read_pickle(self):
        try:
            with open(self.file_path, "rb") as file:
                data = pickle.load(file)
            return data
        except Exception as e:
            print(f"An error occurred while reading the Pickle file: {e}")
            return None
    
    def write_pickle(self, data):
        self._ensure_directory_exists()
        try:
            with open(self.file_path, "wb") as file:
                pickle.dump(data, file)
            print(f"Data successfully written to {self.file_path}")
        except Exception as e:
            print(f"An error occurred while writing the Pickle file: {e}")

# Example usage
if __name__ == "__main__":
    ops = FileOperator("some/nonexistent/path/example.csv")
    # Attempt to write some data to a CSV file in a nonexistent directory
    ops.write_csv(pd.DataFrame({"A": [1,2,3], "B": [4,5,6]}))
