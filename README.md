# APSP
**[PRIME-MICCAI2024] Automated Patient-Specific Pneumoperitoneum Model Reconstruction for Surgical Navigation Systems in Distal Gastrectomy** 
→ [**PDF**](https://link.springer.com/chapter/10.1007/978-3-031-74561-4_7)

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Running the Code](#running-the-code)
- [Results Visualization](#results-visualization)
- [Authors](#authors)
- [License](#license)

## Introduction

This project aims to develop a model for automated patient-specific pneumoperitoneum model reconstruction to enhance surgical navigation systems, specifically for distal gastrectomy. The repository contains preprocessing, model training, and evaluation scripts for Lasso, TabNet, and CatBoost models.

## Installation

To set up the environment and install the required dependencies, follow these steps:

1. Clone the repository:
    
    ```bash
    git clone https://github.com/PRIME-MICCAI24-APSP/APSP.git
    cd APSP
    
    ```
    
2. Install the required packages:
    
    ```bash
    pip install torch torchvision torchaudio --extra-index-url <https://download.pytorch.org/whl/cu117>
    pip install pandas numpy scikit-learn matplotlib tqdm
    
    ```
    

## Usage

1. Set up the directory structure as follows:
    
    ```
    /workspace
    ├── config
    ├── data
    ├── src
    └── main.py
    
    ```
    
2. Place your configuration files in the `config` directory and the data files in the `data` directory.
3. Update the paths in the `main.py` file if necessary.

## Directory Structure

Here is the overview of the directory structure:

```
/workspace
├── config
│   ├── tabnet_train_all.json
│   ├── lasso_temporal.json
│   └── catboost_test.json
├── data
│   └── pneumo_dataset.xlsx
├── src
│   ├── preprocessor.py
│   ├── lasso_trainer.py
│   ├── tabnet_trainer.py
│   ├── catboost_trainer.py
│   ├── file_operator.py
│   ├── error_visualizer.py
│   └── displacement_saver.py
└── main.py

```

## Configuration

The configuration files for the models should be placed in the `config` directory. Each configuration file should be a JSON file specifying the parameters for training and evaluation.

Example of a configuration file (`config/tabnet_train_all.json`):

```json
{
    "n_splits": 10,
    "seed": 42,
    "save_model": false,
    "id": "p_num",
    "specific_test": true,
    "train_set_eval": true,
    "train_set_eval_cv_type": "train_all",
    "cv_type": "train_all",
    "valid_cases": ["S003", "S004", "S005", "S009", "S010",
                    "S012", "S013", "S015", "S016", "S017",
                    "S018", "S019", "S020", "S021", "S022",
                    "S023", "S026", "S027", "S028", "S029",
                    "S031", "S035"],
    "gpu_ids": [6],
    
    "tabnet_params": {
        "n_d": 2,
        "n_a": 2,
        "n_steps": 3,
        "gamma": 1.3,
        "n_independent": 2,
        "n_shared": 2,
        "lambda_sparse": 0.01,
        "optimizer_fn": "Adam",
        "optimizer_params": {
            "lr": 0.02
        },
        "mask_type": "entmax",
        "scheduler_params": {
            "step_size": 50,
            "gamma": 0.9
        },
        "scheduler_fn": "StepLR",
        "epsilon": 1e-15
    },
    "fit_params": {
        "eval_metric": ["mae"],
        "max_epochs": 100,
        "patience": 50,
        "batch_size": 16,
        "virtual_batch_size": 16,
        "num_workers": 0,
        "drop_last": false,
        "loss_fn": "mse_loss"
    }
}

```

## Running the Code

To run the code, execute the `main.py` script:

```bash
python main.py
```

### What `main.py` Does

When you run `main.py`, it performs the following tasks:

1. **Configuration Loading**: Reads the configuration files specified in `config_paths`. These files contain parameters for model training and evaluation.
2. **Data Preprocessing**: Utilizes the `CPreprocessor` class to load and preprocess the data according to the settings in the configuration files.
3. **Model Training**: Depending on the model specified in each configuration file (`LassoTrainer`, `TabNetTrainer`, `CatBoostTrainer`), it trains the model using the preprocessed data.
4. **Evaluation**: Evaluates the trained model using cross-validation or a specific test set, as defined in the configuration.
5. **Results Saving**: Saves the results of the evaluation, including model performance metrics, to the `results` directory.
6. **Visualization**: Uses the `ErrorVisualizer` class to generate visualizations of the model performance.
7. **Displacement Saving**: For specific test cases, uses the `DisplacementSaver` class to save predicted displacements to text files.

The script ensures that each step is executed sequentially, and the results are saved and visualized appropriately.


## Results Visualization

The results of the model training and evaluation will be saved in the `results` directory. You can use the `error_visualizer.py` script to visualize the performance of the models.

## Data Availability

To protect patient privacy, only three sample cases demonstrating the best, worst, and typical performance are publicly available. Theses sample cases can be found in the `data` directory.

## Authors

This project is developed by the research team working on the APSP project.

## License
This project is licensed under the MIT License.
