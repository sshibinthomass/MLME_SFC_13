# MLME Project - Final

This repository contains all code, models, and data processing scripts for the MLME Project, focusing on time series prediction, uncertainty quantification, and model evaluation for SFC (Spray Fluidized Coating) data.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Main Scripts](#main-scripts)
- [Folder Descriptions](#folder-descriptions)
- [Data Files](#data-files)
- [Outputs](#outputs)
- [How to Run](#how-to-run)
- [Authors](#authors)

---

## Project Structure

```
.
├── main_final.py
├── preprocessing_final.py
├── Additional_task_2.py
├── Model_Train/
│   ├── training_model.py
│   ├── predict_narx.py
│   ├── Model/
│   ├── Data/
│   └── ...
├── Beat-the-Felix-Prediction/
│   └── OUTPUTS/
├── NARX_Prediction/
│   └── prediction_results_YYYYMMDD_HHMMSS/
├── Additional_Task_Prediction/
└── Readme.md
```

---

## Main Scripts

### `main_final.py`
- **Purpose:** Main entry point for running predictions and evaluating model performance on SFC data files. Computes MSE/MAE metrics, generates plots, and saves predictions and metrics to output folders.
- **Key Features:** 
  - Loads trained models and metadata.
  - Processes and cleans input data.
  - Detects clusters and applies the appropriate NARX model.
  - Supports both open-loop and closed-loop predictions.
  - Saves results and visualizations.

### `preprocessing_final.py`
- **Purpose:** Contains all helper functions for data cleaning, preprocessing, plotting, and metric calculation. Used by main scripts for consistent data handling.
- **Key Features:** 
  - Data cleaning (IQR, outlier removal).
  - Open-loop and closed-loop prediction logic.
  - Uncertainty quantification using CQR (Conformalized Quantile Regression).
  - Plotting utilities.

### `Additional_task_2.py`
- **Purpose:** Implements advanced uncertainty quantification methods (Gaussian, Monte Carlo, Kalman Filter) and compares their performance.
- **Key Features:** 
  - Runs all three uncertainty propagation methods.
  - Generates comparison plots and summary metrics.

---

## Model Training

### `Model_Train/training_model.py`
- **Purpose:** Trains the NARX and Quantile Regression models on the provided SFC data.
- **Key Features:**
  - Data preprocessing and cleaning.
  - Model architecture definition (deep NARX, QR).
  - Training with early stopping and custom loss functions.
  - Saves trained models and scalers for later use.
  - Visualizes data distributions before and after cleaning.

### `Model_Train/predict_narx.py`
- **Purpose:** Provides a class-based interface for loading trained NARX models, making predictions, evaluating performance, and generating plots.
- **Key Features:**
  - Loads models and scalers.
  - Predicts on new or calibration data.
  - Exports results and metrics to CSV and PNG.
  - Supports cluster-based model selection.

---

## Folder Descriptions

### `Model_Train/Model/`
- **Contents:** Trained model files (`.keras`), scalers (`.pkl`), metadata (`.json`), and analysis outputs (plots, CSVs).
- **Subfolders:**
  - `narx/`: Cluster-specific NARX models and scalers.
  - `qr/`: Quantile Regression models and loss curves.
  - `data_visualization/`: Plots comparing data before/after cleaning.
  - `clustering_analysis/`: Clustering results and reports.

### `Model_Train/Data/`
- **Contents:** Raw and processed data files for training, calibration, and testing.
- **Subfolders:**
  - `RAW DATA/`: Original data files, further split into `train/`, `calib/`, and `trash/`.
  - `train/`, `calib/`: Training and calibration datasets.

### `Beat-the-Felix-Prediction/OUTPUTS/`
- **Contents:** Prediction results, metrics, and plots for the Beat-the-Felix challenge data.
- **Subfolders:**
  - `file_12738/`: Detailed prediction and comparison plots for a specific file.

### `NARX_Prediction/`
- **Contents:** Output folders for each NARX prediction run, named by timestamp.

### `Additional_Task_Prediction/`
- **Contents:** Outputs and plots for additional uncertainty quantification tasks.

---

## How to Run

1. **Train Models:**
   - Run `Model_Train/training_model.py` to train NARX and QR models on your data.
2. **Make Predictions:**
   - Use `main_final.py` to generate predictions and evaluate performance.
    - Use `Model_Train/predict_narx.py` to generate predictions afrom NARX model.
3. **Uncertainty Analysis:**
   - Run `Additional_task_2.py` for advanced uncertainty quantification and comparison.

---

## Authors

- Shibin Paul
- Nishitkumar Karkar
- Aadhithya Krishnakumar
- Sankar Nair

---

**Note:** For detailed usage instructions and parameter settings, refer to the comments and docstrings within each script. 