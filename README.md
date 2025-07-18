<h1 style="animation: blinker 1s linear infinite; color: #00E5FF;">
  We've outperformed the Felix model!!!
</h1>

<!-- Add this in an HTML-compatible markdown viewer -->
<style>
@keyframes blinker { 50% { opacity: 0; } }
</style>


## Dependencies

This project requires **Python 3.10**.

> **Note:** The project was developed on **Linux** to support the latest version of TensorFlow GPU. However, the prediction part has also been tested on **Windows CPU**.

The following Python packages are required (see `requirements.txt`):

```
numpy==2.1.3
pandas==2.3.1
tensorflow==2.19.0
matplotlib==3.10.3
scikit-learn==1.7.0
scipy==1.15.3
tqdm==4.67.1
seaborn==0.13.2
```

---

# Deployment Checklist

## 1. Clone the Repository

```sh
cd D:\Projects  # Navigate to the folder where you want to clone the repo

git clone https://github.com/sshibinthomass/MLME_SFC_13.git  # Clone the repo

cd MLME_SFC_13  # Go to the project folder
```

## 2. Set Up the Environment

You can use either **Conda** or a standard **Python venv**. Below are instructions for using `uv`:

```sh
pip install uv  # Install UV if not already done

uv venv -p python3.10 .venv  # Create a virtual environment with Python 3.10

uv init  # Initialize the uv project (if not done already. Ignore error)

.venv\Scripts\activate  # Activate the virtual environment 

uv add -r requirements.txt  # Install the requirements

uv pip install tensorflow  # To install TensorFlow (Not added in requirements due to version issue between windows and linux)
```

## 3. Run Predictions

- Move the test file inside `Model_Train/Beat-the-Felix` and remove the existing Beat-the-Felix file.
- Run the following commands:

```sh
python main.py  # To run the final prediction
python Additional_task_2.py  # To run the additional task
```

---

# MLME Project - Group 13

This repository contains all code, models, and data processing scripts for the MLME Project of group 13, focusing on time series prediction, uncertainty quantification, and model evaluation for SFC (Slug Flow Crystallization) data.

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
├── main.py (To test the final prediction)
├── preprocessing_final.py (Preprocessing the test file)
├── Additional_task_2.py (TO predict the additional Task)
├── Model_Train/
│   ├── training_model.py (Full training script to train the model)
│   ├── predict_narx.py (Predict the NARX model)
│   ├── Model/ (All Trained models)
│   │   ├──narx (Trained NARX model)
│   │   ├──qr (Trained QR model)
│   │   ├──data_visualization (Comparision between processed and preprocessed Data)
│   │   └──clustering_analysis (Clustered Files)
│   │
│   ├── Data/ (Data to be Trained)
│   └── Beat-the-Felix/ (Place the test file inside this folder)
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

- Shibin Thomas Stanley Paul
- Nishitkumar Karkar
- Aadhithya Krishnakumar
- Sankar Santhosh Nair

---

**Note:** For detailed usage instructions and parameter settings, refer to the comments and docstrings within each script. 
