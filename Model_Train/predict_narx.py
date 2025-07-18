#!/usr/bin/env python3
"""
Prediction and Testing Script for SFC Project NARX Models

This script provides functionality to:
1. Load trained NARX models from model_5files7/narx
2. Make predictions on new data
3. Test models on existing data
4. Evaluate model performance with various metrics
5. Generate prediction plots and analysis
6. Export results to CSV files and graphs in a new folder for each run

Author: Based on latest_modeltrain1.py training script
"""
#%%
import os
import random
import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from datetime import datetime

# Set reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration
MODEL_DIR = Path("Model_Train/Model")  # Updated to match training script
PSD_COLS = ('d10', 'd50', 'd90')
now = datetime.now().strftime("%Y%m%d_%H%M%S")
# Load metadata
with open(MODEL_DIR / 'metadata.json', 'r') as f:
    metadata = json.load(f)

STATE_COLS = metadata['state_cols']
EXOG_COLS = metadata['exog_cols']
LAG = metadata['lag']
CLUST_COLS = STATE_COLS + EXOG_COLS

print(f"Loaded configuration:")
print(f"  State columns: {STATE_COLS}")
print(f"  Exogenous columns: {EXOG_COLS}")
print(f"  Lag: {LAG}")
print(f"  Model directory: {MODEL_DIR}")

class NARXPredictor:
    """Class for making predictions using trained NARX models."""
    
    def __init__(self, model_dir: Path):
        """Initialize predictor with trained models."""
        self.model_dir = model_dir
        # Automatically detect number of clusters by checking available model files
        cluster_files = list(model_dir.glob('narx/cluster_*.keras'))
        self.n_clusters = len(cluster_files)
        print(f"Detected {self.n_clusters} clusters from model files")
        
        # Load clustering models
        self.feature_scaler = pickle.loads((model_dir / 'feature_scaler.pkl').read_bytes())
        self.kmeans = pickle.loads((model_dir / 'kmeans_model.pkl').read_bytes())
        
        # Load NARX models and scalers
        self.narx_models = {}
        self.scalers_x = {}
        self.scalers_y = {}
        
        for cid in range(self.n_clusters):
            # Load model
            self.narx_models[cid] = tf.keras.models.load_model(
                model_dir / f'narx/cluster_{cid}.keras',
                compile=False
            )
            
            # Load scalers
            self.scalers_x[cid] = pickle.loads(
                (model_dir / f'narx/scaler_X_{cid}.pkl').read_bytes()
            )
            self.scalers_y[cid] = pickle.loads(
                (model_dir / f'narx/scaler_Y_{cid}.pkl').read_bytes()
            )
        
        print(f"Loaded {self.n_clusters} NARX models successfully")
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data using the same method as training."""
        # Clean data using IQR method - only process available columns
        available_cols = [col for col in CLUST_COLS if col in df.columns]
        df = df.dropna(subset=available_cols)
        
        for column in df.columns:
            if column in available_cols:  # Only process relevant columns
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df[column] = df[column].apply(
                    lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
                )
        
        return df
    
    def make_xy(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create input-output pairs for NARX model."""
        X, Y = [], []
        for i in range(LAG, len(df) - 1):
            # newest-to-oldest slice
            hist = []
            for l in range(0, LAG + 1):
                idx = i - l
                hist.extend(df[STATE_COLS + EXOG_COLS].iloc[idx].values)
            X.append(hist)
            Y.append(df[STATE_COLS].iloc[i + 1].values)
        return np.asarray(X, np.float32), np.asarray(Y, np.float32)
    
    def predict_cluster(self, df: pd.DataFrame) -> int:
        """Predict which cluster the data belongs to."""
        # Calculate features (same as training)
        arr = df[CLUST_COLS].values
        feat = np.concatenate([arr.mean(0), arr.std(0), arr.min(0), arr.max(0)])
        feat_scaled = self.feature_scaler.transform([feat])
        cluster_id = self.kmeans.predict(feat_scaled)[0]
        return int(cluster_id)
    
    def predict(self, df: pd.DataFrame, return_cluster_info: bool = False) -> np.ndarray:
        """Make predictions on the given dataframe."""
        # Preprocess data
        df_clean = self.preprocess_data(df)
        
        # Determine cluster
        cluster_id = self.predict_cluster(df_clean)
        
        # Create input-output pairs
        X, Y_true = self.make_xy(df_clean)
        
        if len(X) == 0:
            print("Warning: No valid input-output pairs found")
            return np.array([])
        
        # Scale inputs
        X_scaled = self.scalers_x[cluster_id].transform(X)
        
        # Make predictions
        Y_pred_scaled = self.narx_models[cluster_id].predict(X_scaled, verbose=0)
        Y_pred = self.scalers_y[cluster_id].inverse_transform(Y_pred_scaled)
        
        if return_cluster_info:
            return Y_pred, Y_true, cluster_id
        else:
            return Y_pred, Y_true
    
    def evaluate_predictions(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict:
        """Evaluate prediction performance."""
        if len(Y_true) == 0 or len(Y_pred) == 0:
            return {}
        
        metrics = {}
        
        # Overall metrics
        metrics['mse'] = mean_squared_error(Y_true, Y_pred)
        metrics['mae'] = mean_absolute_error(Y_true, Y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(Y_true, Y_pred)
        
        # Per-column metrics
        for i, col in enumerate(STATE_COLS):
            metrics[f'{col}_mse'] = mean_squared_error(Y_true[:, i], Y_pred[:, i])
            metrics[f'{col}_mae'] = mean_absolute_error(Y_true[:, i], Y_pred[:, i])
            metrics[f'{col}_rmse'] = np.sqrt(metrics[f'{col}_mse'])
            metrics[f'{col}_r2'] = r2_score(Y_true[:, i], Y_pred[:, i])
        
        return metrics
    
    def plot_predictions(self, Y_true: np.ndarray, Y_pred: np.ndarray, 
                        save_path: Optional[Path] = None, title: str = "Predictions"):
        """Plot true vs predicted values."""
        if len(Y_true) == 0 or len(Y_pred) == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(STATE_COLS):
            ax = axes[i]
            
            # Scatter plot
            ax.scatter(Y_true[:, i], Y_pred[:, i], alpha=0.6, s=10)
            
            # Perfect prediction line
            min_val = min(Y_true[:, i].min(), Y_pred[:, i].min())
            max_val = max(Y_true[:, i].max(), Y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            # Calculate R²
            r2 = r2_score(Y_true[:, i], Y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(Y_true[:, i], Y_pred[:, i]))
            
            # Format RMSE for display
            if rmse < 1e-6:
                rmse_str = f"{rmse:.2e}"
            else:
                rmse_str = f"{rmse:.6f}"
            
            ax.set_xlabel(f'True {col}')
            ax.set_ylabel(f'Predicted {col}')
            ax.set_title(f'{col}\nR² = {r2:.3f}, RMSE = {rmse_str}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved prediction plot to {save_path}")
        
        plt.show()
    
    def plot_time_series(self, df: pd.DataFrame, Y_pred: np.ndarray, 
                        save_path: Optional[Path] = None, max_points: int = 1000):
        """Plot time series of predictions vs actual values."""
        if len(Y_pred) == 0:
            print("No predictions to plot")
            return
        
        # Get the corresponding actual values
        df_clean = self.preprocess_data(df)
        _, Y_true = self.make_xy(df_clean)
        
        if len(Y_true) == 0:
            print("No actual values to plot")
            return
        
        # Limit points for visualization
        n_points = min(len(Y_pred), max_points)
        Y_pred_plot = Y_pred[-n_points:]
        Y_true_plot = Y_true[-n_points:]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(STATE_COLS):
            ax = axes[i]
            
            x_axis = range(len(Y_pred_plot))
            ax.plot(x_axis, Y_true_plot[:, i], 'b-', label='True', alpha=0.7)
            ax.plot(x_axis, Y_pred_plot[:, i], 'r-', label='Predicted', alpha=0.7)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(col)
            ax.set_title(f'{col} - Time Series')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved time series plot to {save_path}")
        
        plt.show()

def read_txt(path: Path) -> pd.DataFrame:
    """Read TAB-separated text file into DataFrame."""
    return pd.read_csv(path, sep='\t', engine='python').apply(pd.to_numeric, errors='coerce')

def export_metrics_to_csv(metrics_list: List[Dict], file_paths: List[Path], 
                         output_dir: Path, timestamp: str):
    """Export all metrics to CSV files."""
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # 1. Overall metrics per file
    overall_data = []
    for i, (metrics, file_path) in enumerate(zip(metrics_list, file_paths)):
        if metrics:
            row = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'cluster_id': metrics.get('cluster_id', 'N/A'),
                'num_predictions': metrics.get('num_predictions', 0),
                'overall_r2': metrics.get('r2', 0),
                'overall_rmse': metrics.get('rmse', 0),
                'overall_mae': metrics.get('mae', 0),
                'overall_mse': metrics.get('mse', 0),
                'timestamp': timestamp
            }
            overall_data.append(row)
    
    if overall_data:
        overall_df = pd.DataFrame(overall_data)
        overall_csv_path = output_dir / f"overall_metrics_{timestamp}.csv"
        overall_df.to_csv(overall_csv_path, index=False)
        print(f"Exported overall metrics to: {overall_csv_path}")
    
    # 2. Per-column metrics per file
    column_data = []
    for i, (metrics, file_path) in enumerate(zip(metrics_list, file_paths)):
        if metrics:
            for col in STATE_COLS:
                row = {
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'cluster_id': metrics.get('cluster_id', 'N/A'),
                    'column': col,
                    'r2': metrics.get(f'{col}_r2', 0),
                    'rmse': metrics.get(f'{col}_rmse', 0),
                    'mae': metrics.get(f'{col}_mae', 0),
                    'mse': metrics.get(f'{col}_mse', 0),
                    'timestamp': timestamp
                }
                column_data.append(row)
    
    if column_data:
        column_df = pd.DataFrame(column_data)
        column_csv_path = output_dir / f"column_metrics_{timestamp}.csv"
        column_df.to_csv(column_csv_path, index=False)
        print(f"Exported column metrics to: {column_csv_path}")
    
    # 3. Summary statistics
    if overall_data:
        summary_data = {
            'metric': ['r2', 'rmse', 'mae', 'mse'],
            'mean': [
                np.mean([m.get('overall_r2', 0) for m in overall_data]),
                np.mean([m.get('overall_rmse', 0) for m in overall_data]),
                np.mean([m.get('overall_mae', 0) for m in overall_data]),
                np.mean([m.get('overall_mse', 0) for m in overall_data])
            ],
            'std': [
                np.std([m.get('overall_r2', 0) for m in overall_data]),
                np.std([m.get('overall_rmse', 0) for m in overall_data]),
                np.std([m.get('overall_mae', 0) for m in overall_data]),
                np.std([m.get('overall_mse', 0) for m in overall_data])
            ],
            'min': [
                np.min([m.get('overall_r2', 0) for m in overall_data]),
                np.min([m.get('overall_rmse', 0) for m in overall_data]),
                np.min([m.get('overall_mae', 0) for m in overall_data]),
                np.min([m.get('overall_mse', 0) for m in overall_data])
            ],
            'max': [
                np.max([m.get('overall_r2', 0) for m in overall_data]),
                np.max([m.get('overall_rmse', 0) for m in overall_data]),
                np.max([m.get('overall_mae', 0) for m in overall_data]),
                np.max([m.get('overall_mse', 0) for m in overall_data])
            ],
            'timestamp': timestamp
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = output_dir / f"summary_statistics_{timestamp}.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Exported summary statistics to: {summary_csv_path}")
    
    # 4. Cluster distribution
    if overall_data:
        cluster_counts = {}
        for row in overall_data:
            cluster_id = row.get('cluster_id', 'N/A')
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        cluster_data = [
            {'cluster_id': k, 'count': v, 'percentage': v/len(overall_data)*100, 'timestamp': timestamp}
            for k, v in cluster_counts.items()
        ]
        
        cluster_df = pd.DataFrame(cluster_data)
        cluster_csv_path = output_dir / f"cluster_distribution_{timestamp}.csv"
        cluster_df.to_csv(cluster_csv_path, index=False)
        print(f"Exported cluster distribution to: {cluster_csv_path}")

def test_on_file(predictor: NARXPredictor, file_path: Path, output_dir: Path, timestamp: str, save_plots: bool = True) -> Dict:
    """Test the predictor on a single file."""
    print(f"\nTesting on file: {file_path}")
    
    # Read and preprocess data
    df = read_txt(file_path)
    df_clean = predictor.preprocess_data(df)
    
    # Make predictions
    Y_pred, Y_true, cluster_id = predictor.predict(df_clean, return_cluster_info=True)
    
    if len(Y_pred) == 0:
        print("No valid predictions made")
        return {}
    
    # Evaluate
    metrics = predictor.evaluate_predictions(Y_true, Y_pred)
    
    # Add additional metadata
    metrics['cluster_id'] = cluster_id
    metrics['num_predictions'] = len(Y_pred)
    metrics['file_name'] = file_path.name
    metrics['file_path'] = str(file_path)
    
    print(f"Cluster assigned: {cluster_id}")
    print(f"Number of predictions: {len(Y_pred)}")
    print(f"Overall R²: {metrics.get('r2', 0):.3f}")
    print(f"Overall RMSE: {metrics.get('rmse', 0):.6f}")
    print(f"Overall MAE: {metrics.get('mae', 0):.6f}")
    
    # Print per-column metrics
    for col in STATE_COLS:
        r2_key = f'{col}_r2'
        rmse_key = f'{col}_rmse'
        mae_key = f'{col}_mae'
        if r2_key in metrics:
            rmse_val = metrics[rmse_key]
            mae_val = metrics[mae_key]
            
            # Use scientific notation for very small values
            if rmse_val < 1e-6:
                rmse_str = f"{rmse_val:.2e}"
            else:
                rmse_str = f"{rmse_val:.6f}"
                
            if mae_val < 1e-6:
                mae_str = f"{mae_val:.2e}"
            else:
                mae_str = f"{mae_val:.6f}"
                
            print(f"  {col}: R² = {metrics[r2_key]:.3f}, RMSE = {rmse_str}, MAE = {mae_str}")
    
    # Create plots
    if save_plots:
        file_name = file_path.stem
        predictor.plot_predictions(Y_true, Y_pred, 
                                 save_path=output_dir / f"{file_name}_predictions_{timestamp}.png",
                                 title=f"Predictions for {file_name}")
        
        predictor.plot_time_series(df_clean, Y_pred,
                                 save_path=output_dir / f"{file_name}_timeseries_{timestamp}.png")
    
    return metrics

def main():
    """Main function to demonstrate prediction and testing."""
    print("=== NARX Model Prediction and Testing Script ===\n")
    
    # Create output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"Model_Train/NARX_Prediction/prediction_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Initialize predictor
    try:
        predictor = NARXPredictor(MODEL_DIR)
    except Exception as e:
        print(f"Error loading models: {e}")
        return
    
    # Initialize lists to store results
    all_metrics = []
    all_file_paths = []
    
    # Test on calibration data if available
    calib_dir = Path("Model_Train/Beat-the-Felix")
    if calib_dir.exists():
        print(f"\nTesting on calibration data from: {calib_dir}")
        calib_files = list(calib_dir.glob("*.txt"))
        
        if calib_files:
            for file_path in calib_files[:5]:  # Test on first 5 files
                metrics = test_on_file(predictor, file_path, output_dir, timestamp)
                if metrics:
                    all_metrics.append(metrics)
                    all_file_paths.append(file_path)
            
            # Aggregate results
            if all_metrics:
                print("\n=== Aggregate Results ===")
                avg_r2 = np.mean([m.get('r2', 0) for m in all_metrics])
                avg_rmse = np.mean([m.get('rmse', 0) for m in all_metrics])
                avg_mae = np.mean([m.get('mae', 0) for m in all_metrics])
                print(f"Average R²: {avg_r2:.3f}")
                print(f"Average RMSE: {avg_rmse:.6f}")
                print(f"Average MAE: {avg_mae:.6f}")

    
    # Export results to CSV
    if all_metrics:
        print("\n=== Exporting Results to CSV ===")
        export_metrics_to_csv(all_metrics, all_file_paths, output_dir, timestamp)
        print(f"All results have been exported to: {output_dir}")
    
    print(f"\n=== Prediction and Testing Complete ===")
    print(f"All outputs saved to: {output_dir}")

if __name__ == "__main__":
    main() 

#%%