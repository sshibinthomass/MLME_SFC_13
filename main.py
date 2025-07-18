#%%
import numpy as np
import time
import pandas as pd
import tensorflow as tf
import json, pickle, shutil
from pathlib import Path

# --- IMPORT HELPERS ---
from preprocessing_final import (
    clean_iqr, read_txt, predict_closed, add_cqr, detect_cluster, metric_table,
    smooth_log_psd, predict_recursive_open, plot_ts, plot_scatter
)

"""
GROUP:13
    Shibin Paul
    Nishitkumar Karkar
    Aadhithya Krishnakumar
    Sankar Santhosh Nair

Main script for computing MSE/MAE metrics per file with PLOTS & Predictions saved in file (if required).
Requires separate preprocessing module (preprocessing_final.py) for data cleaning, plotting, etc.
"""

# === EASY TOGGLE FLAGS ===
SAVE_FILES = True   # Toggle file/CSV saving ON/OFF
SAVE_PLOTS = True  # Toggle plot generation ON/OFF

# --- PATHS AND GLOBALS ---
file_path   = Path(r"Model_Train/Beat-the-Felix")         # Directory with test file(s)
MODEL_ROOT = Path(r"Model_Train/Model")          # Directory with saved models
OUT_DIR    = Path(r"Beat-the-Felix-Prediction") / "OUTPUTS"   # Change if needed
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(exist_ok=True)

# --- LOAD METADATA FROM TRAINING ---
meta       = json.loads((MODEL_ROOT/'metadata.json').read_text())
STATE_COLS = meta['state_cols']
EXOG_COLS  = meta['exog_cols']
LAG        = meta['lag']
CLUST_COLS = STATE_COLS + EXOG_COLS

# --- DATA PROCESSING ---
def process_data(file_path):
    """Read, clean, and preprocess a single data file."""
    df = read_txt(file_path)
    df = clean_iqr(df)
    df = smooth_log_psd(df, columns=['d10', 'd50', 'd90'], window=5)
    return df

# --- MODEL LOADING ---
def load_cluster_models(cid, model_root=MODEL_ROOT):
    """Load the scalers and trained model for a specific cluster."""
    scX = pickle.loads((model_root/f'narx/scaler_X_{cid}.pkl').read_bytes())
    scY = pickle.loads((model_root/f'narx/scaler_Y_{cid}.pkl').read_bytes())
    narx = tf.keras.models.load_model(model_root/f'narx/cluster_{cid}.keras', compile=False)
    return scX, scY, narx

# --- EVALUATION/PREDICTION ---
def evaluate_on_file(file_path, out_dir, model_root=MODEL_ROOT):
    t0 = time.time()
    """Run full prediction pipeline on a single file, save results and plots if toggled ON."""
    stem = file_path.stem
    out_f = out_dir / stem
    if SAVE_FILES or SAVE_PLOTS:
       
        out_f.mkdir(parents=True, exist_ok=True)

    # 1. Data Processing
    df = process_data(file_path)

    # 2. Cluster Detection 
    cid = detect_cluster(df)
    scX, scY, narx = load_cluster_models(cid, model_root)
    print(f"\nProcessing {stem}, [Cluster {cid}]")
    print(f"MODEL INFO:  LAG={LAG}, STATE_COLS={STATE_COLS}, EXOG_COLS={EXOG_COLS}")
    print(f"Input dimension expected by model: {narx.input_shape}")
    print("Please wait, process is starting...")
    print("\nExpected time :: 120 Seconds\n")
          
    # 3. Prediction
    df_closed, X_closed, y_closed = predict_closed(df, scX, scY, narx)
    df_c = add_cqr(df_closed, X_closed, y_closed, mode="closed")
    df_c = df_c.reset_index(drop=True)
    df_open, X_open, y_open = predict_recursive_open(df, scX, scY, narx, lag=LAG)
    df_o = add_cqr(df_open, X_open, y_open, mode="open")
    df_o = df_o.reset_index(drop=True)

    # 4. Merge results
    df_pred = pd.concat(
        [df_c, 
         df_o[[f"{c}_{m}" for c in STATE_COLS
                              for m in ("open", "open_lo", "open_hi")]]],
        axis=1)
    
    # 5. File saving (CSV)
    if SAVE_FILES:
        print('Files Save is ON \n')
        df_pred.to_csv(out_f/"predictions.csv", index=False)
    
    # 6. Plots
    if SAVE_PLOTS:
        print('Plots Save is ON \n')
        plot_ts(df_pred, out_f, mode="closed")
        plot_ts(df_pred, out_f, mode="open")
        plot_scatter(df_pred, out_f, mode="closed")  
        plot_scatter(df_pred, out_f, mode="open")

    # 7. Metrics
    m_closed = metric_table(df_pred, mode="closed")
    m_open = metric_table(df_pred, mode="open")
    result_row = {"file": stem, **m_closed, **{f"{k}_open": v for k, v in m_open.items()}}

    t1 = time.time()
    print(f"Time taken for predictions: {t1-t0:.2f} Seconds")
    return result_row

# --- MAIN ---
def run_prediction_on_dir(test_dir=file_path, out_dir=OUT_DIR):
    summary = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(test_dir.glob("*.txt")):
        try:
            result_row = evaluate_on_file(p, out_dir)
            summary.append(result_row)
        except Exception as e:
            print(f"  {p.stem} skipped  →  {e}")

    # Summarize results (terminal print)
    df_sum = pd.DataFrame(summary)
    states = STATE_COLS
    table_rows = []
    for col in states:
        row = [
            col,
            df_sum.get(f"{col}_MSE", pd.Series([np.nan])).mean(),
            df_sum.get(f"{col}_MAE", pd.Series([np.nan])).mean(),
            df_sum.get(f"{col}_MSE_open", pd.Series([np.nan])).mean(),
            df_sum.get(f"{col}_MAE_open", pd.Series([np.nan])).mean()
        ]
        table_rows.append(row)

    header = (
        f"{'':<6} {'Closed loop':^25} {'Open Loop':^25}\n"
        f"{'State':<6} {'MSE':>10} {'MAE':>10}  {'MSE':>10} {'MAE':>10}"
    )
    print(header)
    for row in table_rows:
        print(f"{row[0]:<6} {row[1]:10.3e} {row[2]:10.3e}  {row[3]:10.3e} {row[4]:10.3e}")

    # Optionally save the summary table as CSV
    if SAVE_FILES:
        df_sum.to_csv(out_dir/"metrics_summary.csv", index=False)

# --- RUN MAIN ---
if __name__ == "__main__":
    run_prediction_on_dir()

# %%
