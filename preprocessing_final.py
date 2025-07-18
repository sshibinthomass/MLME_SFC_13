#!/usr/bin/env python3
"""
Enhanced Prediction Script for SFC Data with Open-Loop and Closed-Loop
- Integrates all critical preprocessing and helper functions from training
- Handles one or many files (Beat-The-Felix or any .txt)
- Produces CQR-calibrated uncertainty intervals and summary metrics
"""
#%%
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json, pickle
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- PATHS AND GLOBALS (update as needed) ---
TEST_DIR   = Path(r"Model_Train/Beat-the-Felix")     #Directory with test file(s)
MODEL_ROOT = Path(r"Model_Train/Model")      # Directory with saved models

# --- LOAD METADATA FROM TRAINING ---
meta       = json.loads((MODEL_ROOT/'metadata.json').read_text())
STATE_COLS = meta['state_cols']
EXOG_COLS  = meta['exog_cols']
LAG        = meta['lag']
CLUST_COLS = STATE_COLS + EXOG_COLS
PSD_COLS   = ('d10', 'd50', 'd90')


# --- DATA CLEANING/PROCESSING(similar to training) ---

def read_txt(p: Path) -> pd.DataFrame:
    """Read TAB-separated SFC data."""
    return pd.read_csv(p, sep='\t', engine='python').apply(pd.to_numeric, errors='coerce')

def clean_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
      - Log-transform d10/d50/d90 before outlier handling
      - Use stricter IQR (1.5x) for d10/d50/d90
      - Added engineered features after cleaning
    """
    available_cols = [col for col in CLUST_COLS if col in df.columns]
    df = df.dropna(subset=available_cols) #Drop empty rows

    #Log-transform d10/d50/d90 before outlier handling
    for col in ['d10', 'd50', 'd90']:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    for column in df.columns:
        if column in available_cols:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            if column in ['T_PM', 'T_TM']:
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR
                vals = df[column].values.copy()
                for i in range(len(vals)):
                    if not (lower_bound <= vals[i] <= upper_bound):
                        prev_idx = i - 1
                        while prev_idx >= 0 and not (lower_bound <= vals[prev_idx] <= upper_bound):
                            prev_idx -= 1
                        next_idx = i + 1
                        while next_idx < len(vals) and not (lower_bound <= vals[next_idx] <= upper_bound):
                            next_idx += 1
                        if prev_idx >= 0 and next_idx < len(vals):
                            vals[i] = 0.5 * (vals[prev_idx] + vals[next_idx])
                        elif prev_idx >= 0:
                            vals[i] = vals[prev_idx]
                        elif next_idx < len(vals):
                            vals[i] = vals[next_idx]
                df[column] = vals
            elif column == 'c':
                lower_bound = Q1 - 6.0 * IQR
                upper_bound = Q3 + 6.0 * IQR
                vals = df[column].values.copy()
                mask = ~((lower_bound <= vals) & (vals <= upper_bound))
                i = 0
                n = len(vals)
                while i < n:
                    if mask[i]:
                        run_start = i
                        while i < n and mask[i]:
                            i += 1
                        run_end = i
                        prev_idx = run_start - 1
                        next_idx = run_end
                        prev_val = vals[prev_idx] if prev_idx >= 0 else None
                        next_val = vals[next_idx] if next_idx < n else None
                        if prev_val is not None and next_val is not None:
                            for j in range(run_start, run_end):
                                alpha = (j - run_start + 1) / (run_end - run_start + 1)
                                vals[j] = (1 - alpha) * prev_val + alpha * next_val
                        elif prev_val is not None:
                            vals[run_start:run_end] = prev_val
                        elif next_val is not None:
                            vals[run_start:run_end] = next_val
                    else:
                        i += 1
                df[column] = vals
            elif column in ['d10', 'd50', 'd90']:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            else:
                lower_bound = Q1 - 2 * IQR
                upper_bound = Q3 + 2 * IQR
                df[column] = df[column].apply(
                    lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
                )

    #Add engineered features (span, ratios) in original scale
    if all(col in df.columns for col in ['d10', 'd50', 'd90']):
        d10 = np.expm1(df['d10'])
        d50 = np.expm1(df['d50'])
        d90 = np.expm1(df['d90'])
        df['span_d90_d10'] = d90 - d10
        df['ratio_d90_d50'] = d90 / d50.replace(0, np.nan)
        df['ratio_d50_d10'] = d50 / d10.replace(0, np.nan)

    return df

def smooth_log_psd(df, columns=['d10', 'd50', 'd90'], window=5):
    """
    Smooth selected PSD columns (d10/d50/d90) in log-space and replace in the DataFrame.
    """
    df = df.copy()
    for col in columns:
        #Avoid log(0) by clipping to a small value
        clipped = np.clip(df[col].values, 1e-9, None)
        log_vals = np.log(clipped)
        log_smoothed = pd.Series(log_vals).rolling(window, center=True, min_periods=1).mean()
        smoothed = np.exp(log_smoothed)
        df[col] = smoothed
    return df


# --- CLUSTER DETECTION (from training) ---

sc_feat = pickle.loads((MODEL_ROOT/'feature_scaler.pkl').read_bytes())
kmeans  = pickle.loads((MODEL_ROOT/'kmeans_model.pkl').read_bytes())

def file_signature(df):
    """Same feature vector used for clustering in training."""
    arr = df[CLUST_COLS].values
    return np.concatenate([arr.mean(0), arr.std(0), arr.min(0), arr.max(0)]).reshape(1,-1)

def detect_cluster(df) -> int:
    return int(kmeans.predict(sc_feat.transform(file_signature(df)))[0])

# --- LAG MATRIX FOR NARX ---

def build_lagged(df, lag=LAG):
    """Build lagged feature matrix (newest-to-oldest) for open-loop."""
    rows = []
    for i in range(lag, len(df)-1):
        row = []
        for l in range(0, lag+1): 
            idx = i - l
            row.extend(df[CLUST_COLS].iloc[idx].values)
        rows.append(row)
    return np.asarray(rows, np.float32)

def load_cluster(cid):
    """Load cluster-specific scalers and NARX model."""
    scX = pickle.loads((MODEL_ROOT/f'narx/scaler_X_{cid}.pkl').read_bytes())
    scY = pickle.loads((MODEL_ROOT/f'narx/scaler_Y_{cid}.pkl').read_bytes())
    narx = tf.keras.models.load_model(MODEL_ROOT/f'narx/cluster_{cid}.keras', compile=False)
    return scX, scY, narx

# ---- OPEN-LOPP / CLOSED-LOOP implementation ----

'''### (ADDITIONAL TASK) 1: open-loop prediction: i.e., always feed back predicted y in open loop ###'''


def predict_recursive_open(df, scX, scY, narx, lag=LAG):
    '''
    Recursive open-loop prediction for trained NARX model.
    '''
    total_steps = len(df) - (lag+1)
    preds = []
    Xs_all=[]
    #Prepare initial lag window from true data
    lag_window = []
    for l in range(lag+1):
        idx = l
        row = np.concatenate([df.loc[idx, STATE_COLS].values, df.loc[idx, EXOG_COLS].values])
        lag_window.append(row)
    lag_window = np.array(lag_window)  
    #Predict recursively
    for t in range(total_steps):
        #Build NARX input vector: newest-to-oldest
        x_input = []
        for l in range(lag+1):
            x_input.extend(lag_window[-(l+1)])  #reverse order: newest first
        x_input = np.array(x_input)
        x_input_scaled = scX.transform([x_input])
        Xs_all.append(x_input_scaled[0])
        y_pred_scaled = narx.predict(x_input_scaled, verbose=0)
        y_pred = scY.inverse_transform(y_pred_scaled)[0]
        preds.append(y_pred)
        #Prepare exog for next step
        exog_next = df.loc[lag+1+t, EXOG_COLS].values
        #Update lag_window: drop oldest
        next_row = np.concatenate([y_pred, exog_next])
        lag_window = np.vstack([lag_window[1:], next_row])
    df_out = df.iloc[lag+1:].reset_index(drop=True)
    preds = np.array(preds)
    Xs = np.array(Xs_all)
    return df_out, Xs, preds


def predict_closed(df, scX, scY, narx):
    X      = build_lagged(df)
    Xs     = scX.transform(X)
    y_pred = scY.inverse_transform(narx.predict(Xs, verbose=0))
    return df.iloc[LAG+1:].reset_index(drop=True), Xs, y_pred

# --- LOAD QR + CQR DELTAS ---
QR = {}
for col in STATE_COLS:
    for q in (0.1, 0.9):
        QR[(col, q)] = tf.keras.models.load_model(MODEL_ROOT/f'qr/{col}_{q:.1f}.keras', compile=False)
DELTAS = pickle.loads((MODEL_ROOT/'conformal_deltas.pkl').read_bytes())

#ifneeded :adjust deltas if model requires calibration
DELTAS['c']    *= 1.6
DELTAS['d10']  *= 0.3
DELTAS['d50']  *= 0.3
DELTAS['d90']  *= 0.3
DELTAS['T_PM'] *= 2.5
DELTAS['T_TM'] *= 2.5

def add_cqr(df, Xs, base_pred, mode: str):
    '''Attach CQR bounds to DataFrame for each state variable.'''
    out = df.copy()
    for i, col in enumerate(STATE_COLS):
        lo = QR[(col, 0.1)].predict(Xs, verbose=0).flatten()
        hi = QR[(col, 0.9)].predict(Xs, verbose=0).flatten()
        out[f"{col}_{mode}"]    = base_pred[:, i]
        out[f"{col}_{mode}_lo"] = base_pred[:, i] + lo - DELTAS[col]
        out[f"{col}_{mode}_hi"] = base_pred[:, i] + hi + DELTAS[col]
    return out


def metric_table(df: pd.DataFrame, mode: str):
    '''Compute MAE, MSE, R2, coverage for each variable.'''
    res = {}
    for col in STATE_COLS:
        y_true = df[col].values
        y_pred = df[f"{col}_{mode}"].values
        lo     = df[f"{col}_{mode}_lo"].values
        hi     = df[f"{col}_{mode}_hi"].values
        msk    = np.isfinite(y_true) & np.isfinite(y_pred)
        res[f"{col}_MAE"] = mean_absolute_error(y_true[msk], y_pred[msk])
        res[f"{col}_MSE"] = mean_squared_error(y_true[msk], y_pred[msk])
        res[f"{col}_R2"]  = r2_score(y_true[msk], y_pred[msk])
        inside            = (y_true >= lo) & (y_true <= hi)
        res[f"{col}_COV"] = 100. * inside[msk].mean()
    return res

def plot_ts(df, out, mode):
    t = np.arange(len(df))
    for col in STATE_COLS:
        plt.figure(figsize=(7,3))
        plt.plot(t, df[col],  label='true',color="#1f77b4",linestyle='--', lw=1)
        plt.plot(t, df[f'{col}_{mode}'], label='predicted',color="#ff7f0e", lw=1)
        plt.fill_between(t, df[f'{col}_{mode}_lo'], df[f'{col}_{mode}_hi'],
                         alpha=.25, label='90 % Predicted Interval')
        
        #styling
        # Variable-specific y-axis limits (overrides auto-limits)
        if col in ['T_TM', 'T_PM']:
            plt.ylim(311, None)
        elif col == 'c':
            plt.ylim(0.179, None)
        #set lower bound; upper bound auto
        else:
            plt.ylim(None, None)  #fallback for any other variable

        plt.xlim(0, len(df)-1)
        plt.xlabel(f"Steps", fontsize=13)
        plt.ylabel(f"{col}", fontsize=14)
        plt.title(f"{col} - {mode.capitalize()} Prediction: Time Series", fontsize=15)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=5, loc='upper left', frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.4)
        # optional log‐scale for PSD
        if col in ('d10','d50','d90'):
            plt.yscale('log')
            plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
       
        plt.savefig(out/f"{col}_{mode}.png", dpi=200)
        plt.close()

def plot_scatter(df, out, mode):
    """
    Enhanced scatter plot for each state variable.
    - 1:1 reference line
    - R² score on plot
    - Clear labels, thicker points/line
    """
    for col in STATE_COLS:
        plt.figure(figsize=(5, 5))
        x = df[col]
        y = df[f'{col}_{mode}'] 
        #1:1 line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        lim_margin = (max_val - min_val) * 0.04
        lims = [min_val - lim_margin, max_val + lim_margin]
        plt.plot(lims, lims, 'k--', lw=2, label='1:1 Reference')
        plt.scatter(x, y, s=18, alpha=0.7, color="#1f77b4", edgecolor='k')
        #R_Square
        msk = np.isfinite(x) & np.isfinite(y)
        r2 = r2_score(x[msk], y[msk])
        plt.text(0.03, 0.94, f"$R^2$ = {r2:.3f}", transform=plt.gca().transAxes,
                 fontsize=13, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", alpha=0.7))

        plt.xlabel(f"True {col}", fontsize=13)
        plt.ylabel(f"Predicted {col}", fontsize=13)
        plt.title(f"{col} - {mode.capitalize()} Scatter plot", fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.xlim(lims)
        plt.ylim(lims)
        plt.legend(fontsize=12, loc='best', framealpha=0.85)
        plt.tight_layout()
        plt.savefig(out / f"{col}_{mode}_scatter.png", dpi=200)
        plt.close()
# %%
