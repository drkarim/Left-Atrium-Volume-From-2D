#!/usr/bin/env python3
"""
regression_la_volume_with_plots.py

Fits two regression models to predict true LA volume (3D) from 2D measurements
and saves correlation plots with regression lines and coefficients.

Model 1 (Area–Length calibrated):
    V = alpha * ((A1 * A2) / L) + beta

Model 2 (Multiple linear regression):
    V = a*A1 + b*A2 + c*L + d

Input:
    data.csv in the data folder.

Required columns in data.csv:
    - LA-A1
    - LA-A2
    - LA-L
    - LA-V-3D   (true/reference volume)

Outputs:
    - model1_area_length_regression.png
    - model2_multivariate_regression.png
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# File and column settings
# -------------------------
DATA_PATH = Path("./data/data.csv")

A1_COL = "LA-A1"
A2_COL = "LA-A2"
L_COL  = "LA-L"
Y_COL  = "LA-V-3D"

# -------------------------
# Utility functions
# -------------------------
def metrics(y_true, y_pred):
    resid = y_true - y_pred
    mae = np.mean(np.abs(resid))
    rmse = np.sqrt(np.mean(resid**2))
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = 1 - np.sum(resid**2) / np.sum((y_true - np.mean(y_true))**2)
    return r, r2, mae, rmse


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    for col in [A1_COL, A2_COL, L_COL, Y_COL]:
        if col not in df.columns:
            print(f"ERROR: Missing column '{col}'", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)

    # Clean data
    df = df[[A1_COL, A2_COL, L_COL, Y_COL]].dropna()
    df = df[df[L_COL] != 0]

    A1 = df[A1_COL].to_numpy(float)
    A2 = df[A2_COL].to_numpy(float)
    L  = df[L_COL].to_numpy(float)
    y  = df[Y_COL].to_numpy(float)

    # ==========================================================
    # Model 1: Area–Length calibrated regression
    # ==========================================================
    x1 = (A1 * A2) / L
    X1 = np.column_stack([x1, np.ones_like(x1)])
    alpha, beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    y1_pred = alpha * x1 + beta

    r1, r2_1, mae1, rmse1 = metrics(y, y1_pred)

    # Plot Model 1
    plt.figure(figsize=(7, 7))
    plt.scatter(y, y1_pred, color="blue", alpha=0.8)
    plt.plot([y.min(), y.max()], [y.min(), y.max()],
             "k--", label="Identity line")

    plt.xlabel("True LA Volume (3D)")
    plt.ylabel("Predicted LA Volume")
    plt.title("Area–Length Calibrated Regression")

    textstr = (
        f"V = α·(A1·A2/L) + β\n"
        f"α = {alpha:.3g}\n"
        f"β = {beta:.3g}\n"
        f"R = {r1:.2f}, R² = {r2_1:.2f}"
    )

    plt.text(0.05, 0.95, textstr,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/model1_area_length_regression.png", dpi=300)
    plt.close()

    # ==========================================================
    # Model 2: Multivariate linear regression
    # ==========================================================
    X2 = np.column_stack([A1, A2, L, np.ones_like(A1)])
    a, b, c, d = np.linalg.lstsq(X2, y, rcond=None)[0]
    y2_pred = a*A1 + b*A2 + c*L + d

    r2, r2_2, mae2, rmse2 = metrics(y, y2_pred)

    # Plot Model 2
    plt.figure(figsize=(7, 7))
    plt.scatter(y, y2_pred, color="green", alpha=0.8)
    plt.plot([y.min(), y.max()], [y.min(), y.max()],
             "k--", label="Identity line")

    plt.xlabel("True LA Volume (3D)")
    plt.ylabel("Predicted LA Volume")
    plt.title("Multivariate Linear Regression")

    textstr = (
        f"V = a·A1 + b·A2 + c·L + d\n"
        f"a = {a:.3g}\n"
        f"b = {b:.3g}\n"
        f"c = {c:.3g}\n"
        f"d = {d:.3g}\n"
        f"R = {r2:.2f}, R² = {r2_2:.2f}"
    )

    plt.text(0.05, 0.95, textstr,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./plots/model2_multivariate_regression.png", dpi=300)
    plt.close()

    # ==========================================================
    # Console summary
    # ==========================================================
    print("=== Model 1: Area–Length Calibrated ===")
    print(f"alpha = {alpha:.6g}, beta = {beta:.6g}")
    print(f"R = {r1:.3f}, R^2 = {r2_1:.3f}, MAE = {mae1:.2f}, RMSE = {rmse1:.2f}\n")

    print("=== Model 2: Multivariate Linear ===")
    print(f"a = {a:.6g}, b = {b:.6g}, c = {c:.6g}, d = {d:.6g}")
    print(f"R = {r2:.3f}, R^2 = {r2_2:.3f}, MAE = {mae2:.2f}, RMSE = {rmse2:.2f}")

    print("\nSaved plots:")
    print(" - model1_area_length_regression.png")
    print(" - model2_multivariate_regression.png")


if __name__ == "__main__":
    main()
