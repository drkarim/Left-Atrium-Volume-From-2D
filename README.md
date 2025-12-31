# LA Volume Correlation Analysis

This project visualizes and quantifies the correlation between estimated and true left atrial (LA) volumes. For project context, see the [Background](#background) section. The workflow below focuses on running the analysis inside **WSL Ubuntu** using [`uv`](https://docs.astral.sh/uv/) to manage a virtual environment and dependencies.

## Prerequisites

1. **Windows Subsystem for Linux (WSL)** with an Ubuntu distribution.
2. **Python 3.10+** available inside WSL (verify with `python3 --version`).
3. `pip` installed for the WSL Python interpreter (`sudo apt install python3-pip` if needed).
4. Access to this repository from WSL. The Windows path `d:\rashe\Documents\amani-project\python` maps to `/mnt/d/rashe/Documents/amani-project/python` inside WSL.

## 1. Install `uv`

```bash
python3 -m pip install --user uv
```

If you prefer a system-wide installation, prepend the command with `sudo` (optional). Ensure `$HOME/.local/bin` is on your PATH so that the `uv` executable is discoverable:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 2. Create the virtual environment

Navigate to the project directory (adjust the mount path if your Windows drive letter differs):

```bash
cd /mnt/d/rashe/Documents/amani-project/python
python3 -m uv venv .venv
```

The command above creates a `.venv` folder in the project root using your WSL Python interpreter.

## 3. Install dependencies from `requirements.txt`

```bash
python3 -m uv pip install --python .venv -r requirements.txt
```

This installs all required packages (NumPy, SciPy, Matplotlib, etc.) into `.venv`.

## 4. Activate the environment and run the analyses

For the correlation analysis, run:
```bash
source .venv/bin/activate
python3 la_volume_correlation_analysis.py
```

For the regression analysis, run: 
```bash
python3 la_volume_regression_analysis.py
```

A Matplotlib window will open showing the scatter plot, best-fit line, and Pearson correlation coefficient. It will save the plots to a `.png` file


When you are finished, deactivate the environment:

```bash
deactivate
```

## Troubleshooting

- **`uv: command not found`**: Make sure `$HOME/.local/bin` is on your PATH, or rerun the install command and start a new shell session.
- **Display/back-end issues**: If Matplotlib cannot open a window inside WSL, install an X server on Windows (e.g., VcXsrv) and export the display with `export DISPLAY=:0` before running the script. Alternatively, replace `plt.show()` with `plt.savefig("la_volume_correlation.png")` to generate an image file instead.

You are now ready to explore the LA volume correlation analysis within WSL Ubuntu.

---

## Background

### 1. Project Background

Left atrial (LA) volume is an important clinical marker, as enlargement of the left atrium is associated with conditions such as atrial fibrillation, stroke, and heart failure. The most accurate way to measure LA volume is by performing full three-dimensional (3D) segmentation of cardiac MRI images. However, this process is time-consuming and requires specialist software and expertise.

In clinical practice, faster methods are often used that estimate LA volume from two-dimensional (2D) measurements taken from standard imaging views. These shortcut methods are quicker but rely on geometric assumptions about the shape of the atrium.

This project investigates how well a 2D-based estimation method approximates the true 3D left atrial volume. The study uses open-source cardiac MRI data and compares estimated volumes against reference volumes derived from 3D segmentation. The work was carried out as part of a student research project exploring medical image analysis techniques.

The project uses images and data from the [open-source dataset LASC benchmark, Xiong et al.](https://www.cardiacatlas.org/atriaseg2018-challenge/atria-seg-data/) and the [paper reference is here](https://www.sciencedirect.com/science/article/abs/pii/S1361841520301961)

Here is an infographic that describes the project:

![Project infographic](images/infographic_1.png)

---

### 2. Methods Overview

Two different measurements of left atrial volume are compared in this project:

#### Estimated Left Atrial Volume (2D + PV)
- The left atrial body volume is estimated using the area–length method, based on two perpendicular long-axis atrial areas and a single atrial length measurement.
- An additional volume contribution from the pulmonary veins (and the left atrial appendage) is included using a simplified cylindrical model.
- The pulmonary vein volume is estimated from 2D measurements of vein length and diameter and scaled to represent multiple venous structures.

#### True Left Atrial Volume (3D)
- The reference volume is obtained from full 3D segmentation of cardiac MRI data.
- The segmentation includes the left atrial body and connected venous structures.
- This volume is treated as the ground truth for comparison.

---
## 3. Methods Overview

This project compares multiple approaches for estimating left atrial (LA) volume from two-dimensional (2D) measurements and evaluates them against true three-dimensional (3D) volumes derived from MRI segmentation.

### 3.1 Estimated Left Atrial Volume (2D + PV)

- The left atrial body volume is initially estimated using the **area–length method**, which approximates the atrium as an ellipsoid.
- Two perpendicular long-axis atrial areas (A1 and A2) and a single atrial length (L) are measured from 2D image views.
- The atrial body volume is calculated using the standard area–length formula.
- An additional volume contribution from the pulmonary veins (PVs) and the left atrial appendage is included using a simplified cylindrical model.
- A single pulmonary vein is measured and scaled to represent five structures (four pulmonary veins plus the appendage), matching the anatomical definition used in the 3D segmentation.

### 3.2 True Left Atrial Volume (3D)

- The reference (“gold standard”) volume is obtained from full 3D segmentation of cardiac MRI data.
- The segmentation includes the left atrial body, pulmonary veins, and left atrial appendage.
- Volumes are computed directly from the segmented 3D data and used for comparison with all estimated methods.

### 3.3 Regression-Based Volume Estimation Models

In addition to the standard geometric area–length formula, two regression-based models are explored to investigate whether data-driven calibration can improve agreement with true 3D volumes.

**Regression Model 1: Calibrated Area–Length Model**

![Calibrated area-length model equation1](images/regression_1.svg)

This model retains the structure of the area–length method but allows the scaling (\(\alpha\)) and offset (\(\beta\)) to be learned directly from the data using linear regression.

**Regression Model 2: Multivariate Linear Model**

![Calibrated area-length model equation2](images/regression_2.svg)

This model treats the two atrial areas and the atrial length as independent predictors of volume and learns their relative contributions using multiple linear regression.

Both regression models are trained using measured 2D features and evaluated by comparing their predicted volumes with true 3D segmented volumes.

---

## 4. What the Python Scripts Do

This repository contains Python scripts that perform correlation and regression analysis to evaluate different left atrial volume estimation methods.

- `la_volume_correlation_analysis.py` 
- `la_volume_regression_analysis.py`

### 4.1 Volume Correlation Analysis Script

The original analysis script (`la_volume_correlation_analysis.py`):
- Takes paired measurements of estimated left atrial volume (2D + PV) and true 3D segmented volume
- Generates a scatter (correlation) plot comparing estimated and true volumes
- Fits a best-fit straight line using linear regression
- Computes the Pearson correlation coefficient (r)
- Visualises the strength and consistency of the relationship between estimated and true volumes

### 4.2 Regression Analysis and Visualisation Script

A second Python script (`la_volume_regression_analysis.py`) extends the analysis by fitting and visualising two regression-based volume estimation models.

This script:
- Reads atrial measurements (A1, A2, and L) and true 3D volumes from a CSV file
- Fits:
  - a calibrated area–length regression model  
  - a multivariate linear regression model
- Computes performance metrics including correlation coefficient (R), coefficient of determination (R²), mean absolute error (MAE), and root mean square error (RMSE)
- Generates correlation plots comparing predicted versus true 3D volumes for each model
- Overlays the identity line for reference
- Displays the fitted regression equations and coefficients directly on the plots
- Saves the resulting figures as high-resolution PNG files for use in reports and presentations

Together, these scripts allow direct comparison between geometric and data-driven approaches to left atrial volume estimation.


### 5. Technologies Used

- Python  
- NumPy  
- Matplotlib  
- SciPy  

---

### 6. Intended Use

This repository is intended for educational and research purposes only.

The script demonstrates:
- Basic validation of medical imaging measurements
- Comparison between simplified geometric models and reference 3D measurements
- Use of correlation and regression analysis in biomedical research

The code and results are not intended for clinical diagnosis or decision-making.

