import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Data provided
estimated = np.array([208701,218971,218452,119870,166884,158404,253225,331570,594636,151803])
true = np.array([246900,193830,136762,139128,190241,161650,280731,361897,499068,139551])

# Linear regression (best-fit line)
coeffs = np.polyfit(estimated, true, 1)
fit_line = np.poly1d(coeffs)

# Correlation coefficient
r, p = pearsonr(estimated, true)

# Plot
plt.figure(figsize=(7,7))
plt.scatter(estimated, true, color='blue', label='Data points')
plt.plot(estimated, fit_line(estimated), color='red', label='Best-fit line')
plt.xlabel('Estimated LA Volume (2D + PV)')
plt.ylabel('True LA Volume (3D Segmentation)')
plt.title(f'Correlation of Estimated vs True LA Volume\nPearson r = {r:.2f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("la_volume_correlation.png", dpi=300)

coeffs, r