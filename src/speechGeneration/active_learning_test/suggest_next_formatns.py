import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from scipy.stats import norm

# -------------------------------------------------
# 1. DATA LOADING & LOSS COMPUTATION
# -------------------------------------------------
def load_data(path: str = "Fusion Scores.xlsx", sheet: str = "Vowel Type") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna(subset=["F1_1", "F1_2", "F2", "Fusion (1-10)", "V1_Clarity(1-5)", "V2_Clarity(1-5)"])
    df["loss"] = -(df["Fusion (1-10)"] * df[["V1_Clarity(1-5)", "V2_Clarity(1-5)"]].min(axis=1))
    return df

# -------------------------------------------------
# 2. GAUSSIAN-PROCESS MODEL
# -------------------------------------------------
def fit_gp(X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
    kernel = (
        C(1.0, (1e-2, 1e2))
        * Matern(length_scale=[50, 50, 50], length_scale_bounds=(1, 500), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
    )
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
    gp.fit(X, y)
    return gp

# -------------------------------------------------
# 3. ACQUISITION: EXPECTED IMPROVEMENT
# -------------------------------------------------
def expected_improvement(X_cand: np.ndarray, gp: GaussianProcessRegressor, y_best: float, xi: float = 0.01) -> np.ndarray:
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    imp = y_best - mu - xi
    Z = imp / sigma
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma == 0.0] = 0.0
    return ei

# -------------------------------------------------
# 4. SUGGESTION FUNCTION
# -------------------------------------------------
BOUNDS = [(440, 580), (590, 700), (1700, 2000)]  # (F1_1, F1_2, F2)

def suggest_next_point(gp: GaussianProcessRegressor, bounds=BOUNDS, n_candidates: int = 5000000, strategy: str = "ei"):
    samples = np.column_stack([np.random.uniform(low, high, n_candidates) for (low, high) in bounds])
    if strategy == "var":
        _, sigma = gp.predict(samples, return_std=True)
        idx = np.argmax(sigma)
    else:
        y_best = gp.y_train_.min()
        ei = expected_improvement(samples, gp, y_best)
        idx = np.argmax(ei)
    return samples[idx]

# -------------------------------------------------
# 5. MAIN
# -------------------------------------------------
df = load_data()
X = df[["F1_1", "F1_2", "F2"]].values
y = df["loss"].values

gp = fit_gp(X, y)
next_ei = suggest_next_point(gp, strategy="ei")
next_var = suggest_next_point(gp, strategy="var")

next_ei_rounded = np.round(next_ei, 2)
next_var_rounded = np.round(next_var, 2)

print(next_ei_rounded, next_var_rounded)