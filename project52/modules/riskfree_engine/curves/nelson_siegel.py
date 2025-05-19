# File: curves/nelson_siegel.py
import numpy as np
from scipy.optimize import curve_fit


def nelson_siegel(maturities, beta0, beta1, beta2, tau):
    """Nelson-Siegel functional form."""
    t = np.array(maturities)
    return beta0 + beta1 * ((1 - np.exp(-t/tau))/(t/tau)) + beta2 * (((1 - np.exp(-t/tau))/(t/tau)) - np.exp(-t/tau))


def fit_nelson_siegel(yields_df, initial_params=None):
    """Fit Nelson-Siegel model to yields."""
    maturities = yields_df['maturity']
    yields = yields_df['yield']
    popt, _ = curve_fit(nelson_siegel, maturities, yields, p0=initial_params or [0,0,0,1])
    return popt