# riskfree/curve/bootstrap_fast.py
import numpy as np

def bootstrap_fast(row):
    tenors = np.array([1,2,3,5,7,10,20,30], dtype=float)
    yields = row.to_numpy(dtype=float)  # already decimals
    z = -np.log(1.0 / (1.0 + yields * tenors)) / tenors
    return dict(zip(tenors, z))
