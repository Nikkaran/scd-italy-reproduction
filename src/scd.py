"""
Synthetic Control with Differencing (SCD) — core estimator.
"""
import numpy as np
from scipy.optimize import minimize

def _weights(treated_pre: np.ndarray, donors_pre: np.ndarray) -> np.ndarray:
    n = donors_pre.shape[0]

    def obj(w):
        resid = treated_pre - w @ donors_pre
        return resid @ resid

    def grad(w):
        resid = treated_pre - w @ donors_pre
        return -2.0 * donors_pre @ resid

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones_like(w)}]
    bounds = [(0.0, 1.0)] * n
    x0 = np.full(n, 1.0 / n)

    res = minimize(obj, x0, jac=grad, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"ftol": 1e-9, "maxiter": 1000})
    
    if not res.success and res.nit >= 1000:
        raise RuntimeError(f"SLSQP failed completely: {res.message}")
        
    return res.x

def fit_scd(Y_treated: np.ndarray, Y_donors: np.ndarray, T0: int):
    Y_treated = np.asarray(Y_treated, float)
    Y_donors = np.asarray(Y_donors, float)
    T = Y_treated.shape[0]
    assert Y_donors.shape[1] == T and 1 <= T0 < T

    base_t = Y_treated[T0 - 1]
    base_d = Y_donors[:, T0 - 1:T0]
    d_treated = Y_treated - base_t
    d_donors = Y_donors - base_d

    w = _weights(d_treated[:T0], d_donors[:, :T0])

    synth_post = w @ d_donors[:, T0:]
    tau_post = d_treated[T0:] - synth_post
    att = float(np.mean(tau_post))

    return {"weights": w, "effect_path": tau_post, "att": att}