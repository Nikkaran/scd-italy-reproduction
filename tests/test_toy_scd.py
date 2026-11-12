import numpy as np
from src.scd import fit_scd

def test_toy_scd_recovers_injected_shock():
    rng = np.random.default_rng(0)
    N, T0, T1 = 10, 15, 5
    Y = rng.normal(100.0, 5.0, (N + 1, T0 + T1))
    Y[0, T0:] += 10.0  # treated unit gets +10 shock

    out = fit_scd(Y[0], Y[1:], T0)

    assert np.isclose(out["weights"].sum(), 1.0, atol=1e-6)
    assert (out["weights"] >= -1e-8).all()
    assert np.isfinite(out["att"])
    assert out["att"] > 5.0
    
    print(f"\nATT = {out['att']:.3f} (injected 10), sum(w) = {out['weights'].sum():.6f}")