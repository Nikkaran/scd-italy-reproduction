import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

def load_and_filter(path):
    df = pd.read_parquet(path)
    df = df.rename(columns={'geo': 'province'})
    df = df[(df['year'] >= 2000) & (df['year'] <= 2022)].copy()
    
    # Find Modena's code dynamically
    modena_code = None
    for code in ['ITH54', 'ITD54']:
        if code in df['province'].values:
            modena_code = code
            break
            
    if modena_code is None:
        print(f"ERROR: Modena not found. Available codes: {df['province'].unique()}")
        return None, None
        
    print(f"Treating Modena: {modena_code}")
    
    # Exclusions
    drop_regions = df['province'].str.startswith(modena_code[:4]) & (df['province'] != modena_code)
    drop_zones = df['province'].isin(['ITC4B', 'ITH37'])
    drop_other_shocks = df['province'].isin(['ITF11', 'ITF43', 'ITC33', 'ITG12'])
    
    df_donors = df[~(drop_regions | drop_zones | drop_other_shocks)]
    
    treated_df = df[df['province'] == modena_code]
    donors_df = df_donors[df_donors['province'] != modena_code]
    
    valid_donors = sorted(donors_df['province'].unique())[:25]
    donors_df = donors_df[donors_df['province'].isin(valid_donors)]
    
    outcome = 'employment'
    y0 = treated_df.sort_values('year')[outcome].values
    Yd = donors_df.pivot(index='province', columns='year', values=outcome).values
    
    # Ensure data actually exists
    if len(y0) == 0 or Yd.shape[1] == 0:
        print("ERROR: Data extraction failed. Check if columns have NaNs.")
        return None, None
        
    return np.log(y0), np.log(Yd)

# (The fit_scd and placebo functions remain exactly the same)
def fit_scd(y0, Yd, T0_idx):
    d_y0 = y0 - y0[T0_idx]
    d_Yd = Yd - Yd[:, T0_idx:T0_idx+1]
    g = d_y0[:T0_idx]
    G = d_Yd[:, :T0_idx]
    H = G @ G.T
    h = G @ g
    def obj(w): return 0.5 * w @ H @ w - h @ w
    def grad(w): return H @ w - h
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0, 'jac': lambda w: np.ones_like(w)}]
    bounds = [(0.0, 1.0)] * Yd.shape[0]
    res = minimize(obj, np.full(Yd.shape[0], 1/Yd.shape[0]), jac=grad, bounds=bounds, constraints=cons, method='SLSQP', options={'ftol': 1e-10})
    w = res.x
    synth = w @ d_Yd
    theta = d_y0 - synth
    post_idx = np.arange(T0_idx + 1, len(y0))
    att = np.mean(theta[post_idx])
    return w, theta, att

def placebo_inference(y0, Yd, T0_idx):
    _, _, true_att = fit_scd(y0, Yd, T0_idx)
    null_atts = []
    for r in range(Yd.shape[0]):
        fake_y0 = Yd[r]
        fake_Yd = np.vstack([Yd[:r], Yd[r+1:]])
        _, _, att_r = fit_scd(fake_y0, fake_Yd, T0_idx)
        null_atts.append(abs(att_r))
    null_atts = np.array(null_atts)
    p_value = (1 + np.sum(null_atts >= abs(true_att))) / (1 + len(null_atts))
    return true_att, p_value

if __name__ == "__main__":
    print("Loading data and applying Donor Screen...")
    y0, Yd = load_and_filter("data/italy_nuts3_panel.parquet")
    
    if y0 is not None:
        T0_idx = 11 # 2011 (since 2000 is index 0)
        print(f"Donors: {Yd.shape[0]}. Years: 2000-2022.")
        print("Fitting SCD and running placebos...")
        
        w, theta, att = fit_scd(y0, Yd, T0_idx)
        true_att, p_val = placebo_inference(y0, Yd, T0_idx)
        
        print("\n--- RESULTS ---")
        print(f"ATT (log employment effect): {true_att:.4f}")
        print(f"Placebo p-value: {p_val:.4f}")
    else:
        print("Execution aborted due to missing data.")