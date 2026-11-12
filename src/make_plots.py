import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from run_scd import load_and_filter, fit_scd

def plot_results(y0, Yd, T0_idx):
    # 1. Fit main model
    w, theta, att = fit_scd(y0, Yd, T0_idx)
    
    # 2. Run placebos to get null paths
    null_paths = []
    for r in range(Yd.shape[0]):
        fake_y0 = Yd[r]
        fake_Yd = np.vstack([Yd[:r], Yd[r+1:]])
        _, theta_r, _ = fit_scd(fake_y0, fake_Yd, T0_idx)
        null_paths.append(theta_r)
    null_paths = np.array(null_paths)
    
    years = np.arange(2000, 2023)
    
    # --- PLOT 1: Event Study (Effect Path) ---
    plt.figure(figsize=(10, 6))
    plt.plot(years, theta, color='black', linewidth=2, label='Modena (Treated)')
    
    # Plot placebos in light gray
    for path in null_paths:
        plt.plot(years, path, color='gray', alpha=0.3, linewidth=0.8)
        
    plt.axvline(2012, color='red', linestyle='--', label='Earthquake (2012)')
    plt.axhline(0, color='black', linestyle='-', linewidth=1)
    plt.title('Synthetic Control with Differencing: Modena Earthquake Effect')
    plt.xlabel('Year')
    plt.ylabel('Effect on Log Employment (Theta)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/event_study.png', dpi=300)
    print("Saved plot to data/event_study.png")
    
    # --- PLOT 2: Donor Weights ---
    top_idx = np.argsort(w)[-10:] # Top 10 donors
    top_w = w[top_idx]
    
    plt.figure(figsize=(8, 5))
    plt.barh(range(len(top_w)), top_w, color='steelblue')
    plt.yticks(range(len(top_w)), [f"Donor {i}" for i in top_idx])
    plt.title('Top 10 Synthetic Control Donor Weights')
    plt.xlabel('Weight')
    plt.tight_layout()
    plt.savefig('data/weights.png', dpi=300)
    print("Saved plot to data/weights.png")

if __name__ == "__main__":
    print("Generating plots...")
    y0, Yd = load_and_filter("data/italy_nuts3_panel.parquet")
    if y0 is not None:
        plot_results(y0, Yd, T0_idx=11) # 2011