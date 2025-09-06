#!/usr/bin/env python3
"""
evm_monte_carlo.py
Generate synthetic project-control data using Monte Carlo simulation + EVM metrics.
Outputs train/test/val CSVs in ./data and prints a quick run summary.

Usage:
    python evm_monte_carlo.py --n 5000 --seed 42
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def simulate_project_snapshot(project_id:int, period:int, rng:np.random.Generator):
    """
    Simulate a single project snapshot for a given time period.
    We generate Planned Value (PV), Earned Value (EV), and Actual Cost (AC),
    then derive CPI, SPI, schedule/cost forecasts, and late/overrun flags.
    """
    # Project baseline scale (randomized per project to add heterogeneity)
    baseline_budget = rng.normal(1_000_000, 150_000)  # mean $1M, sd 150k
    baseline_duration = max(6, int(rng.normal(18, 4)))  # months, min 6
    
    # Progress as a fraction of the baseline duration
    # period in [1..baseline_duration]; clip in case of stochastic draws
    t = min(max(1, period), baseline_duration)
    planned_progress = t / baseline_duration
    
    # Planned Value ~ fraction of baseline budget with some planning noise
    PV = planned_progress * baseline_budget * rng.normal(1.0, 0.03)
    
    # Random "risk state" that influences EV and AC (schedule/cost uncertainty)
    # 0: favorable, 1: neutral, 2: unfavorable
    risk_state = rng.choice([0,1,2], p=[0.25, 0.5, 0.25])
    if risk_state == 0:
        ev_mu, ev_sd = 1.03, 0.04
        ac_mu, ac_sd = 0.98, 0.04
    elif risk_state == 1:
        ev_mu, ev_sd = 1.00, 0.05
        ac_mu, ac_sd = 1.00, 0.05
    else:
        ev_mu, ev_sd = 0.95, 0.06
        ac_mu, ac_sd = 1.05, 0.06
    
    # Earned Value is PV adjusted by performance noise (can lag or lead)
    EV = max(0.0, PV * rng.normal(ev_mu, ev_sd))
    
    # Actual Cost deviates due to cost risk
    AC = max(1.0, PV * rng.normal(ac_mu, ac_sd))
    
    # EVM indices
    CPI = EV / AC if AC > 0 else np.nan
    SPI = EV / PV if PV > 0 else np.nan
    
    # Simple schedule forecast via SPI rule-of-thumb
    # Forecast Duration (FD) = Baseline / SPI (bounded)
    FD = baseline_duration / np.clip(SPI if SPI>0 else np.nan, 0.5, 2.0)
    
    # Simple cost forecast via CPI rule-of-thumb
    # EAC (Estimate At Completion) = BaselineBudget / CPI (bounded)
    EAC = baseline_budget / np.clip(CPI if CPI>0 else np.nan, 0.5, 2.0)
    
    # Late & overrun flags relative to baseline
    late_flag = 1 if FD > baseline_duration * 1.05 else 0
    overrun_flag = 1 if EAC > baseline_budget * 1.05 else 0
    
    return {
        "project_id": project_id,
        "period": t,
        "baseline_budget": baseline_budget,
        "baseline_duration_months": baseline_duration,
        "PV": PV,
        "EV": EV,
        "AC": AC,
        "CPI": CPI,
        "SPI": SPI,
        "forecast_duration_months": FD,
        "forecast_EAC": EAC,
        "late_flag": late_flag,
        "overrun_flag": overrun_flag,
        "risk_state": risk_state,
    }

def generate_dataset(n:int, seed:int=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    
    # We will simulate multiple snapshots per project to mimic monthly control points
    # Choose projects count so that ~10 snapshots per project on average
    avg_snapshots = 10
    n_projects = max(10, n // avg_snapshots)
    
    rows = []
    for pid in range(1, n_projects+1):
        # Random baseline duration per project
        baseline_duration = max(6, int(rng.normal(18, 4)))
        # Number of snapshots for the project (1..baseline_duration)
        k = rng.integers(low=max(4, baseline_duration//3), high=baseline_duration+1)
        sample_periods = rng.choice(np.arange(1, baseline_duration+1), size=k, replace=False)
        for p in sample_periods:
            rows.append(simulate_project_snapshot(pid, int(p), rng))
    
    df = pd.DataFrame(rows).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="Target number of snapshots (approximate).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--outdir", type=str, default="data", help="Output directory for CSVs.")
    args = ap.parse_args()
    
    df = generate_dataset(n=args.n, seed=args.seed)
    
    # Basic cleaning: drop inf/nan if any arise
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    
    # Train/Test/Val split: 70/20/10
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=args.seed, shuffle=True)
    test_df, val_df = train_test_split(temp_df, test_size=0.3334, random_state=args.seed, shuffle=True)  # ~20/10
    
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    train_path = os.path.join(outdir, "train.csv")
    test_path = os.path.join(outdir, "test.csv")
    val_path = os.path.join(outdir, "val.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    # Quick summary print (useful for your screen recording)
    print("=== Monte Carlo + EVM Synthetic Data Generation ===")
    print(f"Seed: {args.seed}")
    print(f"Total rows: {len(df)} | Train: {len(train_df)} | Test: {len(test_df)} | Val: {len(val_df)}")
    print(f"Projects: {df['project_id'].nunique()} | Periods (min..max): {df['period'].min()}..{df['period'].max()}")
    print(f"CPI mean: {df['CPI'].mean():.3f} | SPI mean: {df['SPI'].mean():.3f}")
    print(f"Late %: {100*df['late_flag'].mean():.1f}% | Overrun %: {100*df['overrun_flag'].mean():.1f}%")
    print(f"Wrote: {train_path}, {test_path}, {val_path}")
    print("\nColumns:", ", ".join(df.columns))

if __name__ == "__main__":
    main()
