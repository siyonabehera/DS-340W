# Monte Carlo + EVM Synthetic Project-Control Dataset

This repo contains a **ready-to-run** Python script that generates a synthetic dataset for **Earned Value Management (EVM)** with **Monte Carlo** uncertainty.

It produces:
- `data/train.csv` (70%)
- `data/test.csv` (20%)
- `data/val.csv` (10%)

and prints a short summary for your **screen recording** submission.

## How to Run

```bash
python evm_monte_carlo.py --n 5000 --seed 42
```

- Increase `--n` for more snapshots (e.g., 20,000).
- Change `--seed` for different random draws.
- Use `--outdir` to write CSVs elsewhere (default is `data`).

## What It Simulates

For each project snapshot (a monthly control point), we generate:
- **PV** (Planned Value), **EV** (Earned Value), **AC** (Actual Cost)
- **CPI** = EV / AC, **SPI** = EV / PV
- Simple **schedule forecast**: Forecast Duration = Baseline Duration / SPI (bounded)
- Simple **cost forecast**: EAC = Baseline Budget / CPI (bounded)
- Flags: `late_flag` (forecast duration > 105% of baseline), `overrun_flag` (EAC > 105% of baseline)

The simulation includes a latent **risk_state** (favorable / neutral / unfavorable) that shifts EV and AC noise to mimic risk impacts.

## Files

- `evm_monte_carlo.py` – main script
- `data/` – output CSVs (created on first run)
- `requirements.txt` – minimal dependencies

## Suggested Screen Recording Flow

1. Open your public GitHub repo (after you upload these files).
2. `python evm_monte_carlo.py --n 5000 --seed 42`
3. Show the terminal summary.
4. Open `data/train.csv` to show columns.
5. Briefly explain CPI/SPI and flags.

## Academic Notes

- This synthetic dataset aligns with parent papers that integrate **EVM** and **Monte Carlo** for project schedule/cost risk forecasting.
- You can justify synthetic data usage by citing that many studies use **simulated control points** when real EVM logs are proprietary.

## License

MIT
