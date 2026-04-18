# Mean-Reverting Pairs Trading Strategy

A quantitative finance project that implements a **statistical arbitrage pairs trading strategy** based on mean reversion. The project covers the full pipeline — from asset selection and stationarity/cointegration testing, to signal generation and backtesting — applied primarily to commodity ETF pairs (e.g., GLD/SLV).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Dependencies](#dependencies)
- [Disclaimer](#disclaimer)

---

## Overview

Pairs trading exploits the tendency of two historically correlated assets to revert to a long-run equilibrium when their price relationship temporarily diverges. This project:

1. Tests candidate asset pairs for **stationarity** (ADF test) and **cointegration** (Johansen test)
2. Constructs a **spread** using optimal hedge weights derived from the cointegrating vector
3. Generates **entry/exit signals** using Bollinger Bands applied to the spread's Z-score
4. Runs a **backtest** to evaluate strategy performance over historical data

---

## Project Structure

```
├── simulation.py            # Basic ratio and Z-score simulation for a pair (GLD/SLV)
├── stationarity_test.py     # ADF test, Johansen cointegration test, rolling hedge weights,
│                            # Bollinger Band signal generation, and backtesting logic
├── commodities.py           # Asset data fetching and preprocessing for commodity pairs
├── cointegration_check.ipynb  # Notebook: exploratory cointegration analysis
├── backtest.ipynb           # Notebook: full backtest with performance visualization
├── AAPL_historical_data.csv # Historical price data for AAPL (used in testing)
├── commodities.csv          # Historical price data for commodity ETFs
└── README.md
```

---

## Methodology

### 1. Pair Selection & Stationarity Testing
- Candidate pairs are checked for stationarity using the **Augmented Dickey-Fuller (ADF)** test
- Only non-stationary individual series that are stationary when combined are eligible for the strategy

### 2. Cointegration Testing (Johansen Test)
- The **Johansen trace test** determines whether a cointegrating relationship exists between two assets
- Optimal lag order is selected via **AIC** using a VAR model
- The cointegrating vector (hedge weights) is extracted from the first eigenvector

### 3. Spread Construction
- A **rolling hedge weight** is calculated using a lookback window (default: 20 days) to produce time-varying hedge ratios
- The spread is computed as a weighted linear combination of the two asset prices

### 4. Signal Generation (Bollinger Bands)
- **Bollinger Bands** are applied to the spread (or price ratio):
  - **Long signal**: spread crosses below the lower band → buy asset 1, short asset 2
  - **Short signal**: spread crosses above the upper band → short asset 1, buy asset 2
  - **Exit signal**: spread reverts to the mean
- Alternatively, a simple **price ratio + Z-score** approach is implemented in `simulation.py`

### 5. Backtesting
- Trades are executed with a fixed trade size (default: $10,000) against an initial capital of $100,000
- Portfolio value is tracked daily; cumulative returns are plotted

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ianliu1119/Mean-Reverting-Pairs-Trading-Strategy.git
   cd Mean-Reverting-Pairs-Trading-Strategy
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually:
   ```bash
   pip install yfinance pandas numpy statsmodels matplotlib plotly python-dateutil
   ```

---

## Usage

### Run the basic simulation (ratio + Z-score)
```bash
python simulation.py
```
Outputs a two-panel plot of the GLD/SLV price ratio and its Z-score with ±1 standard deviation bands.

### Run the full stationarity & backtest pipeline
```bash
python stationarity_test.py
```
This runs the Johansen test, computes rolling hedge weights, generates Bollinger Band signals, backtests the strategy, and plots results.

### Run commodity pair analysis
```bash
python commodities.py
```

### Explore the notebooks
Open `cointegration_check.ipynb` for an interactive walkthrough of the cointegration analysis, or `backtest.ipynb` for the full backtesting workflow with visualizations.

```bash
jupyter notebook
```

---

## Data

- Price data is fetched automatically via **yfinance** for the specified date range
- Default pair: **GLD** (SPDR Gold Shares) and **SLV** (iShares Silver Trust)
- Date range used in analysis: `2023-10-01` to `2024-10-01` (training), with a 3-year forward window for out-of-sample testing
- Static CSV files (`AAPL_historical_data.csv`, `commodities.csv`) are included for offline use and reproducibility

To change the asset pair or date range, modify the following variables at the top of `stationarity_test.py` or `simulation.py`:
```python
tickers = ['GLD', 'SLV']   # Replace with any two cointegrated tickers
start_date = '2023-10-01'
end_date = '2024-10-01'
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Fetching historical price data |
| `pandas` / `numpy` | Data manipulation |
| `statsmodels` | ADF test, Johansen test, VAR model |
| `matplotlib` | Static plotting |
| `plotly` | Interactive Bollinger Band visualization |
| `python-dateutil` | Date arithmetic for rolling windows |

---

## Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice. Past performance of a backtested strategy does not guarantee future results.
