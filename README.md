# DSE4211 Project: Volatility Spillovers and Speculative Bubbles

This repository contains the code and outputs for our DSE4211 research project 
examining volatility spillovers from cryptocurrency markets to U.S. equity and 
bond markets, with a focus on the role of speculative bubble periods.

## Project Overview

We investigate whether volatility transmission between crypto and traditional 
financial markets changes during bubble episodes, using a combination of bubble 
detection, GARCH volatility modelling, and VAR-based spillover analysis.

## Repository Structure
```
data/                   Raw and merged datasets
notebooks/              Jupyter notebooks for analysis
src/                    Supporting source files
output/                 Generated figures and results
GSADF_and_BSADF_tests.py          Bubble detection (PSY 2015 test)
Volatility Modelling - GARCH.ipynb GARCH family model estimation
VAR Analysis.ipynb                 VAR and Granger causality analysis
Realized_Volatility_XGBoost.ipynb  XGBoost volatility forecasting
garch_functions.py                 GARCH helper functions
```

## Methodology

1. **Bubble Detection** — GSADF test (Phillips, Shi & Yu, 2015) applied to 
   weekly log-prices of BTC, ETH, and S&P 500
2. **Volatility Modelling** — GARCH family models (GARCH, EGARCH, GJR-GARCH) 
   to estimate conditional volatility
3. **Spillover Analysis** — VAR models and Granger causality tests to identify 
   volatility transmission across markets
4. **Forecasting** — XGBoost model incorporating realized volatility and bubble 
   regime indicators

## Requirements
```bash
pip install numpy pandas matplotlib statsmodels arch xgboost tqdm jupyter
```

## Contributors

5 contributors — NUS DSE4211 Group 7
