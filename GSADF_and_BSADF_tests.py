
"""
GSADF Bubble Detection — Phillips, Shi & Yu (2015)
Run: python gsadf_bubble_detection.py

Resamples daily data to weekly to keep runtime manageable (~5-10 min).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ── 0. Load & resample to weekly ──────────────────────────────────────────────
df = pd.read_csv("data/merged_data.csv")
df["date"] = pd.to_datetime(df["date"], dayfirst=True)
df = df.sort_values("date").set_index("date")

weekly = df[["btc_adj_close", "eth_adj_close", "sp500_adj_close"]].resample("W").last()
weekly = np.log(weekly)
weekly.columns = ["BTC", "ETH", "S&P500"]

# ── 1. ADF t-stat (fast numpy OLS, no statsmodels overhead) ──────────────────
def adf_tstat(y):
    """Right-tailed ADF: Δy = α + β·y_{t-1} + ε  →  t-stat on β"""
    n = len(y)
    if n < 6:
        return np.nan
    dy = np.diff(y)
    xl = y[:-1]
    m  = n - 1
    Sx, Sx2, Sy, Sxy = xl.sum(), (xl**2).sum(), dy.sum(), (xl * dy).sum()
    denom = m * Sx2 - Sx**2
    if abs(denom) < 1e-12:
        return np.nan
    b    = (m * Sxy - Sx * Sy) / denom
    a    = (Sy - b * Sx) / m
    e    = dy - (a + b * xl)
    s2   = (e @ e) / max(m - 2, 1)
    vb   = s2 * m / denom
    return float(b / np.sqrt(vb)) if vb > 0 else np.nan

# ── 2. BSADF sequence ─────────────────────────────────────────────────────────
def bsadf_sequence(y, r0):
    """BSADF(r2) = sup_{r1} ADF(y[r1:r2]),  r2 = 2r0…T"""
    T   = len(y)
    seq = np.full(T - 2 * r0 + 1, np.nan)
    for i, r2 in enumerate(range(2 * r0, T + 1)):
        best = -np.inf
        for r1 in range(0, r2 - r0 + 1):
            t = adf_tstat(y[r1:r2])
            if not np.isnan(t) and t > best:
                best = t
        seq[i] = best if best > -np.inf else np.nan
    return seq

# ── 3. GSADF scalar ───────────────────────────────────────────────────────────
def gsadf(y, r0):
    seq = bsadf_sequence(y, r0)
    v   = seq[~np.isnan(seq)]
    return float(np.max(v)) if len(v) else np.nan

# ── 4. Bootstrap GSADF critical values ───────────────────────────────────────
def bootstrap_gsadf_cv(T, r0, n_boot=299, quantiles=(0.90, 0.95, 0.99)):
    vals = [gsadf(np.cumsum(np.random.standard_normal(T)), r0)
            for _ in tqdm(range(n_boot), desc="  GSADF bootstrap", leave=False)]
    arr = np.array([v for v in vals if not np.isnan(v)])
    return {q: float(np.quantile(arr, q)) for q in quantiles}

# ── 5. Bootstrap time-varying CVs for date-stamping ──────────────────────────
def bootstrap_tv_cv(T, r0, n_boot=299, quantiles=(0.90, 0.95, 0.99)):
    n_steps  = T - 2 * r0 + 1
    boot_mat = np.full((n_boot, n_steps), np.nan)
    for b in tqdm(range(n_boot), desc="  TV-CV bootstrap", leave=False):
        boot_mat[b] = bsadf_sequence(np.cumsum(np.random.standard_normal(T)), r0)
    return {q: np.nanquantile(boot_mat, q, axis=0) for q in quantiles}

# ── 6. Date-stamp bubble episodes ─────────────────────────────────────────────
def date_stamp(bsadf_vals, cv_95, min_dur=5):
    exceed = bsadf_vals > cv_95
    bubbles, start = [], None
    for i, flag in enumerate(exceed):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_dur:
                bubbles.append((start, i - 1))
            start = None
    if start is not None and len(exceed) - start >= min_dur:
        bubbles.append((start, len(exceed) - 1))
    return bubbles

# ── 7. Run ────────────────────────────────────────────────────────────────────

np.random.seed(42)
r0_frac = 0.10
N_BOOT  = 299
MIN_DUR = 5
results = {}

for asset in weekly.columns:
    print(f"\n{'='*50}\n  {asset}\n{'='*50}")
    series = weekly[asset].dropna()
    y, dates = series.values, series.index
    T  = len(y)
    r0 = max(int(r0_frac * T), 15)
    print(f"  Weekly obs={T}   r0={r0} weeks")

    # GSADF test
    print("  GSADF...", end=" ", flush=True)
    g_stat = gsadf(y, r0)
    print(f"{g_stat:.4f}")

    g_cv = bootstrap_gsadf_cv(T, r0, n_boot=N_BOOT)
    for q in sorted(g_cv):
        tag = " <--" if g_stat > g_cv[q] else ""
        print(f"    CV {int(q*100)}%: {g_cv[q]:.4f}{tag}")

    # BSADF + date-stamping
    print("  BSADF sequence...", end=" ", flush=True)
    bsadf_seq = bsadf_sequence(y, r0)
    print("done")

    tv_cv = bootstrap_tv_cv(T, r0, n_boot=N_BOOT)
    bsadf_dates = dates[2 * r0 - 1:]
    bubbles_idx = date_stamp(bsadf_seq, tv_cv[0.95], min_dur=MIN_DUR)

    bubble_periods = []
    print(f"  Bubble periods (95% CV, min {MIN_DUR} weeks):")
    for s, e in bubbles_idx:
        ds, de = bsadf_dates[s].date(), bsadf_dates[e].date()
        print(f"    {ds}  →  {de}  ({e - s + 1} wks)")
        bubble_periods.append((ds, de))
    if not bubbles_idx:
        print("    None detected.")

    results[asset] = dict(
        dates=dates, prices=np.exp(y),
        bsadf=bsadf_seq, bsadf_dates=bsadf_dates,
        tv_cv=tv_cv, bubbles=bubble_periods,
        gsadf_stat=g_stat, gsadf_cv=g_cv,
    )

# ── 8. Plot ───────────────────────────────────────────────────────────────────
n = len(results)
fig, axes = plt.subplots(n, 2, figsize=(18, 5.5 * n))
fig.suptitle("GSADF Bubble Detection — Phillips, Shi & Yu (2015) [Weekly]",
             fontsize=14, fontweight="bold")

for row, (asset, res) in enumerate(results.items()):
    ax_p, ax_b = axes[row, 0], axes[row, 1]
    dates_pd = pd.to_datetime(res["dates"])
    bd       = pd.to_datetime(res["bsadf_dates"])

    ax_p.plot(dates_pd, res["prices"], color="#1f77b4", lw=1.0)
    ax_b.plot(bd, res["bsadf"],         color="#ff7f0e", lw=1.0, label="BSADF")
    ax_b.plot(bd, res["tv_cv"][0.95],   color="#d62728", lw=1.0, ls="--", label="CV 95%")
    ax_b.plot(bd, res["tv_cv"][0.90],   color="dimgray", lw=0.7, ls=":",  label="CV 90%")
    ax_b.axhline(0, color="black", lw=0.5, alpha=0.4)

    for bs, be in res["bubbles"]:
        ax_p.axvspan(pd.Timestamp(bs), pd.Timestamp(be), alpha=0.22, color="#d62728", label="Bubble")
        ax_b.axvspan(pd.Timestamp(bs), pd.Timestamp(be), alpha=0.15, color="#d62728")

    for ax in (ax_p, ax_b):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.25)

    ax_p.set_title(f"{asset} — Weekly Price", fontweight="bold")
    ax_p.set_ylabel("Price (USD)")
    handles, labels = ax_p.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax_p.legend(by_label.values(), by_label.keys(), fontsize=8)

    ax_b.set_title(f"{asset} — BSADF", fontweight="bold")
    ax_b.set_ylabel("t-stat")
    ax_b.legend(fontsize=8, loc="upper left")

    g, c95 = res["gsadf_stat"], res["gsadf_cv"][0.95]
    sig = "significant at 5%" if g > c95 else "not significant"
    ax_b.set_xlabel(f"GSADF={g:.3f}  CV95%={c95:.3f}  [{sig}]", fontsize=8.5, color="dimgray")

plt.tight_layout()
plt.savefig("gsadf_bubble_detection.png", dpi=150, bbox_inches="tight")
print("\nPlot saved → gsadf_bubble_detection.png")

# ── 9. Summary table ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  GSADF TEST SUMMARY")
print("="*60)
print(f"{'Asset':<8} {'GSADF':>8} {'CV90':>8} {'CV95':>8} {'CV99':>8}  Sig")
print("-"*60)
for asset, res in results.items():
    g, c = res["gsadf_stat"], res["gsadf_cv"]
    sig  = "***" if g > c[0.99] else "**" if g > c[0.95] else "*" if g > c[0.90] else "n.s."
    print(f"{asset:<8} {g:>8.3f} {c[0.90]:>8.3f} {c[0.95]:>8.3f} {c[0.99]:>8.3f}   {sig}")
print("\n*** p<0.01  ** p<0.05  * p<0.10  n.s. not significant")

import pickle

with open("gsadf_results.pkl", "wb") as f:
    pickle.dump(results, f)
