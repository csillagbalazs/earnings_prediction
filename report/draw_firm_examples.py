"""Reproduce Section 5's firm examples and economic-scale diagnostics.

Reads saved out-of-sample forecasts; does not refit or modify any model.
"""
from pathlib import Path
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "output" / "figures"
TABLES = ROOT / "output" / "tables"
MODELS = ["rw", "rwd", "foster", "xgb"]
LABELS = {"rw": "SRW", "rwd": "SRW + drift", "foster": "Foster", "xgb": "XGBoost"}
STYLES = {
    "rw": dict(color="#7b7b7b", linestyle="--", linewidth=1.15),
    "rwd": dict(color="#2a78d6", linestyle=":", linewidth=1.25),
    "foster": dict(color="#d97826", linestyle="-.", linewidth=1.15),
    "xgb": dict(color="#168568", linestyle="-", linewidth=1.55),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#aaaaaa"})

fc = pd.read_csv(ROOT / "output" / "forecasts" / "eps_forecasts_oos.csv")
common = fc.loc[fc.common].copy()
assert len(common) == 95732
assert not common.duplicated(["permno", "fyear", "fqtr"]).any()
assert common[[f"f_{m}" for m in MODELS] + ["eps_actual", "scale_hist"]].notna().all().all()
assert common.scale_hist.gt(0).all()
for m in MODELS:
    np.testing.assert_allclose(common[f"e_{m}"], common.eps_actual - common[f"f_{m}"], atol=1e-12)

# Compute the ranking directly, then check it against the report's saved table.
firm = common.groupby("permno", sort=True).agg(
    n=("eps_actual", "size"),
    MAE_rw=("e_rw", lambda x: x.abs().mean()),
    MAE_xgb=("e_xgb", lambda x: x.abs().mean()),
)
firm = firm.loc[firm.n.ge(12)].copy()
firm["dMAE_xgb"] = 100 * (1 - firm.MAE_xgb / firm.MAE_rw)
saved = pd.read_csv(TABLES / "firm_level_metrics.csv").set_index("permno")
assert set(firm.index) == set(saved.index)
np.testing.assert_allclose(firm.dMAE_xgb, saved.loc[firm.index, "dMAE_xgb"], atol=1e-9)
ranked = firm.reset_index().sort_values(["dMAE_xgb", "permno"], ascending=[False, True])
best = ranked.head(4)
worst = ranked.tail(4).sort_values(["dMAE_xgb", "permno"])
excluded = set(best.permno) | set(worst.permno)
random = ranked.loc[~ranked.permno.isin(excluded)].sort_values("permno").sample(4, random_state=42)
groups = {"best": best, "worst": worst, "random": random}
selections = pd.concat([g.assign(group=key, position=range(1, 5)) for key, g in groups.items()], ignore_index=True)
assert len(selections) == selections.permno.nunique() == 12
selections.to_csv(TABLES / "firm_forecast_examples.csv", index=False)


def short_name(name):
    name = name.split(" - UNADJUSTED PRICE")[0]
    name = re.sub(r"\s*'A'", "", name)
    return textwrap.fill(name, width=32)


for group, selected in groups.items():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    for ax, row in zip(axes.flat, selected.itertuples()):
        history = common.loc[common.permno.eq(row.permno)].copy()
        history["quarter_index"] = history.fyear * 4 + history.fqtr - 1
        history = history.set_index("quarter_index").sort_index()
        # Missing common-sample quarters are gaps, not interpolated forecasts.
        history = history.reindex(range(int(history.index.min()), int(history.index.max()) + 1))
        x = history.index.to_numpy() / 4
        for m in MODELS:
            ax.plot(x, history[f"f_{m}"], label=LABELS[m], **STYLES[m])
        ax.plot(x, history.eps_actual, color="#171717", linewidth=1.5,
                marker="o", markersize=2.6, label="Actual EPS", zorder=5)
        ax.axhline(0, color="#c7c7c7", linewidth=0.6, zorder=0)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.set_xlim(x.min() - 0.12, x.max() + 0.12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Fiscal year", fontsize=8)
        ax.set_title(short_name(row.permno), fontsize=9, weight="bold", pad=32)
        ax.text(0, 1.035, f"N = {row.n}  |  XGB MAE gain vs SRW: {row.dMAE_xgb:+.1f}%",
                transform=ax.transAxes, fontsize=7.7, color="#52514e")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    order = [4, 0, 1, 2, 3]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5,
               frameon=False, fontsize=8.5, handlelength=2.4, columnspacing=1.2)
    fig.supylabel("EPS (CNY/share)", x=0.015, fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.08, top=0.80, hspace=0.90, wspace=0.35)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_firm_forecasts_{group}.{ext}", dpi=200)
    plt.close(fig)

# All figures below use the untrimmed common sample, not the selected firms.
context = []
for m in MODELS:
    error = common[f"e_{m}"].abs()
    scaled = error / common.scale_hist
    context.append({
        "model": m, "N": len(common), "MAE_CNY_per_share": error.mean(),
        "aggregate_abs_error_pct_abs_actual": 100 * error.sum() / common.eps_actual.abs().sum(),
        "mean_error_pct_historical_scale": 100 * scaled.mean(),
        "median_error_pct_historical_scale": 100 * scaled.median(),
    })
context = pd.DataFrame(context).set_index("model")
context.to_csv(TABLES / "economic_error_context.csv")
rw, xgb = context.loc["rw"], context.loc["xgb"]
paragraph = rf"""\paragraph{{Error size in context.}} On the untrimmed common sample, XGBoost's MAE is
{xgb.MAE_CNY_per_share:.4f} CNY per share, against {rw.MAE_CNY_per_share:.4f} for the SRW. Total absolute error divided by
total absolute realized EPS is {xgb.aggregate_abs_error_pct_abs_actual:.1f}\% for XGBoost and
{rw.aggregate_abs_error_pct_abs_actual:.1f}\% for the SRW. This is an aggregate ratio, not the average observation's
percentage error. Using each observation's historical EPS scale instead, the mean absolute scaled error is
{xgb.mean_error_pct_historical_scale:.1f}\% versus {rw.mean_error_pct_historical_scale:.1f}\%; the median is
{xgb.median_error_pct_historical_scale:.1f}\% versus {rw.median_error_pct_historical_scale:.1f}\%.
The historical scale is the mean absolute EPS known before the announcement within the preceding eight fiscal
quarters, floored at 0.01 CNY per share. Direct percentage errors relative to actual EPS are unsuitable here:
{common.eps_actual.eq(0).sum():,} observations have zero EPS, and near-zero EPS makes such ratios unstable.
Thus XGBoost improves materially on the benchmark, but the remaining errors are large relative to earnings.
Forecast accuracy alone does not establish an improvement in investment returns or in the measurement of market expectations.
"""
(TABLES / "economic_error_context.tex").write_text(paragraph)
print(selections[["group", "permno", "n", "dMAE_xgb"]].to_string(index=False))
print(context.round(3).to_string())
