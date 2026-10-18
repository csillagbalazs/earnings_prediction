"""Rebuild selected forecast batches and explain them with native TreeSHAP.

Use the eps-forecast environment. Reuses notebook reconstruction/functions and
saved tuning choices; never reruns tuning or overwrites the original forecasts.
"""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "shap_selected"
OUT.mkdir(exist_ok=True)
notebook = json.loads((ROOT / "notebooks" / "eps_forecasting.ipynb").read_text())
scope = dict(np=np, pd=pd, warnings=warnings, xgb=xgb, DATA=ROOT / "data" / "clean",
             FIRST_FY=2005, LAST_FY=2021, FIRST_INFO_FY=2001, MAX_TARGET_LAG=365,
             SCALE_FLOOR=0.01, WINDOW=12, SEED=42, XGB_THREADS=8,
             display=lambda *args: None)

# Reconstruction cells can write a data-quality CSV: send that to a temporary
# directory, preserving every original artifact. The functions are reused from
# the notebook so this audit has the same feature definitions as the report.
with tempfile.TemporaryDirectory(prefix="eps-shap-") as tmp:
    scope["TAB_DIR"] = Path(tmp)
    with contextlib.redirect_stdout(io.StringIO()):
        for index in [4, 8, 16]:
            exec(compile("".join(notebook["cells"][index]["source"]),
                         f"notebook_cell_{index}", "exec"), scope)
    tree = ast.parse("".join(notebook["cells"][20]["source"]))
    names = {"build_features", "make_feature_table", "FEATURES", "XGB_FIXED"}
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in names)
             or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names
                                                     for t in node.targets))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook_features", "exec"), scope)

E, D = scope["E"], scope["D"]
meta, X = scope["make_feature_table"](E, D, scope["REC_FI"], scope["REC_CI"])
features = scope["FEATURES"]
row_lookup = pd.Series(np.arange(len(meta)), index=meta.fi * scope["N_T"] + meta.ci)
fc = pd.read_csv(ROOT / "output" / "forecasts" / "eps_forecasts_oos.csv")
tuning = pd.read_csv(ROOT / "output" / "tables" / "xgb_tuning_by_year.csv").set_index("year")
selected = fc.loc[fc.permno.str.contains("CHONGYI ZHANGYUAN|SHENZHEN SALUBRIS")
                  & fc.fperiod.isin(["2019Q3", "2019Q4", "2020Q1"])].copy()
assert len(selected) == 6
details, summary, audits, input_audits = [], [], [], []

for (period, cutoff), targets in selected.groupby(["fperiod", "est_cutoff"], sort=True):
    first = targets.iloc[0]
    ci = int(first.fyear * 4 + first.fqtr - 1 - scope["T_BASE"])
    M = scope["to_day"]([cutoff])[0]
    year = int(cutoff[:4])
    tr = meta.ci.between(ci - 12, ci - 1) & meta.A.lt(M) & np.isfinite(meta.y)
    params = {key: int(tuning.loc[year, key]) for key in ["max_depth", "min_child_weight", "n_estimators"]}
    params["reg_lambda"] = float(tuning.loc[year, "reg_lambda"])
    lo, hi = np.quantile(meta.loc[tr, "y"], [0.01, 0.99])
    model = xgb.XGBRegressor(**scope["XGB_FIXED"], **params)
    model.fit(X.loc[tr], meta.loc[tr, "y"].clip(lo, hi), verbose=False)

    # Validate the entire fitted batch against saved forecasts, not just the six
    # explained observations. Abort before presenting an attribution if it differs.
    batch = fc.loc[fc.fperiod.eq(period) & fc.est_cutoff.eq(cutoff) & fc.avail_xgb].copy()
    batch_fi = scope["FIRM_IDX"].reindex(batch.permno).to_numpy()
    rows = row_lookup.reindex(batch_fi * scope["N_T"] + ci).to_numpy().astype(int)
    predicted = meta.base.to_numpy()[rows] + model.predict(X.iloc[rows]) * meta.scale.to_numpy()[rows]
    max_error = float(np.max(np.abs(predicted - batch.f_xgb.to_numpy())))
    np.testing.assert_allclose(predicted, batch.f_xgb.to_numpy(), rtol=0, atol=1e-7)
    np.testing.assert_equal(batch.xgb_n_train.to_numpy(), tr.sum())
    audits.append(dict(fperiod=period, est_cutoff=cutoff, n_train=int(tr.sum()),
                       n_batch_checked=len(batch), max_forecast_difference=max_error,
                       xgboost_version=xgb.__version__, **params))
    model.save_model(OUT / f"model_{period}_{cutoff}.ubj")
    for row in targets.itertuples():
        fi = int(scope["FIRM_IDX"][row.permno])
        idx = int(row_lookup.loc[fi * scope["N_T"] + ci])
        A = scope["to_day"]([row.ann_date])[0]
        assert meta.loc[idx, "max_input_day"] < A
        vals, dates = scope["gather"](E, D, np.array([fi]), np.array([ci]),
                                       -np.arange(1, 17), np.array([A]))
        for lag in range(1, 17):
            date = dates[0, lag - 1]
            input_audits.append(dict(permno=row.permno, fperiod=period, lag=lag,
                                     eps=vals[0, lag - 1],
                                     public_date=str(np.datetime64(int(date), "D")) if date >= 0 else None,
                                     target_announcement=row.ann_date))
        contributions = model.get_booster().predict(xgb.DMatrix(X.iloc[[idx]]),
                                                     pred_contribs=True, approx_contribs=False)[0]
        scale = float(meta.loc[idx, "scale"])
        base = float(meta.loc[idx, "base"])
        phi = contributions[:-1].astype(float) * scale
        bias = float(contributions[-1]) * scale
        baseline = base + bias
        reconstructed = baseline + phi.sum()
        np.testing.assert_allclose(reconstructed, row.f_xgb, rtol=0, atol=1e-6)
        summary.append(dict(permno=row.permno, fperiod=period, ann_date=row.ann_date,
                            eps_actual=row.eps_actual, forecast=row.f_xgb, srw_eps=base,
                            historical_scale=scale, shap_bias_cny=bias,
                            baseline_cny=baseline, contribution_sum_cny=phi.sum(),
                            reconstructed_forecast=reconstructed,
                            max_input_date=row.max_input_date))
        for feature, value, contribution in zip(features, X.iloc[idx], phi):
            details.append(dict(permno=row.permno, fperiod=period, feature=feature,
                                feature_value=value, shap_cny_per_share=contribution))
    print(f"Verified {period}: {len(batch)} saved forecasts, max difference {max_error:.2g}", flush=True)

summary = pd.DataFrame(summary)
details = pd.DataFrame(details)
summary.to_csv(OUT / "prediction_summary.csv", index=False)
details.to_csv(OUT / "feature_contributions.csv", index=False)
pd.DataFrame(audits).to_csv(OUT / "reproduction_audit.csv", index=False)
pd.DataFrame(input_audits).to_csv(OUT / "input_publication_audit.csv", index=False)

labels = {"fq": "Fiscal quarter", "h_avail": "Gap to latest known EPS",
          "ds_mean12": "Mean prior YoY changes", "ds_median12": "Median prior YoY changes",
          "ds_std12": "Volatility of YoY changes", "n_ds12": "Count of prior YoY changes",
          "eps_std8": "Historical EPS volatility", "ttm_growth": "Change in annual EPS sum",
          "loss_share8": "Historical loss frequency", "loss_l4": "Loss one year earlier",
          "log_scale": "Log historical EPS scale", "n_avail12": "Available EPS count"}
labels.update({f"eps_l{k}": f"EPS {k} quarter{'s' if k != 1 else ''} earlier" for k in [1,2,3,4,5,6,7,8,12]})
labels.update({f"ds_l{k}": f"YoY change {k} quarter{'s' if k != 1 else ''} earlier" for k in [1,2,3,4]})
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 1, figsize=(7.4, 8.2))
for ax, row in zip(axes, summary.loc[summary.fperiod.eq("2019Q4")].itertuples()):
    d = details.loc[details.permno.eq(row.permno) & details.fperiod.eq(row.fperiod)].copy()
    d["magnitude"] = d.shap_cny_per_share.abs()
    d = d.sort_values("magnitude", ascending=False)
    top = d.head(6)
    values = top.shap_cny_per_share.to_list() + [d.iloc[6:].shap_cny_per_share.sum()]
    names = [labels[f] for f in top.feature] + ["All other features"]
    running = row.baseline_cny
    ax.barh(0, row.baseline_cny, color="#8b9299", height=.6)
    names = ["Baseline: SRW + model bias"] + names + ["XGBoost forecast"]
    for i, delta in enumerate(values, 1):
        end = running + delta
        ax.barh(i, abs(delta), left=min(running, end), color="#c34a46" if delta < 0 else "#2a78d6", height=.6)
        ax.plot([running, running], [i-1+.3, i-.3], color="#b3b3b3", linewidth=.6)
        ax.text(max(running, end) + .004, i, f"{delta:+.3f}", va="center", fontsize=8)
        running = end
    ax.barh(8, row.forecast, color="#168568", height=.6)
    ax.set_yticks(range(9), names)
    ax.invert_yaxis()
    ax.axvline(0, color="#888888", linewidth=.6)
    ax.set_xlabel("Contribution / prediction (CNY per share)")
    name = "Chongyi Zhangyuan" if "CHONGYI" in row.permno else "Shenzhen Salubris"
    ax.set_title(f"{name}: 2019Q4, announced {row.ann_date}\n"
                 f"SRW {row.srw_eps:.3f} | XGBoost {row.forecast:.3f} | Actual {row.eps_actual:.3f}",
                 fontsize=10, loc="left", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=.20)
fig.subplots_adjust(left=.34, right=.97, top=.92, bottom=.08, hspace=.55)
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"shap_2019q4.{ext}", dpi=200)
plt.close(fig)

text = """# SHAP explanations for selected EPS forecasts

Native XGBoost TreeSHAP (`pred_contribs=True`, `approx_contribs=False`), using
training-path weights. Each relevant model was refitted with the saved tuning
parameters and the original notebook's reconstruction and feature functions.
Every available saved forecast in each reconstructed batch was checked within
1e-7 CNY/share. `reproduction_audit.csv` records the results and fitted models
are saved here for reuse.

The model predicts scaled year-over-year EPS changes. For each observation:
forecast EPS = EPS(q-4) + scale * SHAP bias + sum(scale * feature SHAP).
The plotted baseline is EPS(q-4) plus the scaled SHAP bias. Contributions are
expressed in CNY/share, and sum to the saved forecast within 1e-6 CNY/share.
The plots show 2019Q4; the CSVs also cover 2019Q3 and 2020Q1 for both firms.
`feature_value` is the actual model input, often scaled, not necessarily raw EPS.
`input_publication_audit.csv` lists historical EPS and its publication date.

SHAP explains the fitted model, not the economic causes of actual earnings.
Correlated EPS features share credit; attribution depends on the method used.
The baseline includes the firm's fixed SRW anchor and scale; this decomposition
does not attribute changes in those quantities to their underlying inputs.
These checks do not establish that the source data are free of retrospective
revisions or other data-provenance issues.

Documentation: https://xgboost.readthedocs.io/en/stable/prediction.html
"""
(OUT / "README.md").write_text(text)
for row in summary.loc[summary.fperiod.eq("2019Q4")].itertuples():
    print(row.permno, "baseline", row.baseline_cny, "forecast", row.forecast)
    d = details.loc[details.permno.eq(row.permno) & details.fperiod.eq(row.fperiod)]
    print(d.loc[d.shap_cny_per_share.abs().nlargest(5).index,
                ["feature", "feature_value", "shap_cny_per_share"]].to_string(index=False))
