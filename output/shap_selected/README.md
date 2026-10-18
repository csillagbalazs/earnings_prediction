# SHAP explanations for selected EPS forecasts

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
