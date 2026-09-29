# HealthWise Smart Premium Engine

Two-stage pricing: a decision tree sorts each customer into a risk tier (Low / Medium / High),
then that tier's own linear regression predicts the annual charge, and the premium formula adds loadings.

## Files
| File | What it is |
|---|---|
| `HealthWise_Lab.ipynb` | The full lab (Parts A–E + extensions + bonus), executed, with reflections |
| `app.py` | Streamlit GUI + dashboard (8 tabs: quote, what-if, portfolio, Stage 1, Stage 2, segments & PCA, formula, batch scoring) |
| `healthwise_model.joblib` | Saved engine (Task E1). Re-created every time you run the notebook |
| `healthwise.csv` | The dataset (the app needs it next to `app.py`) |
| `unseen_customers_sample.csv` | Six made-up customers for Task E3 — replace with your instructor's `test_customers.csv` |
| `unseen_customers_scored.csv` | The notebook's scored output for E3 |
| `healthwise_dashboard.html` | Stand-alone web version of the calculator + dashboard (opens in any browser) |
| `.streamlit/config.toml` | App colour theme (optional) |

## Run
```
pip install -r requirements.txt
jupyter notebook HealthWise_Lab.ipynb      # Kernel → Restart & Run All
streamlit run app.py
```
Colab: upload `healthwise.csv` (and `test_customers.csv` if you have it) and run all cells. The ipywidgets GUI in Task E2 works in Colab.

If `healthwise_model.joblib` was saved with a different scikit-learn version, the app retrains the same models from `healthwise.csv` automatically.
