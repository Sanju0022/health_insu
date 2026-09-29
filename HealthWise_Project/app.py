"""
HealthWise Insurance - Smart Premium Engine (GUI + interactive dashboard)

Run:    streamlit run app.py
Needs:  healthwise.csv in the same folder.
Loads:  healthwise_model.joblib (saved by the notebook, Task E1). If it is missing, or was saved with an
        incompatible library version, the app retrains the same models from healthwise.csv on start-up.
"""
# ================= HealthWise two-stage engine (shared by notebook & app) =================
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

NUM = ['age', 'bmi', 'children', 'exercise_freq']      # kept numeric features
CAT = ['sex', 'smoker', 'region']                     # kept categorical features
FEATURES = NUM + CAT
ENG = ['obese', 'age_45plus', 'age_58plus', 'smoker_obese']   # engineered risk flags
TIERS = ['Low', 'Medium', 'High']
DEFAULT_PRICING = {'expense_load': 0.10, 'profit_margin': 0.05, 'safety_z': 0.0}


def clean_data(raw):
    """Drop duplicates, impute bmi from weight_kg (r≈0.99), exercise_freq with the median."""
    d = raw.drop_duplicates().reset_index(drop=True).copy()
    known = d['bmi'].notna()
    fit = LinearRegression().fit(d.loc[known, ['weight_kg']], d.loc[known, 'bmi'])
    if (~known).any():
        d.loc[~known, 'bmi'] = fit.predict(d.loc[~known, ['weight_kg']]).round(1)
    d['exercise_freq'] = d['exercise_freq'].fillna(d['exercise_freq'].median())
    return d, (float(fit.intercept_), float(fit.coef_[0]))


def _smoker01(d):
    if 'smoker_yes' in d.columns:
        return d['smoker_yes'].astype(int)
    return (d['smoker'].astype(str).str.strip().str.lower() == 'yes').astype(int)


def add_features(d):
    """Threshold flags suggested by the EDA (WHO obesity cut-off, age bands)."""
    d = d.copy()
    d['obese'] = (d['bmi'] >= 30).astype(int)
    d['age_45plus'] = (d['age'] >= 45).astype(int)
    d['age_58plus'] = (d['age'] >= 58).astype(int)
    d['smoker_obese'] = _smoker01(d) * d['obese']
    return d


def risk_points(d):
    """Transparent scorecard: 2×smoker + 1×(BMI≥30) + 1×(age≥45) + 1×(age≥58)."""
    return (2 * _smoker01(d) + (d['bmi'] >= 30).astype(int)
            + (d['age'] >= 45).astype(int) + (d['age'] >= 58).astype(int))


def scorecard_tier(points):
    points = np.asarray(points)
    return np.select([points == 0, points <= 2], ['Low', 'Medium'], 'High')


def make_pipe(model, num, scale=False):
    """Encoder + model in ONE object, so a single customer row is encoded exactly like training."""
    pre = ColumnTransformer([
        ('num', StandardScaler() if scale else 'passthrough', num),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CAT)])
    return Pipeline([('pre', pre), ('model', model)])


def build_bundle(d, bmi_fit, hp, random_state=42):
    """Train Stage 1 (classifiers) + Stage 2 (one regressor per tier) on the cleaned data."""
    d = add_features(d)
    y_tier = d['risk_tier']
    classifiers = {
        'Decision Tree (Entropy)': {'model': make_pipe(DecisionTreeClassifier(
            criterion='entropy', random_state=random_state, **hp['dt_entropy']), NUM), 'features': FEATURES},
        'Decision Tree (Gini)': {'model': make_pipe(DecisionTreeClassifier(
            criterion='gini', random_state=random_state, **hp['dt_gini']), NUM), 'features': FEATURES},
        'KNN': {'model': make_pipe(KNeighborsClassifier(**hp['knn']), NUM + ENG, scale=True),
                'features': FEATURES + ENG},
        'Naive Bayes': {'model': make_pipe(GaussianNB(), NUM + ENG), 'features': FEATURES + ENG},
        'Logistic Regression': {'model': make_pipe(LogisticRegression(max_iter=5000, **hp['logreg']),
                                                   NUM + ENG, scale=True), 'features': FEATURES + ENG},
    }
    for c in classifiers.values():
        c['model'].fit(d[c['features']], y_tier)

    regs, sigma = {}, {}
    for t in TIERS:
        sub = d[d['risk_tier'] == t]
        regs[t] = make_pipe(LinearRegression(), NUM).fit(sub[FEATURES], sub['annual_charge'])
        cvp = cross_val_predict(make_pipe(LinearRegression(), NUM), sub[FEATURES], sub['annual_charge'],
                                cv=KFold(5, shuffle=True, random_state=random_state))
        sigma[t] = float(np.sqrt(np.mean((sub['annual_charge'] - cvp) ** 2)))

    return {
        'clf': classifiers['Decision Tree (Entropy)']['model'],      # Stage 1 (deployed)
        'regs': regs,                                                 # Stage 2 (one per tier)
        'classifiers': classifiers,                                   # all Stage-1 candidates
        'features': FEATURES, 'num': NUM, 'cat': CAT, 'eng': ENG, 'tiers': TIERS,
        'med': {c: float(d[c].median()) for c in NUM},
        'cat_mode': {c: str(d[c].mode()[0]) for c in CAT},
        'bmi_from_weight': bmi_fit, 'sigma': sigma, 'pricing': dict(DEFAULT_PRICING),
        'hyperparams': hp, 'flat_charge': float(d['annual_charge'].mean()),
        'ranges': {c: (float(d[c].min()), float(d[c].max())) for c in NUM},
    }


def prepare_row(customer, M):
    """One customer dict -> clean one-row DataFrame (fills blanks the same way as training)."""
    row = pd.DataFrame([dict(customer)])
    for c in M['cat']:
        v = row.at[0, c] if c in row.columns else None
        row[c] = M['cat_mode'][c] if v is None or pd.isna(v) or str(v).strip() == '' else str(v).strip().lower()
    for c in M['num']:
        v = row.at[0, c] if c in row.columns else None
        v = pd.to_numeric(v, errors='coerce')
        if pd.isna(v) and c == 'bmi' and 'weight_kg' in row.columns and pd.notna(pd.to_numeric(row.at[0, 'weight_kg'], errors='coerce')):
            a, b = M['bmi_from_weight']
            v = round(a + b * float(row.at[0, 'weight_kg']), 1)
        row[c] = M['med'][c] if pd.isna(v) else float(v)
    return add_features(row)


def predict_full(customer, M, expense_load=None, profit_margin=None, safety_z=None, weighted=False):
    """Stage 1 -> tier (+ probabilities); Stage 2 -> that tier's regressor; then the pricing formula."""
    row = prepare_row(customer, M)
    X = row[M['features']]
    clf = M['clf']
    tier = str(clf.predict(X)[0])
    proba = {str(t): float(p) for t, p in zip(clf.classes_, clf.predict_proba(X)[0])}
    by_tier = {t: float(M['regs'][t].predict(X)[0]) for t in M['tiers']}
    expected = by_tier[tier]
    weighted_charge = float(sum(proba.get(t, 0) * by_tier[t] for t in M['tiers']))
    p = dict(M['pricing'])
    for k, v in (('expense_load', expense_load), ('profit_margin', profit_margin), ('safety_z', safety_z)):
        if v is not None:
            p[k] = v
    base = weighted_charge if weighted else expected
    risk_margin = p['safety_z'] * M['sigma'][tier]
    quoted = base * (1 + p['expense_load'] + p['profit_margin']) + risk_margin
    return {'tier': tier, 'proba': proba, 'expected_charge': expected, 'weighted_charge': weighted_charge,
            'charge_by_tier': by_tier, 'quoted_premium': float(quoted), 'risk_margin': float(risk_margin),
            'pricing': p, 'risk_points': int(risk_points(row).iloc[0]),
            'scorecard_tier': str(scorecard_tier(risk_points(row))[0]), 'row': row}
# ===========================================================================================

# ======================================== app setup ========================================
import os, io, json
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score, adjusted_rand_score,
                             precision_score, recall_score)
from sklearn.linear_model import Ridge, Lasso
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import export_graphviz

RANDOM_STATE = 42
DATA_FILE, MODEL_FILE, SAMPLE_FILE = 'healthwise.csv', 'healthwise_model.joblib', 'unseen_customers_sample.csv'
TIER_COLORS = {'Low': '#2a9d8f', 'Medium': '#e0a526', 'High': '#d8573c'}
INK, ACCENT, LINE = '#16323a', '#1f6f8b', '#d6e0e1'
DEFAULT_HP = {'dt_entropy': {'max_depth': 4, 'min_samples_leaf': 5}, 'dt_gini': {'max_depth': 4, 'min_samples_leaf': 5},
              'knn': {'n_neighbors': 7, 'weights': 'distance'}, 'logreg': {'C': 1}}
PRETTY = {'age': 'Age', 'bmi': 'BMI', 'children': 'Children', 'exercise_freq': 'Workouts/week',
          'sex_male': 'Male', 'smoker_yes': 'Smoker', 'region_north': 'North', 'region_south': 'South', 'region_west': 'West'}

st.set_page_config(page_title='HealthWise · Smart Premium Engine', page_icon='🏥', layout='wide')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select { font-family: 'Public Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp { background: #f4f7f7; }
.block-container { padding-top: 3.4rem; max-width: 1340px; }
h1, h2, h3, h4 { color: #16323a; letter-spacing: -0.01em; }
[data-testid="stSidebar"] { background: #e8eff0; border-right: 1px solid #d6e0e1; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
.hw-head { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; margin-bottom:.2rem; }
.hw-title { font-size: 1.85rem; margin: 0; font-weight: 800; color:#16323a; letter-spacing:-0.015em; line-height:1.2; }
.hw-head span { color:#4b6770; font-size: .98rem; }
.slip { background:#fff; border:1px solid #d6e0e1; border-radius:16px; padding:22px 26px 18px; }
.slip-row { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }
.pill { display:inline-block; padding:5px 14px; border-radius:999px; color:#fff; font-weight:700; font-size:.95rem; }
.prob { color:#4b6770; font-size:.9rem; }
.premium { font-size: clamp(2.6rem, 5vw, 3.6rem); font-weight:800; color:#16323a; line-height:1.05;
           font-variant-numeric: tabular-nums; letter-spacing:-0.02em; margin-top:.4rem; }
.premium small { font-size:1rem; font-weight:500; color:#4b6770; letter-spacing:0; }
.sub { color:#4b6770; font-size:.92rem; margin-top:.15rem; }
.calc { font-variant-numeric: tabular-nums; color:#16323a; background:#f4f7f7; border-radius:10px;
        padding:8px 12px; margin-top:12px; font-size:.9rem; }
.ruler { display:grid; grid-template-columns: repeat(6, 1fr); gap:4px; margin:16px 0 4px; }
.ruler div { height:34px; border-radius:6px; display:flex; align-items:center; justify-content:center;
             font-weight:700; font-size:.85rem; color:rgba(255,255,255,.92); opacity:.35; }
.ruler div.on { opacity:1; outline:3px solid #16323a; outline-offset:2px; }
.ruler-legend { display:grid; grid-template-columns: 1fr 2fr 3fr; gap:4px; font-size:.78rem; color:#4b6770; text-align:center; }
.why li { margin-bottom:.25rem; }
.why .yes { color:#b43d24; font-weight:700; } .why .no { color:#7d9399; }
</style>
""", unsafe_allow_html=True)

_V = tuple(int(x) for x in st.__version__.split('.')[:2] if x.isdigit())
_WIDE = {} if _V >= (1, 48) else {'use_container_width': True}


def show_fig(fig, key=None, height=None):
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8), font_family='Public Sans, system-ui, sans-serif')
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, key=key, **_WIDE)


def show_df(df, **kw):
    st.dataframe(df, **_WIDE, **kw)


def money(v):
    return f"${v:,.0f}" if v >= 0 else f"−${abs(v):,.0f}"


def recolor_tree(dot):
    # sklearn colours classes generically; repaint each node in its tier colour, keeping the purity shading
    import re

    def fix(m):
        body = m.group(0)
        cls, fc = re.search(r'class = (\w+)', body), re.search(r'fillcolor="#([0-9a-fA-F]{6})"', body)
        if not (cls and fc) or cls.group(1) not in TIER_COLORS:
            return body
        rgb = [int(fc.group(1)[i:i + 2], 16) for i in (0, 2, 4)]
        alpha = max(0.10, min(1.0, (255 - min(rgb)) / 198))
        return body.replace(fc.group(0), f'fillcolor="{TIER_COLORS[cls.group(1)]}{int(alpha * 255):02x}"')
    return re.sub(r'^\d+ \[label=.*$', fix, dot, flags=re.M)


# ======================================== data + engine ========================================
if not os.path.exists(DATA_FILE):
    st.title('🏥 HealthWise — Smart Premium Engine')
    st.warning('`healthwise.csv` was not found next to `app.py`. Upload it once and the app will save a copy.')
    up = st.file_uploader('Upload healthwise.csv', type='csv')
    if up is None:
        st.stop()
    with open(DATA_FILE, 'wb') as f:
        f.write(up.getbuffer())
    st.rerun()


@st.cache_data(show_spinner=False)
def load_data():
    raw = pd.read_csv(DATA_FILE)
    d, bmi_fit = clean_data(raw)
    return raw, d, bmi_fit


@st.cache_resource(show_spinner='Loading the premium engine…')
def load_engine():
    _, d, bmi_fit = load_data()
    try:
        M = joblib.load(MODEL_FILE)
        for k in ('classifiers', 'sigma', 'cat_mode', 'bmi_from_weight'):
            if k not in M:
                raise KeyError(f'bundle has no "{k}"')
        predict_full({'age': 40, 'sex': 'male', 'bmi': 27, 'children': 1, 'smoker': 'no',
                      'region': 'north', 'exercise_freq': 3}, M)
        return M, 'Loaded healthwise_model.joblib'
    except Exception as e:  # missing file, template-format bundle, or library-version mismatch
        M = build_bundle(d, bmi_fit, DEFAULT_HP, random_state=RANDOM_STATE)
        try:
            joblib.dump(M, MODEL_FILE)
        except Exception:
            pass
        return M, f'Retrained from healthwise.csv ({type(e).__name__})'


raw, dfc, BMI_FIT = load_data()
M, ENGINE_STATUS = load_engine()
DUM = pd.get_dummies(dfc[FEATURES], columns=CAT, drop_first=True, dtype=int)


@st.cache_data(show_spinner='Running the model comparisons…')
def analytics(hp_json):
    hp = json.loads(hp_json)
    d = add_features(dfc)
    idx_tr, idx_te = train_test_split(d.index, test_size=0.2, random_state=RANDOM_STATE, stratify=d['risk_tier'])
    tr, te = d.loc[idx_tr], d.loc[idx_te]
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    specs = {
        'Naive Bayes (Gaussian)': lambda num: make_pipe(GaussianNB(), num),
        'KNN': lambda num: make_pipe(KNeighborsClassifier(**hp['knn']), num, scale=True),
        'Decision Tree (Gini)': lambda num: make_pipe(DecisionTreeClassifier(criterion='gini', random_state=RANDOM_STATE, **hp['dt_gini']), num),
        'Decision Tree (Entropy)': lambda num: make_pipe(DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, **hp['dt_entropy']), num),
        'Logistic Regression': lambda num: make_pipe(LogisticRegression(max_iter=5000, **hp['logreg']), num, scale=True),
    }
    rows, cms, reports = [], {}, {}

    def record(fs, name, p, cvacc, tracc):
        rows.append({'features': fs, 'model': name, 'CV accuracy': cvacc, 'train accuracy': tracc,
                     'test accuracy': accuracy_score(te['risk_tier'], p), 'test macro-F1': f1_score(te['risk_tier'], p, average='macro')})
        cms[f'{name} · {fs}'] = confusion_matrix(te['risk_tier'], p, labels=TIERS).tolist()
        reports[f'{name} · {fs}'] = pd.DataFrame({
            'precision': precision_score(te['risk_tier'], p, labels=TIERS, average=None, zero_division=0),
            'recall': recall_score(te['risk_tier'], p, labels=TIERS, average=None, zero_division=0)}, index=TIERS)

    for fs, num in [('raw', NUM), ('raw + engineered', NUM + ENG)]:
        feats = num + CAT
        for name, mk in specs.items():
            m = mk(num).fit(tr[feats], tr['risk_tier'])
            record(fs, name, m.predict(te[feats]), cross_val_score(mk(num), tr[feats], tr['risk_tier'], cv=cv).mean(),
                   accuracy_score(tr['risk_tier'], m.predict(tr[feats])))

    def bins(x):
        return pd.DataFrame({'a': pd.cut(x['age'], [0, 29, 44, 57, 200], labels=False),
                             'b': pd.cut(x['bmi'], [0, 18.5, 25, 30, 35, 200], labels=False, right=False),
                             's': _smoker01(x), 'e': pd.cut(x['exercise_freq'], [-1, 2, 4, 7], labels=False)}, index=x.index)
    cnb = CategoricalNB().fit(bins(tr), tr['risk_tier'])
    record('binned', 'Naive Bayes (Categorical)', cnb.predict(bins(te)),
           cross_val_score(CategoricalNB(), bins(tr), tr['risk_tier'], cv=cv).mean(), accuracy_score(tr['risk_tier'], cnb.predict(bins(tr))))
    sc_pred = scorecard_tier(risk_points(te))
    record('rules', 'Risk scorecard', sc_pred, accuracy_score(tr['risk_tier'], scorecard_tier(risk_points(tr))),
           accuracy_score(tr['risk_tier'], scorecard_tier(risk_points(tr))))
    cls_tbl = pd.DataFrame(rows)

    # Gini / entropy of candidate root splits (training rows)
    def gini(s): p = s.value_counts(normalize=True); return 1 - (p ** 2).sum()
    def ent(s): p = s.value_counts(normalize=True); return -(p * np.log2(p)).sum()
    y = tr['risk_tier']; split_rows = []
    for nm, mk in {'Smoker = yes': _smoker01(tr) == 1, 'Age ≥ 45': tr['age'] >= 45, 'BMI ≥ 30': tr['bmi'] >= 30,
                   'Workouts ≤ 2': tr['exercise_freq'] <= 2, 'Sex = male': tr['sex'] == 'male', 'Children ≥ 2': tr['children'] >= 2}.items():
        L, R = y[mk], y[~mk]; wl, wr = len(L) / len(y), len(R) / len(y)
        split_rows.append({'candidate first split': nm, 'Gini decrease': gini(y) - wl * gini(L) - wr * gini(R),
                           'Information gain (bits)': ent(y) - wl * ent(L) - wr * ent(R)})
    splits = pd.DataFrame(split_rows).sort_values('Information gain (bits)', ascending=False)

    # Stage 2: one-size-fits-all vs two-stage, same train/test rows
    Xtr, Xte, ytr, yte = tr[FEATURES], te[FEATURES], tr['annual_charge'], te['annual_charge']
    glob = make_pipe(LinearRegression(), NUM).fit(Xtr, ytr)
    gbr = make_pipe(GradientBoostingRegressor(random_state=RANDOM_STATE), NUM).fit(Xtr, ytr)
    regs_tr = {t: make_pipe(LinearRegression(), NUM).fit(tr.loc[tr['risk_tier'] == t, FEATURES], tr.loc[tr['risk_tier'] == t, 'annual_charge']) for t in TIERS}
    clf_tr = specs['Decision Tree (Entropy)'](NUM).fit(Xtr, tr['risk_tier'])
    t_hat = clf_tr.predict(Xte)

    def route(tiers):
        out = np.zeros(len(te)); tiers = np.asarray(tiers)
        for t in TIERS:
            if (tiers == t).any():
                out[tiers == t] = regs_tr[t].predict(Xte[tiers == t])
        return out
    preds = {'Flat price (everyone pays the average)': np.full(len(te), ytr.mean()),
             'Global linear regression': glob.predict(Xte),
             'Global gradient boosting': gbr.predict(Xte),
             'Two-stage · true tier': route(te['risk_tier']),
             'Two-stage · predicted tier (realistic)': route(t_hat)}
    reg_tbl = pd.DataFrame({k: {'MAE': mean_absolute_error(yte, v), 'RMSE': float(np.sqrt(mean_squared_error(yte, v))),
                                'R²': r2_score(yte, v)} for k, v in preds.items()}).T

    # Feature audit on the cleaned (deduplicated) data — junk columns included
    audit = []
    for c in ['age', 'bmi', 'weight_kg', 'children', 'exercise_freq', 'lucky_number']:
        r, p = stats.pearsonr(dfc[c], dfc['annual_charge'])
        audit.append({'column': c, 'type': 'numeric', 'evidence': f'r = {r:+.3f}', 'p-value': p})
    for c in ['sex', 'smoker', 'region', 'favorite_color', 'zodiac_sign', 'preferred_contact', 'marketing_opt_in']:
        p = stats.kruskal(*[g['annual_charge'].values for _, g in dfc.groupby(c)]).pvalue
        g = dfc.groupby(c)['annual_charge'].mean()
        audit.append({'column': c, 'type': f'{dfc[c].nunique()} categories', 'evidence': f'spread {(g.max() - g.min()) / dfc["annual_charge"].mean():.1%}', 'p-value': p})
    decision = {'age': 'Keep', 'bmi': 'Keep', 'children': 'Keep', 'exercise_freq': 'Keep', 'sex': 'Keep', 'smoker': 'Keep', 'region': 'Keep (weak)',
                'weight_kg': 'Drop — redundant with BMI (r = 0.99, VIF 66)', 'lucky_number': 'Drop — irrelevant',
                'favorite_color': 'Drop — irrelevant', 'zodiac_sign': 'Drop — irrelevant (the zodiac trap)',
                'preferred_contact': 'Drop — irrelevant', 'marketing_opt_in': 'Drop — irrelevant'}
    audit = pd.DataFrame(audit); audit['decision'] = audit['column'].map(decision)
    audit = pd.concat([audit, pd.DataFrame([{'column': 'date_of_birth', 'type': 'date', 'evidence': 'r(age) = 1.00', 'p-value': np.nan, 'decision': 'Drop — redundant with age'},
                                             {'column': 'customer_id', 'type': 'label', 'evidence': 'unique per row', 'p-value': np.nan, 'decision': 'Drop — identifier'}])])

    return {'cls': cls_tbl, 'cms': cms, 'reports': reports, 'splits': splits, 'reg': reg_tbl, 'audit': audit,
            'test_actual': yte.values, 'test_tier': te['risk_tier'].values, 'test_pred_tier': t_hat,
            'preds': {k: v.tolist() for k, v in preds.items()}, 'flat_mean': float(ytr.mean()),
            'scorecard_ct': pd.crosstab(risk_points(d).rename('risk points'), d['risk_tier'])[TIERS]}


A = analytics(json.dumps(M['hyperparams'], sort_keys=True, default=str))

# ======================================== sidebar: the customer ========================================
PRESETS = {
    'Young, healthy non-smoker': dict(age=26, sex='female', bmi=22.5, children=0, smoker=False, region='south', exercise_freq=5),
    'Family, overweight': dict(age=38, sex='male', bmi=31.2, children=3, smoker=False, region='north', exercise_freq=2),
    'Mid-life smoker': dict(age=52, sex='male', bmi=27.0, children=1, smoker=True, region='east', exercise_freq=2),
    'Senior, obese non-smoker': dict(age=61, sex='female', bmi=33.5, children=2, smoker=False, region='west', exercise_freq=3),
}
DEFAULTS = dict(age=35, sex='female', bmi=26.0, children=1, smoker=False, region='north', exercise_freq=4)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault('bmi_from_hw', False)


def apply_preset():
    p = PRESETS.get(st.session_state['preset'])
    if p:
        st.session_state.update(p)
        st.session_state['bmi_from_hw'] = False


with st.sidebar:
    st.markdown('### Customer')
    st.selectbox('Start from an example', ['Custom'] + list(PRESETS), key='preset', on_change=apply_preset)
    st.slider('Age', 18, 64, key='age')
    st.radio('Sex', ['female', 'male'], key='sex', horizontal=True)
    if st.toggle('Work out BMI from height and weight', key='bmi_from_hw'):
        c1, c2 = st.columns(2)
        h_cm = c1.number_input('Height (cm)', 130, 215, 170)
        w_kg = c2.number_input('Weight (kg)', 35, 180, 75)
        bmi_val = round(w_kg / (h_cm / 100) ** 2, 1)
        st.caption(f'BMI = **{bmi_val}**')
    else:
        st.slider('BMI', 16.0, 45.0, key='bmi', step=0.1)
        bmi_val = st.session_state['bmi']
    st.slider('Children (dependants on the policy)', 0, 4, key='children')
    st.toggle('Smoker', key='smoker')
    st.selectbox('Region', ['north', 'south', 'east', 'west'], key='region')
    st.slider('Workouts per week', 0, 7, key='exercise_freq')

    st.markdown('### Pricing')
    exp_load = st.slider('Expense loading', 0, 30, 10, format='%d%%') / 100
    margin = st.slider('Profit margin', 0, 20, 5, format='%d%%') / 100
    safety = st.slider('Safety margin (× the tier\'s typical error σ)', 0.0, 2.0, 0.0, 0.25)
    price_mode = st.radio('Price using', ['Predicted tier', 'Probability-weighted tiers'],
                          help='Probability-weighted blends the three tier equations by the classifier\'s probabilities.')
    st.caption(f'Engine: {ENGINE_STATUS}')

CUSTOMER = dict(age=st.session_state['age'], sex=st.session_state['sex'], bmi=float(bmi_val),
                children=st.session_state['children'], smoker='yes' if st.session_state['smoker'] else 'no',
                region=st.session_state['region'], exercise_freq=st.session_state['exercise_freq'])
WEIGHTED = price_mode.startswith('Probability')


def quote(c):
    return predict_full(c, M, expense_load=exp_load, profit_margin=margin, safety_z=safety, weighted=WEIGHTED)


def price_of(r):
    return r['quoted_premium']


Q = quote(CUSTOMER)

# ======================================== header + tabs ========================================
st.markdown('<div class="hw-head"><div class="hw-title">🏥 HealthWise Smart Premium Engine</div>'
            '<span>Stage 1 sorts the customer into a risk tier · Stage 2 prices them with that tier\'s own cost model</span></div>',
            unsafe_allow_html=True)

tabs = st.tabs(['🧮 Quote', '🔁 What-if', '📊 Portfolio', '🧠 Stage 1 · Risk tier', '📈 Stage 2 · Charge',
                '🧩 Segments & PCA', '🧾 Premium formula', '📂 Batch scoring'])

# ---------------------------------------------------------------- Quote
with tabs[0]:
    tier, proba = Q['tier'], Q['proba']
    col = TIER_COLORS[tier]
    left, right = st.columns([1.15, 1], gap='large')
    with left:
        pts = Q['risk_points']
        cells = ''.join(f'<div class="{"on" if i == min(pts, 5) else ""}" style="background:{TIER_COLORS[scorecard_tier([i])[0]]}">{i}</div>' for i in range(6))
        p = Q['pricing']
        base_label = 'probability-weighted charge' if WEIGHTED else 'expected charge'
        base_val = Q['weighted_charge'] if WEIGHTED else Q['expected_charge']
        st.markdown(f"""
<div class="slip">
  <div class="slip-row">
    <div><span class="pill" style="background:{col}">{tier} risk</span>
    <span class="prob">&nbsp; {proba.get(tier, 0):.0%} confident · Low {proba.get('Low', 0):.0%} · Medium {proba.get('Medium', 0):.0%} · High {proba.get('High', 0):.0%}</span></div>
  </div>
  <div class="ruler">{cells}</div>
  <div class="ruler-legend"><span>Low · 0 pts</span><span>Medium · 1–2 pts</span><span>High · 3+ pts</span></div>
  <div class="premium">{money(Q['quoted_premium'])} <small>per year · {money(Q['quoted_premium'] / 12)} per month</small></div>
  <div class="sub">Expected annual medical charge (Stage 2): <b>{money(Q['expected_charge'])}</b></div>
  <div class="calc">{money(base_val)} {base_label} × (1 + {p['expense_load']:.0%} expenses + {p['profit_margin']:.0%} margin)
  {'+ ' + money(Q['risk_margin']) + ' safety margin' if Q['risk_margin'] > 0 else ''} = <b>{money(Q['quoted_premium'])}</b></div>
</div>""", unsafe_allow_html=True)

    with right:
        row = Q['row'].iloc[0]
        flags = [('Smoker', CUSTOMER['smoker'] == 'yes', 2), ('BMI ≥ 30 (WHO obesity line)', row['bmi'] >= 30, 1),
                 ('Age ≥ 45', row['age'] >= 45, 1), ('Age ≥ 58', row['age'] >= 58, 1)]
        items = ''.join(f'<li><span class="{"yes" if on else "no"}">{"+" + str(w) if on else "+0"}</span> &nbsp;{name}</li>' for name, on, w in flags)
        st.markdown(f'#### Why {tier}?')
        st.markdown(f'<ul class="why">{items}</ul>Risk points = <b>{pts}</b> → scorecard says <b>{Q["scorecard_tier"]}</b>'
                    + ('' if Q['scorecard_tier'] == tier else f' (the decision tree says <b>{tier}</b>; they disagree for &lt;1% of customers)'),
                    unsafe_allow_html=True)
        X1 = Q['row'][M['features'] + M['eng']]
        ops = []
        for name, c in M['classifiers'].items():
            pr = c['model'].predict_proba(X1[c['features']])[0]
            k = int(np.argmax(pr)); ops.append({'model': name, 'tier': str(c['model'].classes_[k]), 'confidence': f'{pr[k]:.0%}'})
        st.markdown('**What every Stage-1 model says**')
        show_df(pd.DataFrame(ops).set_index('model'))

    st.markdown('#### How this premium is built')
    reg = M['regs'][tier]
    names = [n.split('__', 1)[1] for n in reg.named_steps['pre'].get_feature_names_out()]
    xv = reg.named_steps['pre'].transform(Q['row'][M['features']])[0]
    contrib = pd.Series(reg.named_steps['model'].coef_ * xv, index=names)
    region_total = contrib[[n for n in names if n.startswith('region_')]].sum()
    parts = [(f'{tier} base', reg.named_steps['model'].intercept_)]
    for n in ['age', 'bmi', 'children', 'exercise_freq', 'sex_male', 'smoker_yes']:
        if abs(contrib[n]) > 0.5:
            parts.append((PRETTY[n], contrib[n]))
    if abs(region_total) > 0.5:
        parts.append(('Region', region_total))
    labels = [a for a, _ in parts] + ['Expected charge']
    measures = ['absolute'] + ['relative'] * (len(parts) - 1) + ['total']
    values = [b for _, b in parts] + [0]
    if WEIGHTED:
        labels.append('Tier blending'); measures.append('relative'); values.append(Q['weighted_charge'] - Q['expected_charge'])
    labels += ['Expenses', 'Margin']; measures += ['relative', 'relative']
    values += [base_val * p['expense_load'], base_val * p['profit_margin']]
    if Q['risk_margin'] > 0:
        labels.append('Safety'); measures.append('relative'); values.append(Q['risk_margin'])
    labels.append('Quoted premium'); measures.append('total'); values.append(0)
    fig = go.Figure(go.Waterfall(x=labels, y=values, measure=measures, text=[('' if m == 'total' else money(v) if m == 'absolute' else ('+' if v >= 0 else '') + money(v)) for v, m in zip(values, measures)],
                                 textposition='outside', connector={'line': {'color': LINE}},
                                 increasing={'marker': {'color': '#d8573c'}}, decreasing={'marker': {'color': '#2a9d8f'}},
                                 totals={'marker': {'color': ACCENT}}))
    fig.update_layout(title=f'The {tier}-tier equation applied to this customer, then the loadings', showlegend=False, yaxis_title='$ per year')
    show_fig(fig, height=420, key='waterfall')

    c1, c2 = st.columns(2)
    with c1:
        bt = pd.DataFrame({'tier': TIERS, 'charge': [Q['charge_by_tier'][t] for t in TIERS],
                           'probability': [proba.get(t, 0) for t in TIERS]})
        fig = px.bar(bt, x='tier', y='charge', color='tier', color_discrete_map=TIER_COLORS, text=bt['charge'].map(money),
                     title='The same customer priced by each tier\'s equation')
        fig.update_layout(showlegend=False, yaxis_title='expected charge ($)')
        show_fig(fig, height=330, key='bytier')
    with c2:
        st.markdown('**Reading the quote**')
        st.markdown(f"- Stage 1 (decision tree) puts this customer in **{tier}** with {proba.get(tier, 0):.0%} confidence.\n"
                    f"- Stage 2 uses only the **{tier}-tier** regression, trained on {int((dfc['risk_tier'] == tier).sum())} customers like them.\n"
                    f"- That tier's model is typically off by about **{money(M['sigma'][tier])}** (cross-validated RMSE).\n"
                    f"- Change anything in the sidebar and every number here recalculates instantly.")

# ---------------------------------------------------------------- What-if
with tabs[1]:
    st.markdown('#### What moves this customer\'s premium?')
    c = CUSTOMER
    scen = [('Quits smoking' if c['smoker'] == 'yes' else 'Starts smoking', {'smoker': 'no' if c['smoker'] == 'yes' else 'yes'}),
            ('One more child', {'children': min(4, c['children'] + 1)}), ('One child fewer', {'children': max(0, c['children'] - 1)}),
            ('5 years older', {'age': min(64, c['age'] + 5)}), ('10 years older', {'age': min(64, c['age'] + 10)}),
            ('BMI 3 points lower', {'bmi': max(16.0, c['bmi'] - 3)}), ('BMI 3 points higher', {'bmi': min(45.0, c['bmi'] + 3)}),
            ('2 more workouts a week', {'exercise_freq': min(7, c['exercise_freq'] + 2)}),
            ('Other sex', {'sex': 'male' if c['sex'] == 'female' else 'female'})]
    scen += [(f'Moves to the {r}', {'region': r}) for r in ['north', 'south', 'east', 'west'] if r != c['region']]
    rows = []
    for name, ch in scen:
        if all(c.get(k) == v for k, v in ch.items()):
            continue
        r2 = quote({**c, **ch})
        rows.append({'change': name, 'new tier': r2['tier'], 'tier move': f"{Q['tier']} → {r2['tier']}" if r2['tier'] != Q['tier'] else 'same tier',
                     'new premium': r2['quoted_premium'], 'Δ premium': r2['quoted_premium'] - Q['quoted_premium']})
    wi = pd.DataFrame(rows).sort_values('Δ premium')
    fig = px.bar(wi, x='Δ premium', y='change', orientation='h', text=wi['Δ premium'].map(lambda v: f'{v:+,.0f}'),
                 color=np.where(wi['Δ premium'] >= 0, 'costs more', 'costs less'),
                 color_discrete_map={'costs more': '#d8573c', 'costs less': '#2a9d8f'}, hover_data=['tier move', 'new tier'],
                 title=f'Change in the quoted premium from {money(Q["quoted_premium"])} (one change at a time)')
    fig.update_layout(legend_title_text='', yaxis_title='', xaxis_title='Δ premium ($ per year)')
    show_fig(fig, height=460, key='whatif')
    show_df(wi.set_index('change').style.format({'new premium': '${:,.0f}', 'Δ premium': '{:+,.0f}'}))

    st.markdown('#### Premium curves — the jumps are tier changes')
    c1, c2 = st.columns(2)
    for holder, var, grid, label in [(c1, 'age', np.arange(18, 65), 'Age'), (c2, 'bmi', np.round(np.arange(16, 45.01, 0.5), 1), 'BMI')]:
        recs = []
        for sm in ['no', 'yes']:
            for v in grid:
                r2 = quote({**c, var: float(v), 'smoker': sm})
                recs.append({label: v, 'premium': r2['quoted_premium'], 'tier': r2['tier'], 'smoker': sm})
        cur = pd.DataFrame(recs)
        fig = px.line(cur, x=label, y='premium', line_dash='smoker', color_discrete_sequence=['#8aa3aa'])
        fig.add_trace(go.Scatter(x=cur[label], y=cur['premium'], mode='markers', showlegend=False,
                                 marker=dict(size=6, color=cur['tier'].map(TIER_COLORS))))
        fig.add_trace(go.Scatter(x=[c[var]], y=[Q['quoted_premium']], mode='markers', name='this customer',
                                 marker=dict(size=16, color=INK, symbol='diamond')))
        fig.update_layout(title=f'Quoted premium vs {label} (dot colour = tier)', yaxis_title='$ per year')
        with holder:
            show_fig(fig, height=380, key=f'curve_{var}')

# ---------------------------------------------------------------- Portfolio
with tabs[2]:
    f1, f2, f3, f4 = st.columns([1.2, 1, 1.4, 1.4])
    sel_t = f1.multiselect('Tiers', TIERS, TIERS)
    sel_s = f2.multiselect('Smoker', ['no', 'yes'], ['no', 'yes'])
    sel_r = f3.multiselect('Regions', ['north', 'south', 'east', 'west'], ['north', 'south', 'east', 'west'])
    a_lo, a_hi = f4.slider('Age range', 18, 64, (18, 64))
    view = dfc[dfc['risk_tier'].isin(sel_t) & dfc['smoker'].isin(sel_s) & dfc['region'].isin(sel_r) & dfc['age'].between(a_lo, a_hi)]
    if view.empty:
        st.info('No customers match these filters — widen one of them.')
    else:
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric('Customers', f'{len(view):,}')
        k2.metric('Average annual charge', money(view['annual_charge'].mean()))
        k3.metric('Smokers', f"{(view['smoker'] == 'yes').mean():.0%}")
        k4.metric('High-risk share', f"{(view['risk_tier'] == 'High').mean():.0%}")
        k5.metric('Total claims', f"${view['annual_charge'].sum() / 1e6:,.1f}M")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(view, x='annual_charge', color='risk_tier', category_orders={'risk_tier': TIERS},
                               color_discrete_map=TIER_COLORS, nbins=50, title=f"Annual charge — skew {view['annual_charge'].skew():.2f}")
            fig.update_layout(legend_title_text='tier', xaxis_title='annual charge ($)'); show_fig(fig, height=360, key='hist')
        with c2:
            fig = px.scatter(view, x='age', y='annual_charge', color='risk_tier', symbol='smoker', category_orders={'risk_tier': TIERS},
                             color_discrete_map=TIER_COLORS, opacity=.75, title='Age vs charge (shape = smoker)',
                             hover_data=['bmi', 'children', 'exercise_freq'])
            fig.update_layout(legend_title_text=''); show_fig(fig, height=360, key='scatter_age')
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.box(view, x='smoker', y='annual_charge', color='smoker', color_discrete_map={'no': '#2a9d8f', 'yes': '#d8573c'},
                         title='Charge by smoking'); fig.update_layout(showlegend=False); show_fig(fig, height=340, key='box_smk')
        with c2:
            fig = px.scatter(view, x='bmi', y='annual_charge', color='risk_tier', color_discrete_map=TIER_COLORS,
                             category_orders={'risk_tier': TIERS}, opacity=.7, title='BMI vs charge')
            fig.add_vline(x=30, line_dash='dot', line_color=INK, annotation_text='BMI 30')
            fig.update_layout(showlegend=False); show_fig(fig, height=340, key='scatter_bmi')
        with c3:
            mix = view['risk_tier'].value_counts().reindex(TIERS).fillna(0).reset_index()
            mix.columns = ['tier', 'customers']
            fig = px.pie(mix, names='tier', values='customers', color='tier', color_discrete_map=TIER_COLORS, hole=.55, title='Tier mix')
            show_fig(fig, height=340, key='pie')
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown('**Feature audit — what was kept and why** (on all 1,200 customers)')
            show_df(A['audit'].set_index('column').style.format({'p-value': lambda v: '' if pd.isna(v) else f'{v:.3g}'}))
        with c2:
            corr = view[['age', 'bmi', 'weight_kg', 'children', 'exercise_freq', 'lucky_number', 'annual_charge']].corr()
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title='Correlations')
            show_fig(fig, height=430, key='corr')

# ---------------------------------------------------------------- Stage 1
with tabs[3]:
    cls = A['cls']
    st.markdown('#### Five classifiers, with and without the engineered threshold flags')
    st.caption('Same stratified 80/20 split as the notebook and the deployed hyper-parameters. Engineered flags: BMI ≥ 30, age ≥ 45, age ≥ 58, smoker × obese.')
    order = ['Naive Bayes (Gaussian)', 'Naive Bayes (Categorical)', 'KNN', 'Logistic Regression', 'Decision Tree (Gini)', 'Decision Tree (Entropy)', 'Risk scorecard']
    fig = px.bar(cls, y='model', x='test accuracy', color='features', barmode='group', orientation='h', text=cls['test accuracy'].map('{:.3f}'.format),
                 category_orders={'model': order}, color_discrete_map={'raw': '#9aa5b1', 'raw + engineered': '#2a9d8f', 'binned': ACCENT, 'rules': INK},
                 title='Test accuracy (240 unseen customers)')
    fig.update_layout(xaxis_range=[0.5, 1.0], yaxis_title='', legend_title_text='features')
    show_fig(fig, height=470, key='cls_bar')
    show_df(cls.set_index(['model', 'features']).sort_values('test accuracy', ascending=False).style.format('{:.3f}'))

    c1, c2 = st.columns([1, 1])
    with c1:
        pick = st.selectbox('Confusion matrix for', list(A['cms']), index=list(A['cms']).index('Decision Tree (Entropy) · raw'))
        cm = np.array(A['cms'][pick])
        fig = px.imshow(cm, x=[f'pred {t}' for t in TIERS], y=[f'true {t}' for t in TIERS], text_auto=True,
                        color_continuous_scale='Blues', title=f'{pick} — accuracy {np.trace(cm) / cm.sum():.3f}')
        fig.update_layout(coloraxis_showscale=False); show_fig(fig, height=380, key='cm')
        show_df(A['reports'][pick].style.format('{:.2f}'))
        st.caption(f"True High predicted Low: **{cm[2, 0]}** (the costliest error) · true Low predicted High: **{cm[0, 2]}**")
    with c2:
        st.markdown('**How a tree chooses its first question**')
        st.latex(r'\text{Gini}=1-\sum_k p_k^2 \qquad \text{Entropy}=-\sum_k p_k\log_2 p_k')
        show_df(A['splits'].set_index('candidate first split').style.format('{:.4f}'))
        st.caption('Both criteria rank the same splits first: smoker, then age 45, then BMI 30 — which is why the Gini and Entropy trees end up identical here.')
        st.markdown('**The scorecard distilled from the tree** — risk points vs recorded tier')
        show_df(A['scorecard_ct'])

    st.markdown('#### The deployed Stage-1 model: Decision Tree (Entropy)')
    tree = M['clf'].named_steps['model']
    fn = [n.split('__', 1)[1] for n in M['clf'].named_steps['pre'].get_feature_names_out()]
    dot = export_graphviz(tree, feature_names=fn, class_names=[str(x) for x in tree.classes_], filled=True, rounded=True,
                          impurity=True, proportion=False, precision=2)
    try:
        st.graphviz_chart(recolor_tree(dot), **_WIDE)
    except Exception:
        st.code(dot[:3000])

# ---------------------------------------------------------------- Stage 2
with tabs[4]:
    reg_tbl = A['reg']
    st.markdown('#### Does the two-stage engine beat one model for everyone?')
    k1, k2, k3 = st.columns(3)
    g, t2 = reg_tbl.loc['Global linear regression'], reg_tbl.loc['Two-stage · predicted tier (realistic)']
    k1.metric('Global model — MAE', money(g['MAE']), f"R² {g['R²']:.3f}", delta_color='off')
    k2.metric('Two-stage (realistic) — MAE', money(t2['MAE']), f"{t2['MAE'] / g['MAE'] - 1:.0%} error", delta_color='inverse')
    k3.metric('Two-stage — R²', f"{t2['R²']:.3f}", f"{t2['R²'] - g['R²']:+.3f} vs global")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        fig = px.bar(reg_tbl.reset_index(), x='MAE', y='index', orientation='h', text=reg_tbl['MAE'].map(money),
                     color=['one model' if not i.startswith('Two') else 'two-stage' for i in reg_tbl.index],
                     color_discrete_map={'one model': '#9aa5b1', 'two-stage': '#2a9d8f'}, title='Test MAE — lower is better')
        fig.update_layout(yaxis_title='', legend_title_text='', yaxis={'categoryorder': 'total descending'})
        show_fig(fig, height=360, key='reg_bar')
        show_df(reg_tbl.style.format({'MAE': '${:,.0f}', 'RMSE': '${:,.0f}', 'R²': '{:.3f}'}))
    with c2:
        which = st.selectbox('Actual vs predicted for', list(A['preds']), index=4)
        sc = pd.DataFrame({'actual': A['test_actual'], 'predicted': A['preds'][which], 'true tier': A['test_tier']})
        fig = px.scatter(sc, x='actual', y='predicted', color='true tier', color_discrete_map=TIER_COLORS,
                         category_orders={'true tier': TIERS}, title=f"{which} — R² {r2_score(sc['actual'], sc['predicted']):.3f}")
        mx = sc[['actual', 'predicted']].max().max() * 1.05
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode='lines', line=dict(dash='dash', color=INK), showlegend=False))
        show_fig(fig, height=430, key='avp')

    st.markdown('#### The deployed per-tier equations ($ per unit)')
    coef = {}
    for t in TIERS:
        rg = M['regs'][t]; nm = [PRETTY[n.split('__', 1)[1]] for n in rg.named_steps['pre'].get_feature_names_out()]
        coef[t] = pd.Series(np.r_[rg.named_steps['model'].intercept_, rg.named_steps['model'].coef_], index=['Intercept'] + nm)
    coef = pd.DataFrame(coef)
    c1, c2 = st.columns([1, 1.3])
    with c1:
        show_df(coef.style.format('{:,.0f}'))
    with c2:
        long = coef.drop('Intercept').reset_index().melt(id_vars='index', var_name='tier', value_name='$ per unit')
        fig = px.bar(long, x='$ per unit', y='index', color='tier', barmode='group', orientation='h', color_discrete_map=TIER_COLORS,
                     title='Same attribute, different price in each tier')
        fig.update_layout(yaxis_title=''); show_fig(fig, height=420, key='coef_bar')

    st.markdown('#### Regularisation lab — Ridge vs Lasso inside one tier')
    c1, c2, c3 = st.columns(3)
    lab_tier = c1.selectbox('Tier', TIERS, index=2)
    lab_kind = c2.radio('Model', ['Lasso', 'Ridge'], horizontal=True)
    lab_alpha = c3.select_slider('alpha', options=[0.01, 0.1, 1, 5, 10, 25, 50, 100, 200, 400, 800, 1600], value=10)

    @st.cache_data(show_spinner=False)
    def alpha_lab(tier_, kind_, alpha_):
        d = DUM[dfc['risk_tier'] == tier_]; y = dfc.loc[d.index, 'annual_charge']
        Xa, Xb, ya, yb = train_test_split(d, y, test_size=0.2, random_state=RANDOM_STATE)
        scl = StandardScaler().fit(Xa)
        mdl = (Lasso(alpha=alpha_, max_iter=50000) if kind_ == 'Lasso' else Ridge(alpha=alpha_)).fit(scl.transform(Xa), ya)
        ols = LinearRegression().fit(scl.transform(Xa), ya)
        return (r2_score(ya, mdl.predict(scl.transform(Xa))), r2_score(yb, mdl.predict(scl.transform(Xb))),
                r2_score(yb, ols.predict(scl.transform(Xb))), pd.Series(mdl.coef_, index=[PRETTY[c] for c in d.columns]))
    tr_r2, te_r2, ols_r2, cf = alpha_lab(lab_tier, lab_kind, lab_alpha)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Train R²', f'{tr_r2:.3f}'); m2.metric('Test R²', f'{te_r2:.3f}', f'{te_r2 - ols_r2:+.3f} vs plain OLS')
    m3.metric('Features kept', int((cf.abs() > 1e-6).sum())); m4.metric('Zeroed', ', '.join(cf.index[cf.abs() <= 1e-6]) or '—')
    fig = px.bar(cf.sort_values(), orientation='h', title=f'{lab_kind} coefficients, {lab_tier} tier, alpha = {lab_alpha} ($ per 1 standard deviation)',
                 color_discrete_sequence=[TIER_COLORS[lab_tier]])
    fig.update_layout(showlegend=False, yaxis_title='', xaxis_title='standardised coefficient ($)')
    show_fig(fig, height=360, key='alpha_bar')

# ---------------------------------------------------------------- Segments & PCA
with tabs[5]:
    st.markdown('#### Unsupervised segments — do customers group naturally?')
    opts = {'Age': 'age', 'BMI': 'bmi', 'Workouts/week': 'exercise_freq', 'Smoker': 'smoker_yes', 'Children': 'children', 'Male': 'sex_male'}
    c1, c2 = st.columns([2, 1])
    feats_lbl = c1.multiselect('Cluster on', list(opts), ['Age', 'BMI', 'Workouts/week', 'Smoker'])
    k = c2.slider('Number of clusters (k)', 2, 8, 3)
    if len(feats_lbl) < 2:
        st.info('Pick at least two features to cluster on.')
    else:
        cols_ = tuple(opts[f] for f in feats_lbl)

        @st.cache_data(show_spinner=False)
        def sil_curve(cols):
            Z = StandardScaler().fit_transform(DUM[list(cols)])
            return pd.DataFrame([{'k': kk, 'silhouette': silhouette_score(Z, KMeans(kk, n_init=10, random_state=RANDOM_STATE).fit_predict(Z))} for kk in range(2, 9)])

        @st.cache_data(show_spinner=False)
        def run_kmeans(cols, kk):
            Z = StandardScaler().fit_transform(DUM[list(cols)])
            return KMeans(kk, n_init=10, random_state=RANDOM_STATE).fit_predict(Z)

        sil = sil_curve(cols_); labels = run_kmeans(cols_, k)
        age0 = dfc['age'].mean()

        def seg_name(r):
            parts = ['older' if r['avg age'] >= age0 + 5 else 'younger' if r['avg age'] <= age0 - 5 else '',
                     'heavier' if r['avg BMI'] >= 30 else 'lean' if r['avg BMI'] < 25 else '',
                     'active' if r['workouts/wk'] >= 4.5 else '', 'smokers' if r['smokers'] > .5 else 'non-smokers']
            s = ' '.join(x for x in parts if x); return s[0].upper() + s[1:]
        prof = dfc.assign(cluster=labels).groupby('cluster').agg(**{'customers': ('age', 'size'), 'avg age': ('age', 'mean'), 'avg BMI': ('bmi', 'mean'),
                                                                   'workouts/wk': ('exercise_freq', 'mean'), 'smokers': ('smoker', lambda s: (s == 'yes').mean()),
                                                                   'avg charge': ('annual_charge', 'mean')})
        nm = prof.apply(seg_name, axis=1)
        prof['segment'] = [f'{n} #{i}' if (nm == n).sum() > 1 else n for i, n in nm.items()]
        seg = pd.Series(labels, index=dfc.index).map(prof['segment'])
        m1, m2, m3 = st.columns(3)
        m1.metric('Silhouette at this k', f"{sil.set_index('k').loc[k, 'silhouette']:.3f}")
        m2.metric('Agreement with risk tiers (ARI)', f"{adjusted_rand_score(dfc['risk_tier'], labels):.3f}")
        m3.metric('Best k by silhouette', int(sil.loc[sil['silhouette'].idxmax(), 'k']))
        c1, c2 = st.columns([1, 1.4])
        with c1:
            fig = px.line(sil, x='k', y='silhouette', markers=True, title='Silhouette by k (flat = no strong natural groups)')
            fig.add_vline(x=k, line_dash='dot', line_color=INK); show_fig(fig, height=320, key='sil')
        with c2:
            mixc = pd.crosstab(seg.rename('segment'), dfc['risk_tier'].rename('tier'), normalize='index')[TIERS]
            fig = px.imshow(mixc, text_auto='.0%', color_continuous_scale='Teal', title='Tier mix inside each segment',
                            labels=dict(x='risk tier', y='', color='share'))
            fig.update_layout(coloraxis_showscale=False); show_fig(fig, height=320, key='mix')
        show_df(prof.set_index('segment').style.format({'avg age': '{:.1f}', 'avg BMI': '{:.1f}', 'workouts/wk': '{:.1f}', 'smokers': '{:.0%}', 'avg charge': '${:,.0f}'}))

        @st.cache_data(show_spinner=False)
        def pca_lda():
            Z = StandardScaler().fit_transform(DUM)
            pca = PCA(random_state=RANDOM_STATE).fit(Z)
            lda = LinearDiscriminantAnalysis(n_components=2).fit(Z, dfc['risk_tier'])
            return pca.transform(Z)[:, :2], pca.explained_variance_ratio_, pd.DataFrame(pca.components_[:3].T, index=[PRETTY[c] for c in DUM.columns], columns=['PC1', 'PC2', 'PC3']), lda.transform(Z)
        P2, evr, load, L2 = pca_lda()
        st.markdown('#### Dimensionality reduction')
        colour_by = st.radio('Colour points by', ['Risk tier', 'Segment'], horizontal=True)
        pdf = pd.DataFrame({'PC1': P2[:, 0], 'PC2': P2[:, 1], 'LD1': L2[:, 0], 'LD2': L2[:, 1], 'Risk tier': dfc['risk_tier'].values, 'Segment': seg.values})
        cmap = TIER_COLORS if colour_by == 'Risk tier' else None
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(pdf, x='PC1', y='PC2', color=colour_by, color_discrete_map=cmap, opacity=.7,
                             category_orders={'Risk tier': TIERS}, title='PCA (unsupervised) — keeps variance, not tier information')
            show_fig(fig, height=400, key='pca')
        with c2:
            fig = px.scatter(pdf, x='LD1', y='LD2', color=colour_by, color_discrete_map=cmap, opacity=.7,
                             category_orders={'Risk tier': TIERS}, title='LDA (supervised) — directions that separate the tiers')
            show_fig(fig, height=400, key='lda')
        c1, c2 = st.columns(2)
        with c1:
            ev = pd.DataFrame({'component': [f'PC{i + 1}' for i in range(len(evr))], 'variance': evr, 'cumulative': np.cumsum(evr)})
            fig = px.bar(ev, x='component', y='variance', title=f'Variance per component — {int(np.argmax(np.cumsum(evr) >= .9) + 1)} of {len(evr)} needed for 90%',
                         color_discrete_sequence=['#2a9d8f'])
            fig.add_trace(go.Scatter(x=ev['component'], y=ev['cumulative'], mode='lines+markers', name='cumulative', line=dict(color='#d8573c')))
            show_fig(fig, height=340, key='evr')
        with c2:
            fig = px.imshow(load, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title='What each component is made of')
            show_fig(fig, height=340, key='load')

# ---------------------------------------------------------------- Formula
with tabs[6]:
    st.markdown('#### The Smart Premium formula')
    st.latex(r'\text{Quoted premium}=\hat{C}_{\text{tier}}(x)\times(1+e+m)\;+\;z\cdot\sigma_{\text{tier}}')
    st.markdown(f"with expense loading **e = {exp_load:.0%}**, profit margin **m = {margin:.0%}**, safety factor **z = {safety:g}** (sidebar). "
                "The tier comes from Stage 1; the scorecard below explains it in one line.")
    st.latex(r'\text{Risk points}=2\cdot[\text{Smoker}]+[\text{BMI}\ge 30]+[\text{Age}\ge 45]+[\text{Age}\ge 58]'
             r'\quad\Rightarrow\quad 0:\text{Low}\;\;|\;\;1\text{–}2:\text{Medium}\;\;|\;\;3{+}:\text{High}')
    st.markdown('**Stage-2 equations** (expected annual charge; bracketed terms are 1 if true, 0 otherwise; region is relative to east):')
    def tex_num(v):
        return f"{abs(v):,.0f}".replace(',', '{,}')
    for t in TIERS:
        s = coef[t].drop('Intercept')
        nums = ''.join(f" {'+' if v >= 0 else '-'} {tex_num(v)}\\,\\text{{{n}}}" for n, v in s.items() if n in ('Age', 'BMI', 'Children', 'Workouts/week'))
        cats = ''.join(f" {'+' if v >= 0 else '-'} {tex_num(v)}\\,[\\text{{{n}}}]" for n, v in s.items() if n not in ('Age', 'BMI', 'Children', 'Workouts/week'))
        st.latex(rf"\begin{{aligned}}\hat{{C}}_{{\text{{{t}}}}} &= {tex_num(coef[t]['Intercept'])}{nums}\\ &\quad{cats}\end{{aligned}}")
    show_df(pd.DataFrame({'typical error σ (CV RMSE)': M['sigma'], 'customers in tier': dfc['risk_tier'].value_counts()}).reindex(TIERS).style.format({'typical error σ (CV RMSE)': '${:,.0f}'}))

    st.markdown('#### Does the formula fix flat pricing? (240 unseen test customers)')
    load_ = 1 + exp_load + margin
    eng = np.array(A['preds']['Two-stage · predicted tier (realistic)']) * load_ + safety * pd.Series(A['test_pred_tier']).map(M['sigma']).values
    fair = pd.DataFrame({'tier': A['test_tier'], 'cost': A['test_actual'], 'Flat price': A['flat_mean'] * load_, 'Smart Premium Engine': eng})
    g = fair.groupby('tier')[['cost', 'Flat price', 'Smart Premium Engine']].sum().reindex(TIERS)
    ratio = pd.DataFrame({'Flat price': g['Flat price'] / g['cost'], 'Smart Premium Engine': g['Smart Premium Engine'] / g['cost']})
    fig = px.bar(ratio.reset_index().melt(id_vars='tier', var_name='pricing', value_name='premium per $1 of claims'), x='tier',
                 y='premium per $1 of claims', color='pricing', barmode='group', text_auto='.2f',
                 color_discrete_map={'Flat price': '#9aa5b1', 'Smart Premium Engine': '#2a9d8f'},
                 title='Premium collected per $1 of claims, by true tier')
    fig.add_hline(y=load_, line_dash='dash', line_color=INK, annotation_text=f'target {load_:.2f}')
    show_fig(fig, height=400, key='fair')
    st.caption('Flat pricing overcharges Low-risk customers and subsidises High-risk ones; the engine brings every tier close to the target. '
               'Raise the safety factor to see the High tier move up to target.')

    st.markdown('#### Worked example — the customer in the sidebar')
    s = coef[Q['tier']]
    st.markdown(f"1. Risk points = **{Q['risk_points']}** → tree tier **{Q['tier']}** ({Q['proba'].get(Q['tier'], 0):.0%} confident).\n"
                f"2. {Q['tier']}-tier equation → expected charge **{money(Q['expected_charge'])}**"
                + (f" (blended across tiers: {money(Q['weighted_charge'])})" if WEIGHTED else '') + ".\n"
                f"3. × (1 + {exp_load:.0%} + {margin:.0%})" + (f" + {safety:g} × {money(M['sigma'][Q['tier']])}" if safety else '') +
                f" = **{money(Q['quoted_premium'])} per year**.")

# ---------------------------------------------------------------- Batch
with tabs[7]:
    st.markdown('#### Score a file of unseen customers')
    st.caption('Needs columns age, sex, bmi, children, smoker, region, exercise_freq (weight_kg optional — used to fill a blank BMI). '
               'Extra columns are ignored. Pricing settings from the sidebar are applied.')
    up = st.file_uploader('Upload a CSV', type='csv', key='batch')
    if up is not None:
        new = pd.read_csv(up)
    elif os.path.exists(SAMPLE_FILE):
        new = pd.read_csv(SAMPLE_FILE); st.info(f'Showing `{SAMPLE_FILE}` until you upload a file.')
    else:
        new = pd.DataFrame([dict(customer_id='DEMO1', age=25, sex='female', bmi=22.0, children=0, smoker='no', region='south', exercise_freq=5),
                            dict(customer_id='DEMO2', age=52, sex='male', bmi=27.0, children=1, smoker='yes', region='east', exercise_freq=2)])
    change = st.selectbox('Apply a change to every customer (for the instructor\'s "what if" questions)',
                          ['No change', 'smoker → yes', 'smoker → no', 'children + 1', 'children + 2', 'age + 5', 'age + 10', 'BMI − 3', 'BMI + 3'])

    def apply_change(cst):
        cst = dict(cst)
        if change.startswith('smoker'):
            cst['smoker'] = change.split('→ ')[1]
        elif change.startswith('children'):
            cst['children'] = min(4, int(cst.get('children', 0)) + int(change[-1]))
        elif change.startswith('age'):
            cst['age'] = min(64, int(cst.get('age', 40)) + int(change.split('+ ')[1]))
        elif change.startswith('BMI'):
            b = pd.to_numeric(cst.get('bmi'), errors='coerce')
            if pd.notna(b):
                cst['bmi'] = float(np.clip(b + (3 if '+' in change else -3), 16, 45))
        return cst
    out = []
    for _, r in new.iterrows():
        a = quote(r.to_dict())
        rec = {'predicted_tier': a['tier'], 'risk_points': a['risk_points'], 'P(High)': round(a['proba'].get('High', 0), 2),
               'expected_charge': round(a['expected_charge']), 'quoted_premium': round(a['quoted_premium'])}
        if change != 'No change':
            b2 = quote(apply_change(r.to_dict()))
            rec.update({'tier_after_change': b2['tier'], 'premium_after_change': round(b2['quoted_premium']),
                        'Δ premium': round(b2['quoted_premium'] - a['quoted_premium'])})
        out.append(rec)
    scored = pd.concat([new.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    k1, k2, k3 = st.columns(3)
    k1.metric('Customers scored', len(scored)); k2.metric('Average quoted premium', money(scored['quoted_premium'].mean()))
    k3.metric('High-risk share', f"{(scored['predicted_tier'] == 'High').mean():.0%}")
    show_df(scored)
    st.download_button('Download scored customers (CSV)', scored.to_csv(index=False), 'healthwise_scored_customers.csv', 'text/csv')
