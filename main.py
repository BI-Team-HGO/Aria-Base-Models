# streamlit run app.py
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from typing import Tuple, List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

# Optional boosters
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

st.set_page_config(page_title="Aria — Base Models", layout="wide")
st.title("Aria — Base Models")

ID_COL = "customer_no"
TARGET_COL = "Donor_Category"
POS_LABEL = "Patron+"
NEG_LABEL = "Under-Patron"

# ============================== Helpers ==============================

def check_required_columns(df: pd.DataFrame) -> Tuple[bool, str]:
    miss = [c for c in [ID_COL, TARGET_COL] if c not in df.columns]
    if miss:
        return False, f"Missing required column(s): {', '.join(miss)}"
    return True, ""

# Coerce numeric-like strings so they don't get OHE'd (e.g. "$1,234", "12%", "(45)")
NUMERIC_LIKE_RE = re.compile(r"^\s*[-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?\s*%?\s*$")
def _clean_numeric_strings(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (123) -> -123
    is_percent = s.str.endswith("%")
    s = s.str.replace(r"[\$,]", "", regex=True)
    out = pd.to_numeric(s.str.replace("%", "", regex=False), errors="coerce")
    out[is_percent] = out[is_percent] / 100.0
    return out

def coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if c in (ID_COL, TARGET_COL):
            continue
        ser = df[c]
        if pd.api.types.is_numeric_dtype(ser):
            continue
        nonnull = ser.dropna().astype(str).str.strip()
        if nonnull.empty:
            continue
        looks_numeric = nonnull.str.match(NUMERIC_LIKE_RE, na=False)
        if looks_numeric.mean() >= 0.85:
            df[c] = _clean_numeric_strings(ser)
    if ID_COL in df.columns:
        df[ID_COL] = df[ID_COL].astype(str)
    return df

def select_features(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    - Numerics: numeric dtypes (exclude ID)
    - Categoricals: non-numerics (exclude ID/TARGET) with 2–5 unique non-null values
    """
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if ID_COL in num_cols: num_cols.remove(ID_COL)

    cat_candidates = df.select_dtypes(exclude=["number"]).columns.tolist()
    cat_candidates = [c for c in cat_candidates if c not in (ID_COL, TARGET_COL)]

    kept_cat, dropped_cat = [], []
    for c in cat_candidates:
        nun = df[c].nunique(dropna=True)
        if 2 <= nun <= 5:
            kept_cat.append(c)
        else:
            dropped_cat.append(c)
    return num_cols, kept_cat, dropped_cat

def impute_numeric_under_patron(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric NaNs -> mean among Under-Patron (fallback: overall mean)."""
    df = df.copy()
    under = df[df[TARGET_COL] == NEG_LABEL]
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols if c != ID_COL]
    for c in num_cols:
        m = under[c].mean(skipna=True)
        if pd.isna(m): m = df[c].mean(skipna=True)
        df[c] = df[c].fillna(m)
    return df

def impute_categoricals_mode(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Categorical NaNs -> global mode."""
    df = df.copy()
    for c in cat_cols:
        if c in df.columns and not df[c].dropna().empty:
            mode_val = df[c].mode(dropna=True)[0]
            df[c] = df[c].fillna(mode_val)
    return df

def build_tabular_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(("num", MinMaxScaler(), num_cols))
    if cat_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def build_haversine_preprocessor(lat_col: str, lon_col: str) -> Pipeline:
    """
    Select [lat, lon] and convert degrees -> radians so KNN(haversine) works.
    """
    selector = ColumnTransformer(
        transformers=[("ll", "passthrough", [lat_col, lon_col])],
        remainder="drop",
    )
    to_rad = FunctionTransformer(np.radians, feature_names_out="one-to-one")
    return Pipeline([("select_ll", selector), ("to_rad", to_rad)])

def get_feature_names_safe(pre) -> List[str]:
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        pass
    if isinstance(pre, ColumnTransformer):
        names = []
        for _, trans, cols in pre.transformers_:
            if hasattr(trans, "get_feature_names_out"):
                try:
                    fn = trans.get_feature_names_out(cols)
                    names.extend(list(fn)); continue
                except Exception:
                    pass
            names.extend(list(cols))
        return names
    if isinstance(pre, Pipeline):
        for _, step in pre.steps:
            try:
                return list(step.get_feature_names_out())
            except Exception:
                if isinstance(step, ColumnTransformer):
                    names = []
                    for _, trans, cols in step.transformers_:
                        if hasattr(trans, "get_feature_names_out"):
                            try:
                                fn = trans.get_feature_names_out(cols)
                                names.extend(list(fn)); continue
                            except Exception:
                                pass
                        names.extend(list(cols))
                    return names
        return []
    return []

def class_weight_params_for_pos(y: pd.Series):
    classes, _ = np.unique(y, return_counts=True)
    if POS_LABEL in classes and len(classes) >= 2:
        pos = (y == POS_LABEL).sum()
        neg = (y != POS_LABEL).sum()
        spw = float(neg) / max(float(pos), 1.0)
        return {"class_weight": "balanced"}, spw
    return {"class_weight": "balanced"}, None

def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    st.pyplot(fig)

# ============================== UI ==============================

left, right = st.columns(2)
with left:
    uploaded = st.file_uploader("Upload CSV (must include 'customer_no' & 'Donor_Category')", type=["csv"])
with right:
    model_choice = st.selectbox(
        "Model",
        [
            "Logistic Regression",
            "Random Forest",
            "AdaBoost",
            "KNN",
            "KNN (Haversine lat/lon)",
            ("XGBoost" if HAS_XGB else "XGBoost (unavailable)"),
            ("LightGBM" if HAS_LGBM else "LightGBM (unavailable)"),
            "Naive Bayes",
        ],
    )

if uploaded is None:
    st.info("Upload a CSV to begin."); st.stop()

# Load
try:
    df_raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}"); st.stop()

ok, msg = check_required_columns(df_raw)
if not ok:
    st.error(msg); st.stop()

st.subheader("Raw preview")
st.dataframe(df_raw.head())

# Coerce numeric-likes, select features, impute
df = coerce_numeric_like(df_raw)
num_cols_all, cat_small, cat_dropped = select_features(df)

# --- NEW: capture imputation maps used during training so we can reuse at predict time ---
numeric_impute_map = {}
under_for_map = df[df[TARGET_COL] == NEG_LABEL]
num_cols_tmp = df.select_dtypes(include=["number"]).columns.tolist()
num_cols_tmp = [c for c in num_cols_tmp if c != ID_COL]
for c in num_cols_tmp:
    m = under_for_map[c].mean(skipna=True)
    if pd.isna(m): m = df[c].mean(skipna=True)
    numeric_impute_map[c] = m

df = impute_numeric_under_patron(df)

categorical_impute_map = {}
for c in cat_small:
    if c in df.columns and not df[c].dropna().empty:
        categorical_impute_map[c] = df[c].mode(dropna=True)[0]

df = impute_categoricals_mode(df, cat_small)

# Build X/y and final lists (exclude ID)
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
id_series = df[ID_COL].astype(str)

num_cols = [c for c in num_cols_all if c in X.columns and c != ID_COL]
cat_cols = [c for c in cat_small if c in X.columns and c != ID_COL]

st.subheader("Feature sets")
st.write(f"- Numeric columns used: {len(num_cols)}")
st.write(f"- Categorical columns used (2–5 unique): {len(cat_cols)}  → {cat_cols}")
if cat_dropped:
    st.caption(f"Dropped categorical columns (not 2–5 uniques): {cat_dropped}")

# Haversine: choose lat/lon columns
lat_col = lon_col = None
if model_choice == "KNN (Haversine lat/lon)":
    all_cols = X.columns.tolist()
    default_lat = "latitude" if "latitude" in all_cols else (next((c for c in all_cols if "lat" in c.lower()), None))
    default_lon = "longitude" if "longitude" in all_cols else (next((c for c in all_cols if "lon" in c.lower() or "lng" in c.lower()), None))
    if not default_lat and all_cols: default_lat = all_cols[0]
    if not default_lon and all_cols: default_lon = all_cols[0]
    lat_col = st.selectbox("Latitude column", options=all_cols, index=(all_cols.index(default_lat) if default_lat in all_cols else 0))
    lon_col = st.selectbox("Longitude column", options=all_cols, index=(all_cols.index(default_lon) if default_lon in all_cols else 0))

# Preprocessor
if model_choice == "KNN (Haversine lat/lon)":
    pre = build_haversine_preprocessor(lat_col, lon_col)
else:
    pre = build_tabular_preprocessor(num_cols, cat_cols)

# Test split slider: 5% to 15%
test_size = st.slider("Test size", 0.05, 0.15, 0.10, 0.01)
try:
    X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
        X, y, id_series, test_size=test_size, random_state=42, stratify=y
    )
except Exception:
    X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
        X, y, id_series, test_size=test_size, random_state=42
    )

# Complexity control (no slider for NB)
complexity = None
if model_choice == "Naive Bayes":
    st.caption("Naive Bayes has no complexity knob.")
else:
    if model_choice == "Logistic Regression":
        comp_label, min_c, max_c, default_c = "Max iterations", 200, 2000, 1000
    elif model_choice == "Random Forest":
        comp_label, min_c, max_c, default_c = "Number of trees (n_estimators)", 100, 1000, 300
    elif model_choice == "AdaBoost":
        comp_label, min_c, max_c, default_c = "Number of estimators", 50, 800, 200
    elif model_choice == "KNN":
        comp_label, min_c, max_c, default_c = "Number of neighbors (k)", 3, 75, 15
    elif model_choice == "KNN (Haversine lat/lon)":
        comp_label, min_c, max_c, default_c = "Number of neighbors (k)", 3, 75, 25
    elif "XGBoost" in model_choice:
        comp_label, min_c, max_c, default_c = "Number of trees (n_estimators)", 100, 2000, 400
    elif "LightGBM" in model_choice:
        comp_label, min_c, max_c, default_c = "Number of trees (n_estimators)", 100, 2000, 400
    else:
        comp_label = min_c = max_c = default_c = None

    if comp_label is not None:
        step = 1 if max_c <= 1000 else 50
        complexity = st.slider(comp_label, min_value=min_c, max_value=max_c, value=default_c, step=step)

# ===== Threshold: initial default from prevalence (no auto-apply) =====
if "desired_thr" not in st.session_state:
    try:
        pos_rate = float((y == POS_LABEL).mean())
    except Exception:
        pos_rate = 0.5
    st.session_state.desired_thr = 0.25 if pos_rate <= 0.10 else (0.35 if pos_rate <= 0.20 else 0.50)

st.subheader(f"Decision threshold for '{POS_LABEL}'")
st.session_state.desired_thr = st.slider("Threshold", 0.00, 1.00, float(st.session_state.desired_thr), 0.01)

# Model factory (+ XGBoost label encoding fix)
def cw_params_for_pos(y):
    classes, _ = np.unique(y, return_counts=True)
    if POS_LABEL in classes and len(classes) >= 2:
        pos = (y == POS_LABEL).sum()
        neg = (y != POS_LABEL).sum()
        spw = float(neg) / max(float(pos), 1.0)
        return {"class_weight": "balanced"}, spw
    return {"class_weight": "balanced"}, None

cw_params, spw = cw_params_for_pos(y_tr)

if model_choice == "Logistic Regression":
    clf = LogisticRegression(max_iter=int(complexity), **cw_params)
elif model_choice == "Random Forest":
    clf = RandomForestClassifier(
        n_estimators=int(complexity),
        class_weight="balanced_subsample",
        random_state=42
    )
elif model_choice == "AdaBoost":
    clf = AdaBoostClassifier(n_estimators=int(complexity), random_state=42)
elif model_choice == "KNN":
    clf = KNeighborsClassifier(n_neighbors=int(complexity), weights="distance")
elif model_choice == "KNN (Haversine lat/lon)":
    clf = KNeighborsClassifier(
        n_neighbors=int(complexity),
        algorithm="ball_tree",
        metric="haversine",
        weights="distance"
    )
elif model_choice.startswith("XGBoost"):
    if not HAS_XGB:
        st.error("XGBoost not installed."); st.stop()
    clf = XGBClassifier(
        n_estimators=int(complexity), learning_rate=0.1, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss", n_jobs=4,
        scale_pos_weight=(spw if spw is not None else 1.0),
        tree_method="hist", random_state=42
    )
elif model_choice.startswith("LightGBM"):
    if not HAS_LGBM:
        st.error("LightGBM not installed."); st.stop()
    clf = LGBMClassifier(
        n_estimators=int(complexity), learning_rate=0.1, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, class_weight="balanced",
        random_state=42
    )
else:
    clf = GaussianNB()

pipe = Pipeline([("prep", pre), ("clf", clf)])

# Cache artifacts (so threshold updates don't retrain)
for k in ["pipe", "probs_te", "classes_", "y_te", "ids_te", "feature_names",
          "suggested_thr", "suggested_prec", "suggested_rec",
          "numeric_impute_map", "categorical_impute_map", "required_cols"]:
    st.session_state.setdefault(k, None)

if st.button("Train (optimize for Patron+)"):
    # Haversine guard
    if model_choice == "KNN (Haversine lat/lon)":
        for c in [lat_col, lon_col]:
            if c not in X.columns:
                st.error("Please choose valid latitude and longitude columns."); st.stop()
        if X[[lat_col, lon_col]].isnull().any().any():
            st.warning("Rows with missing lat/lon may hurt Haversine KNN.")

    # -------- FIT (XGB fix: encode labels to 0/1 just for XGB) --------
    if model_choice.startswith("XGBoost"):
        y_tr_bin = (y_tr == POS_LABEL).astype(int)
        y_te_bin = (y_te == POS_LABEL).astype(int)
        pipe.fit(X_tr, y_tr_bin)
    else:
        pipe.fit(X_tr, y_tr)

    st.session_state.pipe = pipe

    # -------- PROBS + CLASSES (normalize to string labels) --------
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        probs_te = pipe.predict_proba(X_te)
        # For XGB, classes_ are [0,1]; remap to [NEG_LABEL, POS_LABEL] for the UI
        if model_choice.startswith("XGBoost"):
            if probs_te.ndim == 1:
                probs_te = np.vstack([1 - probs_te, probs_te]).T
            classes_display = [NEG_LABEL, POS_LABEL]
        else:
            classes_model = pipe.named_steps["clf"].classes_.tolist()
            classes_display = classes_model
        st.session_state.probs_te = probs_te
        st.session_state.classes_ = classes_display
    else:
        st.error("Selected model does not support predict_proba; choose a different model.")
        st.stop()

    # Feature names after preprocessing
    try:
        st.session_state.feature_names = get_feature_names_safe(pipe.named_steps["prep"])
    except Exception:
        st.session_state.feature_names = num_cols + cat_cols

    st.session_state.y_te = y_te
    st.session_state.ids_te = ids_te

    # --- NEW: remember imputation maps & required cols for prediction ---
    st.session_state.numeric_impute_map = numeric_impute_map
    st.session_state.categorical_impute_map = categorical_impute_map
    if model_choice == "KNN (Haversine lat/lon)":
        st.session_state.required_cols = [lat_col, lon_col]
    else:
        st.session_state.required_cols = (num_cols or []) + (cat_cols or [])

    # ---- GREEN BANNER: F1-optimal threshold (read-only) ----
    try:
        if POS_LABEL in st.session_state.classes_:
            pos_idx = st.session_state.classes_.index(POS_LABEL)
        else:
            pos_idx = int(np.argmax(np.mean(st.session_state.probs_te, axis=0)))
        y_true_pos = (y_te == POS_LABEL).astype(int).values
        pos_probs = st.session_state.probs_te[:, pos_idx]
        prec, rec, thr = precision_recall_curve(y_true_pos, pos_probs)
        f1 = 2 * (prec[:-1] * rec[:-1]) / np.maximum(prec[:-1] + rec[:-1], 1e-12)
        best_idx = int(np.nanargmax(f1))
        st.session_state.suggested_thr = float(thr[best_idx])
        st.session_state.suggested_prec = float(prec[best_idx])
        st.session_state.suggested_rec  = float(rec[best_idx])

        st.success(
            f"Optimal precision/recall trade-off (F1 for **{POS_LABEL}**): "
            f"**threshold = {st.session_state.suggested_thr:.3f}**, "
            f"precision = **{st.session_state.suggested_prec:.2f}**, "
            f"recall = **{st.session_state.suggested_rec:.2f}**"
        )
    except Exception:
        st.info("Could not compute an F1-optimal threshold (dataset may be too small or degenerate).")

# ------------------- Metrics (use string labels & slider) -------------------
def render_metrics_live():
    if st.session_state.probs_te is None or st.session_state.pipe is None:
        st.info("Train a model to see metrics here.")
        return

    thr = float(st.session_state.desired_thr)
    classes = st.session_state.classes_
    y_te_local = st.session_state.y_te

    if POS_LABEL in classes:
        pos_idx = classes.index(POS_LABEL)
    else:
        pos_idx = np.argmax(np.mean(st.session_state.probs_te, axis=0))

    pos_probs = st.session_state.probs_te[:, pos_idx]
    y_pred = np.where(pos_probs >= thr, POS_LABEL, NEG_LABEL)

    st.subheader(f"Metrics at threshold = {thr:.2f}")
    st.text(classification_report(y_te_local, y_pred, labels=[NEG_LABEL, POS_LABEL], zero_division=0))

    cm = confusion_matrix(y_te_local, y_pred, labels=[NEG_LABEL, POS_LABEL])
    plot_confusion(cm, [NEG_LABEL, POS_LABEL])

    # Top features (normalized) – KNN/Haversine won't expose importances (that’s fine)
    try:
        clf_fitted = st.session_state.pipe.named_steps["clf"]
        feat_names = st.session_state.feature_names or []
        vals, kind = None, ""
        if hasattr(clf_fitted, "feature_importances_"):
            vals = clf_fitted.feature_importances_; kind = "importance"
        elif hasattr(clf_fitted, "coef_"):
            coefs = clf_fitted.coef_
            vals = np.mean(np.abs(coefs), axis=0) if coefs.ndim > 1 else np.abs(coefs[0])
            kind = "weight (|coef|)"
        if vals is not None and len(feat_names) == len(vals):
            s = pd.Series(vals, index=feat_names).sort_values(ascending=False).head(20)
            s = s / s.sum()
            fig, ax = plt.subplots(figsize=(7, 5))
            s.iloc[::-1].plot(kind="barh", ax=ax)
            ax.set_xlabel(f"{kind} (normalized)"); ax.set_title("Top 20 Features")
            st.pyplot(fig)
            st.dataframe(s.rename("value").to_frame())
        else:
            if not hasattr(clf_fitted, "feature_importances_") and not hasattr(clf_fitted, "coef_"):
                st.info("This model does not expose importances/coefficients (e.g., KNN/Haversine).")
    except Exception as e:
        st.warning(f"Could not render feature importances: {e}")

render_metrics_live()

# ------------------- Transformed preview -------------------
with st.expander("Show transformed features head (MinMax + OHE or lat/lon→radians)"):
    try:
        pre_fitted = st.session_state.pipe.named_steps["prep"] if st.session_state.pipe is not None else pre.fit(X_tr)
        expected_cols = []
        if isinstance(pre_fitted, ColumnTransformer):
            for _, _, cols in pre_fitted.transformers_:
                expected_cols.extend(list(cols))
        elif isinstance(pre_fitted, Pipeline):
            inner = None
            for _, step in pre_fitted.steps:
                if isinstance(step, ColumnTransformer):
                    inner = step; break
            if inner is not None:
                for _, _, cols in inner.transformers_:
                    expected_cols.extend(list(cols))
            else:
                expected_cols = X.columns.tolist()
        else:
            expected_cols = X.columns.tolist()

        X_sample = X[expected_cols].head(10).copy()
        X_trans = pre_fitted.transform(X_sample)
        cols = get_feature_names_safe(pre_fitted)
        if X_trans.shape[1] != len(cols):
            cols = [f"f{i}" for i in range(X_trans.shape[1])]
        df_trans = pd.DataFrame(X_trans, columns=cols)
        st.dataframe(df_trans)
    except Exception as e:
        st.info(f"Transform preview not available: {e}")

# ------------------- Download bundle -------------------
st.subheader("Download optimized model")
if st.session_state.pipe is not None:
    bundle = {
        "pipeline": st.session_state.pipe,
        "threshold": float(st.session_state.desired_thr),
        "positive_label": POS_LABEL,
        "classes_": st.session_state.classes_,  # always string labels for UI
        "metadata": {
            "id_column": ID_COL,
            "target_column": TARGET_COL,
            "model_choice": model_choice,
            "test_size": float(test_size),
            "numeric_columns": num_cols,
            "categorical_columns_used": cat_cols,          # only 2–5 uniques
            "dropped_categorical_columns": cat_dropped,    # everything else
            "haversine_lat_col": lat_col if model_choice == "KNN (Haversine lat/lon)" else None,
            "haversine_lon_col": lon_col if model_choice == "KNN (Haversine lat/lon)" else None,
            "xgboost_label_mapping": {NEG_LABEL: 0, POS_LABEL: 1} if model_choice.startswith("XGBoost") else None,
        }
    }
    buf = io.BytesIO()
    joblib.dump(bundle, buf); buf.seek(0)
    st.download_button(
        "Download .pkl bundle",
        data=buf,
        file_name="aria_patronplus.pkl",
        mime="application/octet-stream",
        help="Trained pipeline + your decision threshold + class mapping + feature metadata."
    )
else:
    st.info("Train a model to enable download.")

# ================================
# ### PREDICTION SECTION (NEW) ###
# ================================
st.subheader("Predict with current model")
if st.session_state.pipe is None:
    st.info("Train a model above to enable predictions.")
else:
    pred_csv = st.file_uploader(
        "Upload CSV to score (must include 'customer_no'; target not required)",
        type=["csv"],
        key="pred_uploader",
        help="We’ll validate columns against the last trained model."
    )

    if st.button("Predict with last trained model"):
        if pred_csv is None:
            st.error("Please upload a CSV to score."); 
        else:
            try:
                df_pred_raw = pd.read_csv(pred_csv)
            except Exception as e:
                st.error(f"Could not read prediction CSV: {e}")
            else:
                # Validate ID column
                if ID_COL not in df_pred_raw.columns:
                    st.error(f"Missing required ID column '{ID_COL}'.")
                else:
                    # Prepare like training: coerce + impute using stored maps
                    df_pred = coerce_numeric_like(df_pred_raw.copy())

                    # Apply saved imputation values from training
                    num_map = st.session_state.get("numeric_impute_map", {}) or {}
                    for c, m in num_map.items():
                        if c in df_pred.columns:
                            df_pred[c] = df_pred[c].fillna(m)

                    cat_map = st.session_state.get("categorical_impute_map", {}) or {}
                    for c, mv in cat_map.items():
                        if c in df_pred.columns:
                            df_pred[c] = df_pred[c].fillna(mv)

                    # Confirm required feature columns exist
                    required_cols = st.session_state.get("required_cols", []) or []
                    missing = [c for c in required_cols if c not in df_pred.columns]
                    if missing:
                        st.error(f"Missing required columns for this model: {missing}")
                    else:
                        # Build X for pipeline (drop target if present)
                        X_pred_full = df_pred.drop(columns=[TARGET_COL], errors="ignore")

                        # Predict probabilities
                        try:
                            probs = st.session_state.pipe.predict_proba(X_pred_full)
                        except Exception as e:
                            st.error(f"Prediction failed. Check column types match training set. Details: {e}")
                        else:
                            # Find index for Patron+ prob
                            classes = st.session_state.classes_ or []
                            if POS_LABEL in classes:
                                pos_idx = classes.index(POS_LABEL)
                            else:
                                pos_idx = int(np.argmax(np.mean(probs, axis=0)))

                            # Ensure 2D
                            if probs.ndim == 1:
                                probs = np.vstack([1 - probs, probs]).T

                            pos_probs = probs[:, pos_idx]
                            thr = float(st.session_state.desired_thr)
                            preds = np.where(pos_probs >= thr, POS_LABEL, NEG_LABEL)

                            out = pd.DataFrame({
                                ID_COL: df_pred_raw[ID_COL].astype(str).values,
                                f"proba_{POS_LABEL}": pos_probs,
                                "prediction": preds
                            })

                            st.success("Predictions ready.")
                            st.dataframe(out.head(20))

                            # Download CSV
                            pred_bytes = out.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download predictions CSV",
                                data=pred_bytes,
                                file_name="aria_predictions.csv",
                                mime="text/csv",
                                help="Includes customer_no, proba for Patron+, and the final prediction using your current threshold."
                            )
