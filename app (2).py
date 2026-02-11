

# ============================================================
# DATAFORGE AI – FULL STREAMLIT APPLICATION
# React + Flask functionality replicated 1:1
# ============================================================

import streamlit as st

st.set_page_config(
    page_title="STATYX",
    layout="wide",
    initial_sidebar_state="expanded"
)
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import time
from mpl_toolkits.mplot3d import Axes3D
import textwrap


from sentence_transformers import SentenceTransformer, util
from stats_tests import TEST_REGISTRY


# ============================================================
# MODERN UI / THEME INJECTION (NO LOGIC CHANGE)
# ============================================================

st.markdown("""
<style>

/* ===== RESET STREAMLIT PADDING ===== */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
    max-width: 1200px;
    margin: auto;
}

/* ===== GLOBAL ===== */
html, body {
    background-color: #f8fafc;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #0f172a;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    padding-top: 1.5rem;
}

section[data-testid="stSidebar"] h1 {
    color: #38bdf8;
    text-align: left;
    font-size: 1.2rem;
    padding-left: 0.8rem;
}

/* Sidebar radio */
.stRadio label {
    padding: 10px 12px;
    border-radius: 10px;
    font-size: 0.95rem;
    transition: all 0.2s ease;
}

.stRadio label:hover {
    background-color: rgba(56,189,248,0.15);
}

/* ===== CARDS ===== */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 22px 26px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    margin-bottom: 24px;
}

/* ===== HERO CARD ===== */
.hero {
    padding: 28px 34px;
}

.hero h1 {
    font-size: 2rem;
    margin-bottom: 0.2rem;
}

.hero h3 {
    font-weight: 500;
    color: #475569;
    margin-bottom: 1rem;
}

/* ===== FEATURE CARDS ===== */
.feature {
    text-align: left;
}

.feature b {
    font-size: 1rem;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #38bdf8);
    color: white;
    border-radius: 10px;
    padding: 0.55rem 1.4rem;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    filter: brightness(1.05);
}

/* ===== TABLES ===== */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* ==================================================
   SIDEBAR – FINAL FONT & VISIBILITY UPGRADE
   ================================================== */

/* App title */
section[data-testid="stSidebar"] h1 {
    color: #ffffff !important;
    font-size: 1.35rem;
    font-weight: 700;
}

/* "Navigation" label */
section[data-testid="stSidebar"] p {
    color: #cbd5f5 !important;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 1rem;
}

/* Radio item text */
section[data-testid="stSidebar"] label {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 500;
    padding: 12px 14px;
    border-radius: 12px;
    transition: all 0.2s ease-in-out;
}

/* Hover effect */
section[data-testid="stSidebar"] label:hover {
    background: rgba(255,255,255,0.12);
    font-weight: 600;
}

/* Selected (active) tab */
section[data-testid="stSidebar"] input:checked + div {
    background: linear-gradient(
        90deg,
        rgba(56,189,248,0.45),
        rgba(56,189,248,0.08)
    );
    border-left: 4px solid #38bdf8;
    font-weight: 700;
    color: #ffffff !important;
}

/* Radio circle visibility */
section[data-testid="stSidebar"] svg {
    fill: #ffffff !important;
}

/* Reduce visual clutter */
section[data-testid="stSidebar"] .stRadio {
    gap: 0.35rem;
}

</style>
""", unsafe_allow_html=True)

st.session_state.setdefault("initialized", True)

# ============================================================
# STREAMLIT CONFIG
# ============================================================

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "df" not in st.session_state:
    st.session_state.df = None

if "preview" not in st.session_state:
    st.session_state.preview = None

if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

if "stats" not in st.session_state:
    st.session_state.stats = None

# ============================================================
# CLEANING SESSION STATE (MUST EXIST BEFORE USE)
# ============================================================
if "clean_issues" not in st.session_state:
    st.session_state.clean_issues = []

if "clean_preview" not in st.session_state:
    st.session_state.clean_preview = None

if "clean_download" not in st.session_state:
    st.session_state.clean_download = None

if "clean_actions" not in st.session_state:
    st.session_state.clean_actions = {}



# ============================================================
# AI MODEL (E5 – SAME AS FLASK BACKEND)
# ============================================================
@st.cache_resource
def load_e5_model():
    return SentenceTransformer("intfloat/e5-base")

e5_model = load_e5_model()



TEST_DESCRIPTIONS = {
    "independent_t_test": "compare mean of a continuous variable between two independent groups",
    "paired_t_test": "compare mean before and after intervention on same subjects",
    "one_sample_t_test": "compare mean against known population value",
    "anova": "compare mean across more than two groups",
    "anova_rm": "compare repeated measurements",
    "mann_whitney_u": "nonparametric comparison between two independent groups",
    "kruskal_wallis": "nonparametric comparison across multiple groups",
    "chi_square": "association between two categorical variables",
    "fisher_exact": "association between two binary variables",
    "chisquare_gof": "goodness of fit test",
    "mcnemar_test": "paired categorical association",
    "cochran_q": "compare proportions across repeated groups",
    "pearson_correlation": "linear correlation",
    "spearman_correlation": "rank based correlation",
    "kendall_tau": "ordinal association",
    "mutual_information": "nonlinear dependency",
    "linear_regression": "predict continuous outcome",
    "logistic_regression": "predict binary outcome",
    "poisson_regression": "model count outcome",
    "probit_regression": "binary outcome with probit",
    "cox_regression": "survival analysis",
    "kaplan_meier": "survival probability",
    "shapiro_test": "normality test",
    "ks_test": "distribution test",
    "levene_test": "variance equality",
    "bartlett_test": "variance homogeneity",
    "jarque_bera": "skewness & kurtosis",
    "variance_ratio": "compare variances",
    "binomial_test": "test proportion",
    "z_test_proportion": "z test for proportion",
    "two_proportion_ztest": "compare proportions",
    "roc_auc": "classification performance",
    "cohens_d": "effect size",
    "hedges_g": "bias corrected effect size",
    "phi_coefficient": "binary association",
    "cramers_v": "categorical association strength",
    "odds_ratio": "odds comparison",
    "relative_risk": "risk comparison"
}

TEST_NAMES = list(TEST_DESCRIPTIONS.keys())
TEST_TEXTS = [f"passage: {TEST_DESCRIPTIONS[t]}" for t in TEST_NAMES]
@st.cache_resource
def compute_test_embeddings():
    return e5_model.encode(TEST_TEXTS, convert_to_tensor=True)

TEST_EMBEDDINGS = compute_test_embeddings()



# ============================================================
# HELPER FUNCTIONS (SAME AS FLASK BACKEND)
# ============================================================
def infer_column_type(series):
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return "categorical"

    def looks_numeric(x):
        try:
            float(x)
            return True
        except:
            return False

    numeric_ratio = non_null.apply(looks_numeric).mean()
    return "numeric" if numeric_ratio >= 0.6 else "categorical"


def infer_columns_from_objective(df, objective):
    col_names = df.columns.tolist()

    col_embs = e5_model.encode(
        [f"column: {c}" for c in col_names],
        convert_to_tensor=True
    )

    query_emb = e5_model.encode(
        f"query: {objective}",
        convert_to_tensor=True
    )

    scores = util.cos_sim(query_emb, col_embs)[0]
    ranked = sorted(zip(col_names, scores.tolist()), key=lambda x: x[1], reverse=True)

    target_col, group_col = None, None
    for col, _ in ranked:
        if target_col is None and df[col].nunique() == 2:
            target_col = col
        elif group_col is None and df[col].nunique() < 10:
            group_col = col

    return target_col, group_col

def is_numeric_like(x):
    if x is None:
        return False
    if isinstance(x, (int, float)) and not pd.isna(x):
        return True
    try:
        float(str(x).strip())
        return True
    except Exception:
        return False

def csv_download(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def build_overall_categorical_table(df):
    cat_cols = df.select_dtypes(include="object").columns
    rows = ["count", "unique", "top", "freq", "top_percentage"]
    summary = pd.DataFrame(index=rows)

    for col in cat_cols:
        series = df[col].dropna()
        counts = series.value_counts()

        top_val = counts.idxmax() if not counts.empty else None
        top_freq = counts.max() if not counts.empty else 0
        top_pct = (top_freq / len(series) * 100) if len(series) > 0 else 0

        summary[col] = [
            len(series),
            series.nunique(),
            top_val,
            top_freq,
            round(top_pct, 1)
        ]

    return summary

def plot_excel_style_3d_bar(series, column_name):

    counts = series.value_counts()
    labels = counts.index.astype(str)
    values = counts.values

    # Wrap long labels into multiple lines
    wrapped_labels = [
        "\n".join(textwrap.wrap(label, width=12))
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    bars = ax.bar(
        wrapped_labels,
        values,
        width=0.45,
        color="#4F76C7",
        edgecolor="black",
        linewidth=0.8
    )

    # Value labels on top
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.3,
            f"{val}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )

    # Axis labels
    ax.set_ylabel("Cases (%)", fontsize=8)
    ax.set_xlabel(column_name.replace("_", " ").title(), fontsize=8)

    # Title
    ax.set_title(
        f"{column_name.replace('_', ' ').title()}",
        fontsize=9,
        pad=6
    )

    # Grid and spines
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tick formatting
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)

    # Extra bottom space for wrapped labels
    plt.subplots_adjust(bottom=0.28)

    return fig

def smart_categorical_plot(series, column_name, top_n=10):
    counts = series.value_counts(dropna=True)
    n_categories = len(counts)

    # 🚫 Too many categories → table only
    if n_categories > 20:
        return None, "table_only"

    # 🧠 High-cardinality → Top-N + Others
    if n_categories > top_n:
        top_counts = counts.head(top_n)
        others = counts.iloc[top_n:].sum()
        top_counts["Others"] = others
        counts = top_counts

    labels = counts.index.astype(str)
    values = counts.values

    horizontal = (
        n_categories > 6 or
        max(len(lbl) for lbl in labels) > 12
    )

    fig_size = (4.8, 3.2) if horizontal else (4.0, 3.0)
    fig, ax = plt.subplots(figsize=fig_size)

    if horizontal:
        bars = ax.barh(labels, values, color="#4F76C7", edgecolor="black")
        ax.set_xlabel("Count", fontsize=8)
    else:
        bars = ax.bar(labels, values, color="#4F76C7", edgecolor="black")
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis="x", rotation=30)

    # Value labels
    for bar, val in zip(bars, values):
        if horizontal:
            ax.text(val + 0.1,
                    bar.get_y() + bar.get_height()/2,
                    f"{val}", va="center", fontsize=7, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.2,
                    f"{val}", ha="center", fontsize=7, fontweight="bold")

    ax.set_title(
        f"Distribution of {column_name.replace('_', ' ').title()}",
        fontsize=9
    )

    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y" if not horizontal else "x",
            linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, "plot"


# ============================================================
# SIDEBAR NAVIGATION (MATCHES REACT TABS)
# ============================================================
st.sidebar.title("⚡ STATYX")
tab = st.sidebar.radio(
    "Navigation",
    [
        "Landing",
        "Upload & Preview",
        "Data Cleaning",
        "Statistics",
        "Visualizations",
        "AI Objective Analysis",
        "Cross Tabulation"
    ]
)

# ============================================================
# LANDING PAGE
# ============================================================
if tab == "Landing":

    components.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {
                margin: 0;
                font-family: Inter, Segoe UI, sans-serif;
                background: #f8fafc;
            }

            .hero {
                background: radial-gradient(circle at top right, #38bdf8, #2563eb 40%, #020617);
                padding: 4rem 4rem 3.5rem;
                border-radius: 28px;
                color: white;
                box-shadow: 0 30px 80px rgba(0,0,0,0.35);
            }

            .title {
                font-size: 3rem;
                font-weight: 800;
                letter-spacing: -0.04em;
            }

            .subtitle {
                font-size: 1.35rem;
                margin-top: 0.5rem;
                color: #e0f2fe;
            }

            .tagline {
                margin-top: 1.2rem;
                font-size: 1.05rem;
                color: #c7d2fe;
                line-height: 1.5;
            }

            .cta-row {
                margin-top: 2.2rem;
                display: flex;
                gap: 1rem;
                flex-wrap: wrap;
            }

            .cta {
                padding: 0.7rem 1.8rem;
                border-radius: 999px;
                font-weight: 600;
                border: none;
                cursor: pointer;
                font-size: 0.95rem;
            }

            .cta.primary {
                background: white;
                color: #020617;
                box-shadow: 0 10px 25px rgba(0,0,0,0.25);
            }

            .cta.secondary {
                background: transparent;
                color: white;
                border: 1px solid rgba(255,255,255,0.35);
            }

            .features {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 1.8rem;
                margin-top: 3rem;
            }

            .card {
                background: white;
                border-radius: 22px;
                padding: 1.8rem;
                box-shadow: 0 18px 50px rgba(15,23,42,0.12);
            }

            .icon {
                font-size: 1.8rem;
                margin-bottom: 0.5rem;
            }

            .card-title {
                font-weight: 700;
                margin-bottom: 0.4rem;
                font-size: 1.05rem;
            }

            .card-desc {
                font-size: 0.95rem;
                color: #334155;
                line-height: 1.5;
            }

            .value {
                margin-top: 3rem;
                background: #020617;
                color: #cbd5f5;
                border-radius: 22px;
                padding: 2.5rem 3rem;
            }

            .value h3 {
                color: #38bdf8;
                font-size: 1.6rem;
                margin-bottom: 0.8rem;
            }
        </style>
        </head>

        <body>
            <div class="hero">
                <div class="title">⚡ STATYX</div>
                <div class="subtitle">AI-Powered Statistical Data Analysis</div>

                <div class="tagline">
                    Upload → Clean → Analyze → Visualize → Decide<br>
                    <b>No code. No confusion. Just statistically sound insights.</b>
                </div>

                <div class="cta-row">
                    <button class="cta primary">🚀 Start Analyzing</button>
                    <button class="cta secondary">📊 View Features</button>
                </div>
            </div>

            <div class="features">
                <div class="card">
                    <div class="icon">🧹</div>
                    <div class="card-title">AI Data Cleaning</div>
                    <div class="card-desc">
                        Detect missing values, type mismatches, and inconsistencies with one-click fixes.
                    </div>
                </div>

                <div class="card">
                    <div class="icon">📊</div>
                    <div class="card-title">Advanced Statistics</div>
                    <div class="card-desc">
                        Descriptive, inferential, regression, survival analysis with academic-grade output.
                    </div>
                </div>

                <div class="card">
                    <div class="icon">🧠</div>
                    <div class="card-title">AI Objective Analysis</div>
                    <div class="card-desc">
                        Ask questions in plain English and get the correct statistical test automatically.
                    </div>
                </div>

                <div class="card">
                    <div class="icon">📈</div>
                    <div class="card-title">Smart Visualizations</div>
                    <div class="card-desc">
                        Context-aware plots with safety checks to prevent misleading analysis.
                    </div>
                </div>

                <div class="card">
                    <div class="icon">🎓</div>
                    <div class="card-title">Academic-Ready Output</div>
                    <div class="card-desc">
                        Publication-style summaries for research papers, theses, and reports.
                    </div>
                </div>
            </div>

            <div class="value">
                <h3>Why STATYX?</h3>
                <p>
                    STATYX eliminates trial-and-error statistics, prevents incorrect test selection,
                    and empowers users to trust their results — from students to clinicians to analysts.
                </p>
            </div>
        </body>
        </html>
        """,
        height=1300,
        scrolling=False
    )




# ============================================================
# UPLOAD & PREVIEW
# ============================================================
elif tab == "Upload & Preview":
    st.header("📤 Upload & Preview Dataset")

    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

    @st.cache_data
    def load_csv(file):
        return pd.read_csv(file)


    @st.cache_data
    def load_excel(file):
        return pd.read_excel(file)


    if file:
        if file.name.endswith(".csv"):
            df = load_csv(file)
        else:
            df = load_excel(file)

        df.columns = (
            df.columns.str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
        )

        st.session_state.df = df
        st.session_state.preview = df.head(10)

        st.success("File uploaded successfully")
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.preview)

# ============================================================
# DATA CLEANING (CHECK + APPLY)
# ============================================================
elif tab == "Data Cleaning":
    st.header("🧹 AI Data Cleaning")
    st.caption("Smart detection. One-click fixes. Same logic as React + Flask.")

    df = st.session_state.df
    if df is None:
        st.warning("Upload a dataset first")
        st.stop()

    # ----------------------------
    # STEP 1: DETECT ISSUES
    # ----------------------------
    if st.button("🔍 Detect Issues"):
        issues = []

        for col in df.columns:
            col_type = infer_column_type(df[col])

            # Missing values
            missing = int(df[col].isna().sum())
            if missing > 0:
                issues.append({
                    "column": col,
                    "dtype": col_type,
                    "issue": "Missing values",
                    "count": missing
                })

            # Type mismatch
            if col_type == "numeric":
                non_numeric = df[col].notna().sum() - pd.to_numeric(df[col], errors="coerce").notna().sum()
                if non_numeric > 0:
                    issues.append({
                        "column": col,
                        "dtype": col_type,
                        "issue": "Non-numeric in numeric column",
                        "count": non_numeric
                    })
            else:
                numeric_in_cat = int(df[col].apply(is_numeric_like).sum())
                if numeric_in_cat > 0:
                    issues.append({
                        "column": col,
                        "dtype": col_type,
                        "issue": "Numeric in categorical column",
                        "count": numeric_in_cat
                    })

        st.session_state.clean_issues = issues
        st.session_state.clean_preview = None
        st.session_state.clean_download = None

    # ----------------------------
    # STEP 2: SHOW ISSUES
    # ----------------------------
    if st.session_state.clean_issues:
        st.subheader(f"⚠️ Found {len(st.session_state.clean_issues)} Issues")

        issues_df = pd.DataFrame(st.session_state.clean_issues)
        st.dataframe(issues_df, use_container_width=True)

        st.markdown("### 🔧 Fix All Issues")

        for idx, issue in enumerate(st.session_state.clean_issues):
            col = issue["column"]
            dtype = issue["dtype"]
            issue_type = issue["issue"]

            with st.container():
                st.markdown(
                    f"**{col}** → {issue_type} "
                    f"({issue['count']})"
                )

                key_prefix = f"fix_{idx}_{col}_{issue_type}"

                if dtype == "numeric":
                    method = st.selectbox(
                        "Fix method",
                        ["mean", "zero", "custom"],
                        key=f"{key_prefix}_method"
                    )
                else:
                    method = st.selectbox(
                        "Fix method",
                        ["mode", "custom"],
                        key=f"{key_prefix}_method"
                    )

                custom_val = None
                if method == "custom":
                    custom_val = st.text_input(
                        "Custom value",
                        key=f"{key_prefix}_custom"
                    )

                # Store user choice
                st.session_state.clean_actions[key_prefix] = {
                    "column": col,
                    "dtype": dtype,
                    "issue": issue_type,
                    "method": method,
                    "custom": custom_val
                }

                st.divider()

        # ----------------------------
        # STEP 3: APPLY FIX
        # ----------------------------
        if st.button("✅ Apply All Fixes"):
            for action in st.session_state.clean_actions.values():
                col = action["column"]
                method = action["method"]
                custom = action["custom"]

                col_type = infer_column_type(df[col])

                # ---- Numeric columns ----
                if col_type == "numeric":
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                    if method == "mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == "zero":
                        df[col].fillna(0, inplace=True)
                    elif method == "custom" and custom is not None:
                        df[col].fillna(float(custom), inplace=True)

                # ---- Categorical columns ----
                else:
                    numeric_mask = df[col].apply(is_numeric_like)
                    df.loc[numeric_mask, col] = pd.NA

                    if method == "mode":
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    elif method == "custom" and custom is not None:
                        df[col].fillna(str(custom), inplace=True)

            # Persist cleaned dataset
            st.session_state.df = df
            st.session_state.clean_preview = df.head(20)

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.session_state.clean_download = csv_buffer.getvalue()

            st.success("All selected fixes applied successfully ✅")

    # ----------------------------
    # STEP 4: CLEANED PREVIEW
    # ----------------------------
    if st.session_state.clean_preview is not None:
        st.subheader("✅ Cleaned Dataset Preview")
        st.dataframe(st.session_state.clean_preview, use_container_width=True)

        st.download_button(
            "⬇️ Download Cleaned Dataset",
            st.session_state.clean_download,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )


# ============================================================
# STATISTICS (OVERALL + COLUMN-WISE)
# ============================================================
elif tab == "Statistics" and st.session_state.df is not None:

    st.markdown("""
    <div class="card">
        <h3>📊 Statistical Results</h3>
        <p>Auto-generated analysis with academic-style interpretation</p>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.df
    if df is None:
        st.warning("Upload data first")
        st.stop()

    # =====================================================
    # NUMERICAL SUMMARY + INTERPRETATION
    # =====================================================
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        st.subheader("Overall Numerical Summary")

        desc = df[numeric_cols].describe().T
        st.dataframe(desc, use_container_width=True)

        st.markdown("### 📝 Interpretation (Numerical Variables)")
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            median = df[col].median()
            min_val = df[col].min()
            max_val = df[col].max()

            st.write(
                f"For **{col}**, the mean value was **{mean:.2f}** with a standard deviation "
                f"of **{std:.2f}**, indicating moderate variability in the dataset. "
                f"The median was **{median:.2f}**, with values ranging from **{min_val:.2f}** "
                f"to **{max_val:.2f}**. This suggests that most observations are concentrated "
                f"around the central tendency, with no extreme outliers dominating the distribution."
            )

    # =====================================================
    # OVERALL CATEGORICAL SUMMARY (WIDE FORMAT)
    # =====================================================
    cat_cols = df.select_dtypes(include="object").columns

    if len(cat_cols) > 0:
        st.subheader("Overall Categorical Summary")

        overall_cat = build_overall_categorical_table(df)
        st.dataframe(overall_cat, use_container_width=True)

        st.markdown("### 📝 Interpretation (Overall Categorical Variables)")
        st.write(
            "The table above summarizes categorical variables across the dataset. "
            "For each variable, the total number of observations, number of unique "
            "categories, most frequent (top) category, its frequency, and its "
            "percentage contribution are presented. This provides a compact overview "
            "of distribution patterns and dominant groups before conducting "
            "inferential analysis."
        )

    # =====================================================
    # CATEGORICAL SUMMARY + PERCENTAGE + INTERPRETATION
    # =====================================================
    cat_cols = df.select_dtypes(include="object").columns

    if len(cat_cols) > 0:
        st.subheader("Individual Categorical Summary")

        for col in cat_cols:
            st.markdown(f"#### {col}")

            counts = df[col].value_counts(dropna=False)
            perc = (counts / counts.sum()) * 100

            summary_df = pd.DataFrame({
                "Count": counts,
                "Percentage (%)": perc.round(1)
            })

            # ---- Table ----
            st.dataframe(summary_df, use_container_width=True)

            # ---- 3D Bar Chart ----
            fig, mode = smart_categorical_plot(df[col].dropna(), col)

            if mode == "plot" and fig is not None:
                st.pyplot(fig)
            else:
                st.info(
                    "This variable has many unique categories. "
                    "A table representation is more appropriate for interpretation."
                )

            # ---- Interpretation ----
            dominant = counts.idxmax()
            dominant_pct = perc.loc[dominant]

            interpretation = (
                f"In the present dataset, **{dominant}** was the most frequent category "
                f"for **{col}**, accounting for **{dominant_pct:.1f}%** of observations."
            )

            if len(counts) == 2:
                other = counts.index.difference([dominant])[0]
                ratio = counts[dominant] / max(counts[other], 1)
                interpretation += (
                    f" This indicates a clear predominance with an approximate ratio of "
                    f"**{ratio:.1f}:1**."
                )

            st.markdown("**Interpretation:**")
            st.write(interpretation)

            # -----------------------------
            # Interpretation (Academic style)
            # -----------------------------
            dominant = counts.idxmax()
            dominant_pct = perc.loc[dominant]

            interpretation = (
                f"In the present dataset, **{dominant}** was the most frequently observed "
                f"category for **{col}**, accounting for **{dominant_pct:.1f}%** of cases. "
            )

            if len(counts) == 2:
                other = counts.index.difference([dominant])[0]
                ratio = counts[dominant] / max(counts[other], 1)

                interpretation += (
                    f"This reflects a clear predominance of **{dominant}** over **{other}**, "
                    f"with an approximate ratio of **{ratio:.1f}:1**."
                )
            else:
                interpretation += (
                    "Other categories were less frequently represented, indicating an uneven "
                    "distribution across groups."
                )

            st.markdown("**Interpretation:**")
            st.write(interpretation)


# ============================================================
# VISUALIZATIONS (ALL TYPES)
# ============================================================
# ============================================================
# VISUALIZATIONS (SAFE + USER FRIENDLY)
# ============================================================
elif tab == "Visualizations":
    st.header("📈 Visualizations")

    df = st.session_state.df
    if df is None:
        st.warning("Please upload a dataset first to generate visualizations.")
        st.stop()

    plot_type = st.selectbox(
        "Plot Type",
        ["pie", "bar", "line", "scatter", "histogram", "boxplot"]
    )

    cols = st.multiselect("Select Columns", df.columns)

    if st.button("Generate Plot"):
        # ----------------------------
        # BASIC VALIDATIONS
        # ----------------------------
        if not cols:
            st.error("❌ No column selected. Please select at least one column.")
            st.stop()

        # Column existence check
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            st.error(
                f"❌ Selected column(s) {missing_cols} not found in dataset.\n\n"
                "This may happen if the dataset was cleaned or columns were renamed."
            )
            st.stop()

        try:
            fig, ax = plt.subplots()

            # ----------------------------
            # PLOT-SPECIFIC RULES
            # ----------------------------
            if plot_type in ["histogram", "boxplot"]:
                if len(cols) != 1:
                    st.warning("ℹ️ This plot requires exactly **one numeric column**.")
                    st.stop()

                if not pd.api.types.is_numeric_dtype(df[cols[0]]):
                    st.warning(
                        f"❌ '{cols[0]}' is not numeric.\n\n"
                        f"**{plot_type.title()}** plots require numeric data."
                    )
                    st.stop()

                if plot_type == "histogram":
                    df[cols[0]].dropna().plot.hist(ax=ax)
                else:
                    df.boxplot(column=cols[0], ax=ax)

            elif plot_type in ["scatter", "line"]:
                if len(cols) != 2:
                    st.warning("ℹ️ This plot requires **two numeric columns**.")
                    st.stop()

                if not all(pd.api.types.is_numeric_dtype(df[c]) for c in cols):
                    st.warning(
                        "❌ Both selected columns must be numeric for this plot."
                    )
                    st.stop()

                if plot_type == "scatter":
                    df.plot.scatter(cols[0], cols[1], ax=ax)
                else:
                    df.plot(x=cols[0], y=cols[1], ax=ax)

            elif plot_type == "bar":
                if len(cols) != 2:
                    st.warning("ℹ️ Bar plot requires **one categorical and one numeric column**.")
                    st.stop()

                if not pd.api.types.is_numeric_dtype(df[cols[1]]):
                    st.warning(
                        f"❌ '{cols[1]}' must be numeric for bar plot aggregation."
                    )
                    st.stop()

                df.groupby(cols[0])[cols[1]].mean().plot.bar(ax=ax)

            elif plot_type == "pie":
                if len(cols) != 1:
                    st.warning("ℹ️ Pie chart requires **one categorical column**.")
                    st.stop()

                if pd.api.types.is_numeric_dtype(df[cols[0]]):
                    st.warning(
                        f"❌ Pie charts are not suitable for numeric columns.\n\n"
                        f"Please select a categorical column instead."
                    )
                    st.stop()

                df[cols[0]].value_counts().plot.pie(
                    ax=ax, autopct="%1.1f%%", startangle=90
                )

            st.pyplot(fig)

        except Exception as e:
            # ----------------------------
            # FINAL SAFETY NET
            # ----------------------------
            st.error(
                "⚠️ Unable to generate this plot due to incompatible data selection.\n\n"
                f"**Reason:** {str(e)}"
            )


# ============================================================
# AI OBJECTIVE ANALYSIS (E5 + TEST_REGISTRY)
# ============================================================
elif tab == "AI Objective Analysis":
    st.markdown("""
    <div class="card">
        <h3>AI BASED OBJECTIVE ANALYSIS</h3>
        <p>Auto-generated analysis with academic-style interpretation</p>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.df
    if df is None:
        st.warning("Upload data first")
        st.stop()

    objective = st.text_input("Enter objective (e.g., 'Is outcome associated with gender?')")

    if st.button("Run Suggested Tests"):
        with st.spinner("🧠 Analyzing objective..."):
            query_emb = e5_model.encode(
                f"query: {objective}",
                convert_to_tensor=True
            )

            scores = util.cos_sim(query_emb, TEST_EMBEDDINGS)[0]

            ranked_tests = sorted(
                zip(TEST_NAMES, scores.tolist()),
                key=lambda x: x[1],
                reverse=True
            )

        valid_results = []
        target, group = infer_columns_from_objective(df, objective)

        if not target or not group:
            st.error("Could not confidently infer target and group columns from dataset.")
            st.stop()

        # --------------------------------------------------
        # Step 2: Run until we get 5 VALID tests
        # --------------------------------------------------
        MAX_TEST_ATTEMPTS = 10  # 🔥 ADD THIS LINE

        for test_key, confidence in ranked_tests[:MAX_TEST_ATTEMPTS]:
            if test_key not in TEST_REGISTRY:
                continue

            if test_key in ["cox_regression", "kaplan_meier"]:
                continue

            try:
                result = TEST_REGISTRY[test_key](df, target, group)

                if not isinstance(result, dict) or len(result) == 0:
                    continue

                valid_results.append({
                    "test": test_key.replace("_", " ").title(),
                    "confidence": round(confidence, 3),
                    "result": result
                })

            except Exception:
                continue

            if len(valid_results) >= 5:
                break

        if len(valid_results) < 5:
            st.warning("Only limited suitable tests could be applied to this dataset.")

        # --------------------------------------------------
        # Step 3: Display results
        # --------------------------------------------------
        for test in valid_results:
            st.subheader(test["test"])

            # ---- Table ----
            result_df = pd.DataFrame(
                list(test["result"].items()),
                columns=["Metric", "Value"]
            )
            st.table(result_df)

            # ---- Interpretation ----
            interpretation = []

            p_val = test["result"].get("p_value")

            if p_val is not None:
                if p_val < 0.05:
                    interpretation.append(
                        f"The test shows a **statistically significant relationship** (p = {p_val:.4f})."
                    )
                else:
                    interpretation.append(
                        f"The test does **not show a statistically significant relationship** (p = {p_val:.4f})."
                    )

            interpretation.append(
                f"This result evaluates the relationship between **{group}** and **{target}** "
                f"based on the uploaded dataset."
            )

            interpretation.append(
                "These findings should be interpreted in the context of sample size, data quality, "
                "and domain knowledge."
            )

            st.markdown("**Interpretation:**")
            st.write(" ".join(interpretation))

# ============================================================
# CROSS TABULATION + PREVALENCE
# ============================================================
elif tab == "Cross Tabulation":
    st.markdown("""
    <div class="card">
        <h3>Cross Tabulation</h3>
        <p>Auto-generated analysis with academic-style interpretation</p>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.df
    if df is None:
        st.warning("Upload data first")
        st.stop()

    row = st.selectbox("Row Variable", df.columns)
    col = st.selectbox("Column Variable", df.columns)
    percent = st.selectbox("Show As", ["Counts", "Row %", "Column %"])
    prevalence = st.checkbox("Show Prevalence")

    if st.button("Generate Crosstab"):
        ctab = pd.crosstab(df[row], df[col])

        if percent == "Row %":
            ctab = ctab.div(ctab.sum(axis=1), axis=0) * 100
        elif percent == "Column %":
            ctab = ctab.div(ctab.sum(axis=0), axis=1) * 100

        st.dataframe(ctab)

        st.download_button(
            "Download Crosstab",
            csv_download(ctab.reset_index()),
            "crosstab.csv"
        )

        if prevalence and df[col].nunique() == 2:
            positive = sorted(df[col].dropna().unique())[-1]
            prev = (
                df.groupby(row)[col]
                .apply(lambda x: (x == positive).mean() * 100)
                .reset_index(name=f"Prevalence of {col} (%)")
            )
            st.subheader("Prevalence")
            st.dataframe(prev)
