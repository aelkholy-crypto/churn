import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Churn & Segmentation Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background-color: #1e293b;
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }
        .sidebar-header {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(20, 184, 166, 0.2));
            border: 1px solid rgba(129, 140, 248, 0.35);
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 16px;
            text-align: center;
        }
        .sidebar-header h2 {
            margin: 0;
            font-size: 1.15rem;
            font-weight: 700;
            color: #ffffff;
        }
        .sidebar-header p {
            margin: 4px 0 0 0;
            font-size: 0.8rem;
            color: #cbd5e1;
        }
        .hero-card {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.22), rgba(20, 184, 166, 0.16));
            border: 1px solid rgba(129, 140, 248, 0.35);
            border-radius: 18px;
            padding: 24px 28px;
            margin-bottom: 22px;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.28);
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 8px 0;
            letter-spacing: -0.3px;
        }
        .hero-subtitle {
            color: #cbd5e1;
            font-size: 1rem;
            margin: 0;
            line-height: 1.6;
        }
        .kpi-card {
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 14px;
            padding: 16px 18px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: #6366f1;
        }
        .kpi-label {
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 6px;
        }
        .kpi-value {
            color: #ffffff;
            font-size: 1.55rem;
            font-weight: 700;
        }
        .kpi-badge {
            display: inline-block;
            margin-top: 6px;
            padding: 2px 9px;
            border-radius: 999px;
            font-size: 0.75rem;
            background: rgba(20, 184, 166, 0.18);
            color: #5eead4;
        }
        .stDownloadButton button {
            width: 100%;
            background: linear-gradient(135deg, #14b8a6, #0d9488) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE_MAIN = ["#6366f1", "#14b8a6", "#f59e0b", "#f43f5e", "#38bdf8"]


class TabularPreprocessor:
    def __init__(self):
        self.num_cols = []
        self.cat_cols = []
        self.medians = {}
        self.means = {}
        self.stds = {}
        self.modes = {}
        self.categories = {}
        self.feature_names = []

    def fit(self, df: pd.DataFrame):
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        self.feature_names = []

        for col in self.num_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            med = float(s.median()) if not np.isnan(s.median()) else 0.0
            s_filled = s.fillna(med)
            mean_val = float(s_filled.mean())
            std_val = float(s_filled.std(ddof=0)) or 1.0
            self.medians[col] = med
            self.means[col] = mean_val
            self.stds[col] = std_val
            self.feature_names.append(f"num__{col}")

        for col in self.cat_cols:
            s = df[col].astype("object")
            mode_series = s.dropna().mode()
            mode_val = str(mode_series.iloc[0]) if len(mode_series) > 0 else "Unknown"
            s_filled = s.fillna(mode_val).astype(str)
            cats = sorted(s_filled.unique().tolist())
            self.modes[col] = mode_val
            self.categories[col] = cats
            for cat in cats:
                self.feature_names.append(f"cat__{col}_{cat}")

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        blocks = []
        for col in self.num_cols:
            s = pd.to_numeric(df[col], errors="coerce").fillna(self.medians[col]).to_numpy(dtype=float)
            scaled = (s - self.means[col]) / self.stds[col]
            blocks.append(scaled.reshape(-1, 1))

        for col in self.cat_cols:
            s = df[col].astype("object").fillna(self.modes[col]).astype(str).to_numpy()
            cats = self.categories[col]
            ohe = np.zeros((len(s), len(cats)), dtype=float)
            for idx, cat in enumerate(cats):
                ohe[:, idx] = (s == cat).astype(float)
            blocks.append(ohe)

        if not blocks:
            return np.zeros((len(df), 0), dtype=float)
        return np.hstack(blocks)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


class PureKNNClassifier:
    def __init__(self, n_neighbors: int = 7):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=int)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        probs = np.zeros((len(X), 2), dtype=float)
        batch_size = 512
        k = min(self.n_neighbors, len(self.X_train))

        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch = X[start:end]
            dists = np.linalg.norm(batch[:, None, :] - self.X_train[None, :, :], axis=2)
            nn_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]
            nn_labels = self.y_train[nn_idx]
            p1 = nn_labels.mean(axis=1)
            probs[start:end, 1] = p1
            probs[start:end, 0] = 1.0 - p1

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PureDecisionTreeClassifier:
    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.feature_importances_ = None

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p1 = np.mean(y)
        p0 = 1.0 - p1
        return 1.0 - (p0 * p0 + p1 * p1)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        if n_samples < 2 * self.min_samples_leaf:
            return None, None, 0.0

        parent_gini = self._gini(y)
        if parent_gini == 0.0:
            return None, None, 0.0

        best_gain = 0.0
        best_feat = None
        best_thresh = None

        for feat in range(n_features):
            col = X[:, feat]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 1:
                continue
            if len(unique_vals) > 30:
                thresholds = np.unique(np.quantile(col, np.linspace(0.05, 0.95, 19)))
            else:
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

            for thresh in thresholds:
                left_mask = col <= thresh
                n_left = int(np.sum(left_mask))
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                g_left = self._gini(y[left_mask])
                g_right = self._gini(y[~left_mask])
                weighted_gini = (n_left / n_samples) * g_left + (n_right / n_samples) * g_right
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = float(thresh)

        return best_feat, best_thresh, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int, total_samples: int):
        p1 = float(np.mean(y)) if len(y) > 0 else 0.0
        node = {"p1": p1, "samples": int(len(y)), "is_leaf": True}

        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf or p1 in (0.0, 1.0):
            return node

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 1e-7:
            return node

        self._raw_importances[feat] += (len(y) / total_samples) * gain
        left_mask = X[:, feat] <= thresh

        node["is_leaf"] = False
        node["feature"] = feat
        node["threshold"] = thresh
        node["left"] = self._build(X[left_mask], y[left_mask], depth + 1, total_samples)
        node["right"] = self._build(X[~left_mask], y[~left_mask], depth + 1, total_samples)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self._raw_importances = np.zeros(X.shape[1], dtype=float)
        self.tree = self._build(X, y, depth=0, total_samples=len(y))
        total_imp = np.sum(self._raw_importances)
        self.feature_importances_ = (
            self._raw_importances / total_imp if total_imp > 0 else self._raw_importances
        )
        return self

    def _predict_row(self, row: np.ndarray, node: dict) -> float:
        while not node["is_leaf"]:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return node["p1"]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        p1 = np.array([self._predict_row(row, self.tree) for row in X], dtype=float)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PureKMeans:
    def __init__(self, n_clusters: int = 4, max_iter: int = 100, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        n_samples = len(X)
        k = min(self.n_clusters, n_samples)
        centroids = X[rng.choice(n_samples, size=k, replace=False)].copy()

        labels = np.zeros(n_samples, dtype=int)
        for _ in range(self.max_iter):
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    centroids[c] = X[mask].mean(axis=0)

        self.centroids = centroids
        return labels


def stratified_split(X_df: pd.DataFrame, y_series: pd.Series, test_size: float = 0.2, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    train_indices, test_indices = [], []

    for cls_val in np.unique(y_series):
        cls_idx = np.where(y_series.to_numpy() == cls_val)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_size)))
        test_indices.extend(cls_idx[:n_test])
        train_indices.extend(cls_idx[n_test:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return (
        X_df.iloc[train_indices].reset_index(drop=True),
        X_df.iloc[test_indices].reset_index(drop=True),
        y_series.iloc[train_indices].to_numpy(dtype=int),
        y_series.iloc[test_indices].to_numpy(dtype=int),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    acc = float(np.mean(y_true == y_pred))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_prob) ** 2)))

    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = (2 * prec0 * rec0 / (prec0 + rec0)) if (prec0 + rec0) > 0 else 0.0

    report = (
        f"              precision    recall  f1-score   support\n\n"
        f"           0       {prec0:.2f}      {rec0:.2f}      {f1_0:.2f}      {tn + fp}\n"
        f"           1       {precision:.2f}      {recall:.2f}      {f1:.2f}      {tp + fn}\n\n"
        f"    accuracy                           {acc:.2f}      {len(y_true)}\n"
    )
    cm = np.array([[tn, fp], [fn, tp]], dtype=int)
    return acc, f1, rmse, report, cm


def generate_sample_data(n_rows: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, size=n_rows)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.55, 0.25, 0.20])
    internet = rng.choice(["Fiber optic", "DSL", "No"], size=n_rows, p=[0.45, 0.35, 0.20])
    tech_support = np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], size=n_rows, p=[0.35, 0.65]))
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_rows,
        p=[0.35, 0.22, 0.22, 0.21],
    )
    monthly = np.round(
        np.where(internet == "Fiber optic", rng.uniform(70, 115, n_rows),
        np.where(internet == "DSL", rng.uniform(45, 80, n_rows), rng.uniform(18, 30, n_rows))),
        2,
    )
    total = np.round(monthly * tenure + rng.normal(0, 25, n_rows).clip(-20, 200), 2)

    logit = (
        -1.2
        + 1.35 * (contract == "Month-to-month")
        - 0.85 * (contract == "Two year")
        + 0.75 * (internet == "Fiber optic")
        + 0.55 * (tech_support == "No")
        + 0.45 * (payment == "Electronic check")
        - 0.032 * tenure
        + 0.012 * monthly
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (rng.uniform(0, 1, size=n_rows) < prob).astype(int)

    return pd.DataFrame(
        {
            "Contract": contract,
            "InternetService": internet,
            "TechSupport": tech_support,
            "PaymentMethod": payment,
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": np.maximum(total, monthly),
            "Churn": churn,
        }
    )


@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_and_clean_data(uploaded_file, default_path: str = "111.csv.csv") -> tuple[pd.DataFrame, str]:
    source_label = "Uploaded CSV"
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv(default_path)
            source_label = f"Local File ({default_path})"
        except FileNotFoundError:
            df = generate_sample_data()
            source_label = "Built-in Demo Dataset"

    id_cols = [
        col
        for col in df.columns
        if col.lower() in {"id", "customer_id", "customerid", "user_id", "userid"}
        or col.lower().endswith("_id")
    ]
    df = df.drop(columns=id_cols, errors="ignore")

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    target_col = "Churn"
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}

    if target_col in df.columns:
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )
        df = df.dropna(subset=[target_col]).copy()
        df[target_col] = df[target_col].astype(int)
    else:
        raise ValueError("Target column 'Churn' not found in dataset.")

    if "tenure" in df.columns:
        df["Tenure_Group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 120],
            labels=["0-12m", "13-24m", "25-48m", "49m+"],
        )

    if "MonthlyCharges" in df.columns:
        df["Charges_Tier"] = pd.qcut(
            df["MonthlyCharges"],
            q=3,
            labels=["Low Charges", "Medium Charges", "High Charges"],
            duplicates="drop",
        )

    return df, source_label


@st.cache_resource(show_spinner="Training machine learning models...")
def train_models_and_segments(
    df: pd.DataFrame,
    test_size: float,
    n_neighbors: int,
    max_depth: int,
    min_samples_leaf: int,
    n_clusters: int,
):
    target_col = "Churn"
    base_df = df.drop(
        columns=[c for c in ["Segment", "Churn_Risk_Score", "Risk_Tier", "Tenure_Group", "Charges_Tier"] if c in df.columns],
        errors="ignore",
    )

    X_df = base_df.drop(columns=[target_col])
    y_series = base_df[target_col]

    X_train_df, X_test_df, y_train, y_test = stratified_split(X_df, y_series, test_size=test_size, random_state=42)

    preprocessor = TabularPreprocessor()
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    X_all = preprocessor.transform(X_df)

    knn_model = PureKNNClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    dt_model = PureDecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf).fit(X_train, y_train)

    models = {"KNN": knn_model, "Decision Tree": dt_model}
    metrics_rows = []
    model_outputs = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        acc, f1, rmse, report, cm = compute_metrics(y_test, y_pred, y_prob)
        metrics_rows.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "F1-Score": round(f1, 4),
                "RMSE": round(rmse, 4),
            }
        )
        model_outputs[name] = {
            "y_pred": y_pred,
            "y_prob": y_prob,
            "report": report,
            "cm": cm,
        }

    metrics_df = pd.DataFrame(metrics_rows)

    importances = (
        pd.DataFrame(
            {"Feature": preprocessor.feature_names, "Importance": dt_model.feature_importances_}
        )
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    df_result = df.copy()
    cluster_features = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df_result.columns]

    if len(cluster_features) >= 2:
        cluster_df = df_result[cluster_features].copy()
        for c in cluster_features:
            cluster_df[c] = pd.to_numeric(cluster_df[c], errors="coerce")
            cluster_df[c] = cluster_df[c].fillna(cluster_df[c].median())
        X_cluster = cluster_df.to_numpy(dtype=float)
        means = X_cluster.mean(axis=0)
        stds = X_cluster.std(axis=0)
        stds[stds == 0] = 1.0
        X_cluster_scaled = (X_cluster - means) / stds

        kmeans = PureKMeans(n_clusters=n_clusters, random_state=42)
        df_result["Segment"] = kmeans.fit_predict(X_cluster_scaled)
    else:
        df_result["Segment"] = 0

    df_result["Churn_Risk_Score"] = dt_model.predict_proba(X_all)[:, 1]
    df_result["Risk_Tier"] = pd.cut(
        df_result["Churn_Risk_Score"],
        bins=[-0.001, 0.35, 0.70, 1.001],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    summary_cols = [c for c in cluster_features if c in df_result.columns] + ["Churn_Risk_Score", target_col]
    segment_summary = df_result.groupby("Segment", observed=False)[summary_cols].mean().round(3)

    return (
        preprocessor,
        knn_model,
        dt_model,
        metrics_df,
        model_outputs,
        importances,
        df_result,
        segment_summary,
        cluster_features,
    )


def format_tree_rules(node: dict, feature_names: list, depth: int = 0, max_display_depth: int = 3) -> str:
    indent = "│   " * depth
    if node["is_leaf"] or depth >= max_display_depth:
        label = "Churned (1)" if node["p1"] >= 0.5 else "Retained (0)"
        return f"{indent}└── Predict: {label} | Churn Prob: {node['p1']:.2%} | Samples: {node['samples']}\n"

    feat_name = feature_names[node["feature"]]
    thresh = node["threshold"]
    out = f"{indent}├── [{feat_name} <= {thresh:.3f}] (Samples: {node['samples']}, Churn Prob: {node['p1']:.2%})\n"
    out += format_tree_rules(node["left"], feature_names, depth + 1, max_display_depth)
    out += f"{indent}├── [{feat_name} > {thresh:.3f}]\n"
    out += format_tree_rules(node["right"], feature_names, depth + 1, max_display_depth)
    return out


# ==========================================
# SIDEBAR NAVIGATION & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <h2>📊 Churn Studio</h2>
            <p>Navigation & Model Controls</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "🧭 Navigate To",
        [
            "🤖 Models & Feature Importance",
            "🧩 Customer Segments & Risk",
            "📈 Churn Exploration (EDA)",
            "🔮 Single Customer Predictor",
            "📋 Raw Data Explorer",
        ],
    )

    st.divider()
    st.markdown("### 📂 Data Source")
    uploaded_csv = st.file_uploader("Upload CSV Dataset", type=["csv"])

    st.divider()
    st.markdown("### ⚙️ Hyperparameters")
    test_size = st.slider("Test Split Size", 0.1, 0.4, 0.2, step=0.05)
    n_neighbors = st.slider("KNN Neighbors", 3, 21, 7, step=2)
    max_depth = st.slider("Decision Tree Max Depth", 2, 12, 6)
    min_samples_leaf = st.slider("Min Samples per Leaf", 5, 100, 20, step=5)
    n_clusters = st.slider("K-Means Clusters", 2, 8, 4)

try:
    raw_df, data_source_label = load_and_clean_data(uploaded_csv)
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

(
    preprocessor,
    knn_model,
    dt_model,
    metrics_df,
    model_outputs,
    importances,
    df_analyzed,
    segment_summary,
    cluster_features,
) = train_models_and_segments(
    raw_df, test_size, n_neighbors, max_depth, min_samples_leaf, n_clusters
)

with st.sidebar:
    st.divider()
    st.markdown("### 💾 Model Management (.pkl)")
    st.caption(f"Active Dataset: **{data_source_label}**")

    model_bundle = {
        "preprocessor": preprocessor,
        "decision_tree": dt_model,
        "knn": knn_model,
    }
    st.download_button(
        label="⬇️ Export Models (.pkl)",
        data=pickle.dumps(model_bundle),
        file_name="churn_models.pkl",
        mime="application/octet-stream",
    )

    uploaded_pkl = st.file_uploader("Import Saved Model (.pkl)", type=["pkl"])
    if uploaded_pkl is not None:
        try:
            loaded_bundle = pickle.load(uploaded_pkl)
            preprocessor = loaded_bundle["preprocessor"]
            dt_model = loaded_bundle["decision_tree"]
            knn_model = loaded_bundle["knn"]
            st.success("Models loaded from .pkl")
        except Exception as exc:
            st.error(f"Invalid .pkl file: {exc}")


# ==========================================
# MAIN HEADER & KPI BAR
# ==========================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">📊 Customer Churn & Segmentation Studio</div>
        <p class="hero-subtitle">
            End-to-end analytics workspace to train and benchmark <b>KNN</b> and <b>Decision Tree</b> classifiers,
            cluster customers with <b>K-Means</b>, and predict churn risk in real time.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

total_customers = len(df_analyzed)
churn_rate = df_analyzed["Churn"].mean() * 100
high_risk_count = int((df_analyzed["Risk_Tier"] == "High Risk").sum())
best_acc = metrics_df["Accuracy"].max() * 100

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Analyzed Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
            <span class="kpi-badge">{data_source_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Overall Churn Rate</div>
            <div class="kpi-value">{churn_rate:.2f}%</div>
            <span class="kpi-badge" style="background:rgba(244,63,94,0.18);color:#fda4af;">
                {(df_analyzed["Churn"] == 1).sum():,} Churned Customers
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">High Risk Customers</div>
            <div class="kpi-value">{high_risk_count:,}</div>
            <span class="kpi-badge" style="background:rgba(245,158,11,0.18);color:#fcd34d;">
                {(high_risk_count / max(total_customers, 1)) * 100:.1f}% of Customer Base
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Best Model Accuracy</div>
            <div class="kpi-value">{best_acc:.2f}%</div>
            <span class="kpi-badge">KNN vs Decision Tree</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# SIDEBAR PAGE ROUTING
# ==========================================
if page == "🤖 Models & Feature Importance":
    st.subheader("Model Performance Comparison on Test Set")
    col_m1, col_m2 = st.columns([1.1, 1.4])

    with col_m1:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        selected_model_name = st.selectbox(
            "Select Model for Confusion Matrix & Classification Report:",
            options=["Decision Tree", "KNN"],
        )
        cm = model_outputs[selected_model_name]["cm"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Retained (0)", "Churned (1)"],
            y=["Retained (0)", "Churned (1)"],
            color_continuous_scale="Teal",
        )
        fig_cm.update_layout(
            title=f"Confusion Matrix - {selected_model_name}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f8fafc",
            height=340,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_m2:
        max_feats = max(5, min(25, len(importances)))
        top_n = st.slider("Number of Top Features to Display", 5, max_feats, min(15, max_feats))
        top_features = importances.head(top_n)

        fig_imp = px.bar(
            top_features.iloc[::-1],
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Top {top_n} Features Driving Churn (Decision Tree)",
            color="Importance",
            color_continuous_scale=["#38bdf8", "#6366f1", "#14b8a6"],
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,41,59,0.45)",
            font_color="#f8fafc",
            height=450,
            xaxis_title="Importance Score",
            yaxis_title="Feature",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        with st.expander("📄 View Full Classification Report"):
            st.code(model_outputs[selected_model_name]["report"], language="text")

    with st.expander("🌳 Decision Tree Rules Structure"):
        tree_text = format_tree_rules(dt_model.tree, preprocessor.feature_names, max_display_depth=3)
        st.code(tree_text, language="text")

elif page == "🧩 Customer Segments & Risk":
    st.subheader("K-Means Customer Segments Summary")
    st.dataframe(segment_summary, use_container_width=True)

    if len(cluster_features) >= 2:
        c_col1, c_col2 = st.columns(2)
        with c_col1:
            x_axis_col = st.selectbox("X-Axis Feature", cluster_features, index=0)
        with c_col2:
            y_axis_col = st.selectbox(
                "Y-Axis Feature", cluster_features, index=min(1, len(cluster_features) - 1)
            )

        fig_seg = px.scatter(
            df_analyzed,
            x=x_axis_col,
            y=y_axis_col,
            color=df_analyzed["Segment"].astype(str),
            hover_data=[c for c in ["Churn_Risk_Score", "Risk_Tier", "Contract"] if c in df_analyzed.columns],
            title="Customer Distribution by Cluster Segment",
            color_discrete_sequence=PALETTE_MAIN,
            opacity=0.78,
        )
        fig_seg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,41,59,0.45)",
            font_color="#f8fafc",
            legend_title="Segment",
            height=420,
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    st.divider()
    st.subheader("🚨 High Risk Customers Watchlist")
    risk_filter = st.multiselect(
        "Filter by Risk Tier:",
        options=["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk"],
    )
    filtered_risk_df = df_analyzed[df_analyzed["Risk_Tier"].isin(risk_filter)].sort_values(
        by="Churn_Risk_Score", ascending=False
    )
    st.write(f"Matching Customers Count: **{len(filtered_risk_df):,}**")
    st.dataframe(filtered_risk_df.head(200), use_container_width=True)

    csv_buffer = io.StringIO()
    filtered_risk_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Filtered Customers (CSV)",
        data=csv_buffer.getvalue(),
        file_name="churn_risk_customers.csv",
        mime="text/csv",
    )

elif page == "📈 Churn Exploration (EDA)":
    st.subheader("Churn Rate (%) Breakdown by Categorical & Binned Features")
    df_eda = df_analyzed.copy()

    candidate_cats = [
        c
        for c in df_eda.columns
        if c not in {"Churn", "Churn_Risk_Score", "tenure", "MonthlyCharges", "TotalCharges"}
        and df_eda[c].nunique(dropna=True) <= 15
    ]

    if candidate_cats:
        selected_cat = st.selectbox("Select Feature to Analyze Churn Rate:", candidate_cats)

        ct_pct = (pd.crosstab(df_eda[selected_cat], df_eda["Churn"], normalize="index") * 100).round(2)
        if 0 in ct_pct.columns and 1 in ct_pct.columns:
            ct_pct = ct_pct[[0, 1]]
            ct_pct.columns = ["Retained (%)", "Churned (%)"]

            col_e1, col_e2 = st.columns([1.2, 1.3])
            with col_e1:
                st.markdown(f"**Percentage Breakdown for `{selected_cat}`:**")
                st.dataframe(ct_pct, use_container_width=True)

            with col_e2:
                plot_df = ct_pct.reset_index().melt(
                    id_vars=selected_cat,
                    value_vars=["Retained (%)", "Churned (%)"],
                    var_name="Status",
                    value_name="Percentage",
                )
                fig_eda = px.bar(
                    plot_df,
                    x=selected_cat,
                    y="Percentage",
                    color="Status",
                    barmode="group",
                    text_auto=".1f",
                    title=f"Retained vs Churned by {selected_cat}",
                    color_discrete_map={"Retained (%)": "#14b8a6", "Churned (%)": "#f43f5e"},
                )
                fig_eda.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(30,41,59,0.45)",
                    font_color="#f8fafc",
                    yaxis_title="Percentage (%)",
                    height=380,
                )
                st.plotly_chart(fig_eda, use_container_width=True)

elif page == "🔮 Single Customer Predictor":
    st.subheader("Single Customer Churn Prediction Simulator")
    st.caption("Adjust customer attributes below and click Predict to estimate churn probability.")

    feature_df = raw_df.drop(
        columns=[c for c in ["Churn", "Tenure_Group", "Charges_Tier"] if c in raw_df.columns],
        errors="ignore",
    )

    user_input = {}
    num_input_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_input_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        st.markdown("**Numerical Features**")
        for col in num_input_cols:
            col_min = float(np.nanmin(feature_df[col]))
            col_max = float(np.nanmax(feature_df[col]))
            col_med = float(np.nanmedian(feature_df[col]))
            if col_min == col_max:
                col_max = col_min + 1.0
            user_input[col] = st.number_input(
                label=col,
                min_value=col_min,
                max_value=col_max,
                value=col_med,
            )

    with col_in2:
        st.markdown("**Categorical Features**")
        for col in cat_input_cols:
            options = sorted(feature_df[col].dropna().astype(str).unique().tolist())
            if options:
                user_input[col] = st.selectbox(label=col, options=options)

    if st.button("🚀 Predict Churn Risk", type="primary"):
        single_df = pd.DataFrame([user_input])
        single_X = preprocessor.transform(single_df)
        dt_prob = float(dt_model.predict_proba(single_X)[0, 1])
        knn_prob = float(knn_model.predict_proba(single_X)[0, 1])
        dt_pred = int(dt_model.predict(single_X)[0])
        knn_pred = int(knn_model.predict(single_X)[0])

        res_c1, res_c2 = st.columns(2)
        with res_c1:
            st.metric(
                label="Decision Tree - Churn Probability",
                value=f"{dt_prob * 100:.1f}%",
                delta="Churned (1)" if dt_pred == 1 else "Retained (0)",
                delta_color="inverse",
            )
        with res_c2:
            st.metric(
                label="KNN - Churn Probability",
                value=f"{knn_prob * 100:.1f}%",
                delta="Churned (1)" if knn_pred == 1 else "Retained (0)",
                delta_color="inverse",
            )

        if dt_prob >= 0.70:
            st.error("⚠️ Risk Level: **High Risk** — Immediate retention intervention recommended.")
        elif dt_prob >= 0.35:
            st.warning("🔶 Risk Level: **Medium Risk** — Proactive customer engagement recommended.")
        else:
            st.success("✅ Risk Level: **Low Risk** — Customer is likely to be retained.")

elif page == "📋 Raw Data Explorer":
    st.subheader("Analyzed Dataset Preview & Summary Statistics")
    st.dataframe(df_analyzed.head(250), use_container_width=True)

    st.divider()
    st.subheader("Numerical Columns Summary")
    st.dataframe(df_analyzed.describe().T, use_container_width=True)
