import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------

st.set_page_config(
    page_title="GMM Dashboard",
    page_icon="🎯",
    layout="wide"
)

# -----------------------------------------
# CSS
# -----------------------------------------

st.markdown("""
<style>

.hero{
    padding:25px;
    border-radius:18px;
    background:linear-gradient(135deg,#9333ea,#ec4899);
    color:white;
    text-align:center;
    margin-bottom:20px;
}

.metric-card{
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 4px 10px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# HERO
# -----------------------------------------

st.markdown("""
<div class='hero'>
<h1>🎯 Gaussian Mixture Model Dashboard</h1>
<p>Probabilistic Clustering using Gaussian Distributions</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# LOAD DATA
# -----------------------------------------

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

X_scaled = scaler.transform(df)

labels = model.predict(X_scaled)

probabilities = model.predict_proba(X_scaled)

df["Cluster"] = labels

# -----------------------------------------
# METRICS
# -----------------------------------------

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Samples", len(df))

with c2:
    st.metric("Features", 4)

with c3:
    st.metric("Clusters", len(set(labels)))

# -----------------------------------------
# DATASET PREVIEW
# -----------------------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(df.head(15), use_container_width=True)

# -----------------------------------------
# CLUSTER DISTRIBUTION
# -----------------------------------------

st.subheader("📊 Cluster Distribution")

cluster_counts = (
    pd.Series(labels)
    .value_counts()
    .sort_index()
)

fig1, ax1 = plt.subplots(figsize=(7,4))

ax1.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

ax1.set_title("Cluster Distribution")
ax1.set_xlabel("Cluster")
ax1.set_ylabel("Count")

st.pyplot(fig1)

# -----------------------------------------
# VISUALIZATION
# -----------------------------------------

st.subheader("🎨 Cluster Visualization")

col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox(
        "X Feature",
        iris.feature_names,
        index=0
    )

with col2:
    y_feature = st.selectbox(
        "Y Feature",
        iris.feature_names,
        index=2
    )

fig2, ax2 = plt.subplots(figsize=(8,5))

scatter = ax2.scatter(
    df[x_feature],
    df[y_feature],
    c=labels
)

ax2.set_xlabel(x_feature)
ax2.set_ylabel(y_feature)
ax2.set_title("GMM Clustering")

st.pyplot(fig2)

# -----------------------------------------
# CLUSTER STATISTICS
# -----------------------------------------

st.subheader("📑 Cluster Statistics")

stats = (
    df.groupby("Cluster")
    .mean()
)

st.dataframe(
    stats,
    use_container_width=True
)

# -----------------------------------------
# PROBABILITY TABLE
# -----------------------------------------

st.subheader("🧮 Cluster Membership Probabilities")

prob_df = pd.DataFrame(
    probabilities,
    columns=[
        f"Cluster {i}"
        for i in range(probabilities.shape[1])
    ]
)

st.dataframe(
    prob_df.head(15),
    use_container_width=True
)

# -----------------------------------------
# MODEL INFORMATION
# -----------------------------------------

st.subheader("⚙️ Model Parameters")

st.write("Mixture Weights")

weights_df = pd.DataFrame({
    "Cluster": [f"Cluster {i}" for i in range(len(model.weights_))],
    "Weight": model.weights_
})

st.dataframe(weights_df)

# -----------------------------------------
# DOWNLOAD
# -----------------------------------------

csv = df.to_csv(index=False)

st.download_button(
    "⬇ Download Results",
    csv,
    "gmm_results.csv",
    "text/csv"
)

# -----------------------------------------
# THEORY
# -----------------------------------------

