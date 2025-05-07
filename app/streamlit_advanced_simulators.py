import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

st.title("Advanced A/B & Multivariate Test Simulator")

# Sidebar settings
test_type = st.sidebar.selectbox("Choose Test Type", ["A/B Test", "Multivariate Test (ANOVA)", "Multi-arm Test", "Bayesian A/B Test", "Propensity Score Matching"])

if test_type == "A/B Test":
    st.header("Classical A/B Test")
    p_control = st.slider("Control Conversion Rate (%)", 1, 50, 10) / 100
    p_test = st.slider("Test Conversion Rate (%)", 1, 50, 13) / 100
    n = st.slider("Sample Size per Group", 100, 10000, 1000)

    group_a = np.random.binomial(1, p_control, n)
    group_b = np.random.binomial(1, p_test, n)

    conv_a = group_a.mean()
    conv_b = group_b.mean()
    uplift = (conv_b - conv_a) / conv_a * 100
    z_score, p_val = stats.ttest_ind(group_b, group_a)

    st.metric("Conversion A", f"{conv_a:.2%}")
    st.metric("Conversion B", f"{conv_b:.2%}")
    st.metric("Uplift", f"{uplift:.2f}%")
    st.metric("P-value", f"{p_val:.4f}")

elif test_type == "Multivariate Test (ANOVA)":
    st.header("Multivariate Test (ANOVA)")
    num_groups = st.slider("Number of Variants", 3, 5, 3)
    group_sizes = st.slider("Sample Size per Group", 100, 5000, 1000)

    groups = []
    for i in range(num_groups):
        conv_rate = st.slider(f"Conversion Rate for Group {i+1} (%)", 1, 50, 10 + i*2) / 100
        group = np.random.binomial(1, conv_rate, group_sizes)
        groups.append(group)

    f_stat, p_val = stats.f_oneway(*groups)
    st.write(f"F-statistic: {f_stat:.4f}, P-value: {p_val:.4f}")

elif test_type == "Multi-arm Test":
    st.header("Multi-arm Bandit Style Test")
    k = st.slider("Number of Variants", 3, 6, 3)
    n = st.slider("Sample Size per Variant", 100, 5000, 1000)

    results = []
    for i in range(k):
        p = st.slider(f"Success Rate for Arm {i+1} (%)", 1, 50, 10 + i*3) / 100
        data = np.random.binomial(1, p, n)
        results.append((f"Arm {i+1}", data.mean()))

    results.sort(key=lambda x: x[1], reverse=True)
    st.write("Results by Arm (Best to Worst):")
    for arm, rate in results:
        st.write(f"{arm}: {rate:.2%}")

elif test_type == "Bayesian A/B Test":
    st.header("Bayesian A/B Test (Simplified - Beta Posterior)")
    p_control = st.slider("Control Rate (%)", 1, 50, 10) / 100
    p_test = st.slider("Test Rate (%)", 1, 50, 13) / 100
    size = st.slider("Sample Size", 500, 5000, 2000)

    control = np.random.binomial(1, p_control, size)
    test = np.random.binomial(1, p_test, size)

    alpha, beta = 1, 1
    posterior_control = np.random.beta(alpha + control.sum(), beta + size - control.sum(), 10000)
    posterior_test = np.random.beta(alpha + test.sum(), beta + size - test.sum(), 10000)

    prob_b_better = (posterior_test > posterior_control).mean()
    st.write(f"Probability B > A: {prob_b_better:.2%}")

elif test_type == "Propensity Score Matching":
    st.header("Causal Inference: Propensity Score Matching")
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    size = st.slider("Sample Size", 500, 5000, 1000)
    X = np.random.normal(0, 1, (size, 3))
    treatment = np.random.binomial(1, 0.5, size)
    outcome = 2 * treatment + X[:, 0] + np.random.normal(0, 1, size)

    df = pd.DataFrame(X, columns=["X1", "X2", "X3"])
    df["treatment"] = treatment
    df["outcome"] = outcome

    model = LogisticRegression()
    model.fit(df[["X1", "X2", "X3"]], df["treatment"])
    df["propensity"] = model.predict_proba(df[["X1", "X2", "X3"]])[:, 1]

    treated = df[df["treatment"] == 1]
    control = df[df["treatment"] == 0]

    nn = NearestNeighbors(n_neighbors=1).fit(control[["propensity"]])
    distances, indices = nn.kneighbors(treated[["propensity"]])
    matched_controls = control.iloc[indices.flatten()]
    ate = (treated["outcome"].values - matched_controls["outcome"].values).mean()

    st.metric("Estimated ATE", f"{ate:.2f}")
