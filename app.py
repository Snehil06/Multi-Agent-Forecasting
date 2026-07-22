
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Multi-Agent Forecasting", layout="wide")
st.title("Multi-Agent Financial Forecasting")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

@st.cache_data
def load():
    results = pd.read_csv(os.path.join(RESULTS_DIR, "backtest_results.csv"), index_col=0, parse_dates=True)
    report = pd.read_csv(os.path.join(RESULTS_DIR, "metrics_report.csv"))
    return results, report

results, report = load()


def extract_agent_names(df):
    names = []
    for col in df.columns:
        if col.endswith("_pred") and col != "ensemble_pred":
            if col == "sequence_pred":
                names.append("sequence")
            elif col == "SequenceAgent_pred":
                names.append("SequenceAgent")
            else:
                names.append(col[:-5])
    return sorted(set(names))


def pred_col_for(name):
    if name == "sequence":
        return "sequence_pred"
    if name == "SequenceAgent":
        return "SequenceAgent_pred"
    return f"{name}_pred"


def weight_col_for(name):
    if name == "sequence":
        return "sequence_weight"
    if name == "SequenceAgent":
        return "SequenceAgent_weight"
    return f"{name}_weight"

agent_names = extract_agent_names(results)

for name in agent_names:
    weight_col = weight_col_for(name)
    if weight_col not in results.columns:
        results[weight_col] = 0.0

valid_order = ["trend", "momentum", "volatility", "equal", "sequence", "SequenceAgent"]
filtered_names = []
for v_name in valid_order:
    for a_name in agent_names:
        if v_name.lower() == a_name.lower() or (v_name.lower() + "agent") == a_name.lower() or a_name.lower().startswith(v_name.lower()):
            filtered_names.append(a_name)
            break
agent_names = filtered_names

tab1, tab2, tab3 = st.tabs(["Forecast vs Actual", "Agent Weights", "Performance"])

with tab1:
    n = st.slider("Days to show", 50, len(results), 250)
    recent = results.tail(n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent["actual"], name="Actual", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=recent.index, y=recent["ensemble_pred"], name="Ensemble"))
    for name in agent_names:
        pred_col = pred_col_for(name)
        if pred_col in results.columns:
            label = name.replace("SequenceAgent", "Sequence")
            fig.add_trace(go.Scatter(x=recent.index, y=recent[pred_col], name=label.capitalize()))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(report)

with tab2:
    fig = go.Figure()
    for nm in agent_names:
        col = weight_col_for(nm)
        if col in results.columns:
            fig.add_trace(go.Scatter(x=results.index, y=results[col], name=nm, stackgroup="one"))
    fig.add_vline(x="2020-03-01", line_dash="dash")
    fig.add_vline(x="2022-01-01", line_dash="dash")
    fig.update_layout(title="Hedge Agent Weights Over Time")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.plotly_chart(px.bar(report, x="Label", y="Sharpe Ratio", color="Sharpe Ratio"), use_container_width=True)
    st.plotly_chart(px.bar(report, x="Label", y="Directional Accuracy", color="Directional Accuracy"), use_container_width=True)
    ens_pos = np.sign(results["ensemble_pred"].values)
    ens_curve = 10000 * np.exp(np.cumsum(ens_pos * results["actual"].values))
    bh_curve = 10000 * np.exp(np.cumsum(results["actual"].values))
    eq = go.Figure()
    eq.add_trace(go.Scatter(x=results.index, y=ens_curve, name="Ensemble"))
    eq.add_trace(go.Scatter(x=results.index, y=bh_curve, name="Buy & Hold"))
    eq.update_layout(title="$10,000 invested")
    st.plotly_chart(eq, use_container_width=True)
