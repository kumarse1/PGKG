import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile, os
from sqlalchemy import create_engine
import psycopg2
import requests
from requests.auth import HTTPBasicAuth

# Configuration
LLM_API_URL = "https://your-llama-endpoint.com/generate"
LLM_USERNAME = "your_username_here"
LLM_PASSWORD = "your_password_here"
DB_USER = ""
DB_PASS = ""
DB_HOST = ""
DB_PORT = ""
DB_NAME = ""

# Default PyVis visualization options (shared for both views)
DEFAULT_PYVIS_OPTIONS = """
var options = {
  "physics": {
    "enabled": true,
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.01,
      "springLength": 100,
      "springConstant": 0.08
    },
    "maxVelocity": 50,
    "solver": "forceAtlas2Based",
    "timestep": 0.35,
    "stabilization": {"iterations": 150}
  },
  "nodes": {
    "borderWidth": 2,
    "size": 30,
    "font": {
      "size": 12,
      "face": "Arial"
    }
  },
  "edges": {
    "arrows": {
      "to": {
        "enabled": true,
        "scaleFactor": 1.2
      }
    },
    "color": {
      "inherit": false,
      "color": "#848484"
    },
    "smooth": {
      "enabled": true,
      "type": "dynamic"
    }
  }
}
"""

# Database & Data Loading
engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

@st.cache_resource
def get_connection():
    return psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS,
        host=DB_HOST, port=DB_PORT
    )

@st.cache_data
def load_data():
    conn = get_connection()
    nodes = pd.read_sql("SELECT * FROM kg_nodes", conn)
    edges = pd.read_sql("SELECT * FROM kg_edges", conn)
    return normalize_df(nodes), normalize_df(edges)

# Utility Functions

def normalize_df(df):
    for src, dst in [("int64", "int32"), ("float64", "float32")]:
        cols = df.select_dtypes(include=[src]).columns
        df[cols] = df[cols].astype(dst)
    return df


def to_native(val):
    return "" if pd.isna(val) else str(val)


def build_graph(nodes_df, edges_df, center_ids, direction="Both"):
    G = nx.DiGraph()
    for cid in center_ids:
        center = nodes_df[nodes_df.node_id == cid].iloc[0]
        G.add_node(to_native(center.node_id), label=to_native(center.label), title=center["name"])
        if direction in ("Both", "Outgoing"):
            for _, e in edges_df[edges_df.source_node_id == cid].iterrows():
                tgt = nodes_df[nodes_df.node_id == e.target_node_id].iloc[0]
                G.add_node(to_native(tgt.node_id), label=to_native(tgt.label), title=tgt["name"])
                G.add_edge(to_native(cid), to_native(tgt.node_id), label=to_native(e.relation_type))
        if direction in ("Both", "Incoming"):
            for _, e in edges_df[edges_df.target_node_id == cid].iterrows():
                src = nodes_df[nodes_df.node_id == e.source_node_id].iloc[0]
                G.add_node(to_native(src.node_id), label=to_native(src.label), title=src["name"])
                G.add_edge(to_native(src.node_id), to_native(cid), label=to_native(e.relation_type))
    return G


def render_pyvis(G, options: str):
    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(G)
    net.force_atlas_2based()
    net.set_options(options)
    path = tempfile.mktemp(suffix=".html")
    net.save_graph(path)
    with open(path, encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=650, scrolling=True)

# LLM Context & Query

def format_chat_history(chat_history):
    if not chat_history:
        return "No previous conversation."
    return "\n".join(f"{r.title()}: {m}" for r, m in chat_history[-5:])


def get_node_relationships(node_name, nodes_df, edges_df, max_rel=10):
    row = nodes_df[nodes_df.name == node_name]
    if row.empty:
        return f"Node '{node_name}' not found."
    nid = row.iloc[0].node_id
    rels = [f"Node: {node_name} (Type: {row.iloc[0].label})"]
    out = edges_df[edges_df.source_node_id == nid].head(max_rel)
    if not out.empty:
        rels.append("\nOutgoing:")
        for _, e in out.iterrows():
            tgt = nodes_df[nodes_df.node_id == e.target_node_id].iloc[0]
            rels.append(f"  → {tgt.name} ({tgt.label}) via '{e.relation_type}'")
    inc = edges_df[edges_df.target_node_id == nid].head(max_rel)
    if not inc.empty:
        rels.append("\nIncoming:")
        for _, e in inc.iterrows():
            src = nodes_df[nodes_df.node_id == e.source_node_id].iloc[0]
            rels.append(f"  ← {src.name} ({src.label}) via '{e.relation_type}'")
    return "\n".join(rels)


def query_llm(question, context_data):
    graph_data = "

".join([
        get_node_relationships(name, nodes_df, edges_df, max_rel=3)
        for name in top_nodes.name.unique()
    ])
    show_graph(filt.node_id.tolist(), f"### {sel_label} Overview")
else:
    node = filt[filt.name == sel_node].iloc[0]
    st.markdown("### Node Properties")
    with st.expander("Show Node Properties"):
        if show_props:
            st.json(dict(node))
        else:
            st.json({k: v for k, v in node.items() if k not in ['node_id', 'label', 'name']})
    dir_opt = st.selectbox("Connection Direction", ["Both", "Outgoing", "Incoming"])
    graph_data = get_node_relationships(sel_node, nodes_df, edges_df)
    show_graph([node.node_id], f"### {sel_node} ({dir_opt})")

# Chat Interface
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### Chat")
with col2:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for role, msg in st.session_state.chat_history[-8:]:
    prefix = "Q:" if role == 'user' else "A:"
    st.markdown(f"**{prefix}** {msg}")
    st.markdown("---")

query = st.text_input("Ask about your graph:")
if st.button("Send") and query.strip():
    st.session_state.chat_history.append(("user", query))
    ctx = {"graph_data": graph_data, "chat_history": st.session_state.chat_history[:-1]}
    answer = query_llm(query, ctx)
    st.session_state.chat_history.append(("ai", answer))
    st.rerun()

# Total lines: ~130
