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
    graph_data = context_data.get("graph_data", "No graph data.")
    ctx = f"""
You are an assistant analyzing a structured Knowledge Graph. You must answer ONLY based on the information provided below from the graph.

If the user's question cannot be answered using this data, respond with:
"This information is not available in the current knowledge graph."

=== EXAMPLE ===
Q: What applications are connected to PostgreSQL?
A: In your knowledge graph, PostgreSQL is connected to Application A and Application B via 'used_by'.

Q: What is the birthday of Albert Einstein?
A: This information is not available in the current knowledge graph.

=== KNOWLEDGE GRAPH CONTEXT ===
{graph_data}

=== CHAT HISTORY ===
{format_chat_history(context_data.get('chat_history', []))}

=== USER QUESTION ===
{question}
"""
    payload = {"inputs": ctx, "parameters": {"max_new_tokens": 500, "temperature": 0.3}}
    resp = requests.post(
        LLM_API_URL, json=payload,
        auth=HTTPBasicAuth(LLM_USERNAME, LLM_PASSWORD),
        headers={"Content-Type": "application/json"}, timeout=30
    )
    if resp.ok:
        res = resp.json()
        if isinstance(res, list) and res:
            return res[0].get("generated_text", "")
        if isinstance(res, dict) and "generated_text" in res:
            return res.get("generated_text", "")
        return f"Unexpected format: {res.keys()}"
    return f"LLM Error {resp.status_code}: {resp.text}"

# Main App
st.set_page_config(page_title="KG Explorer", layout="wide")
st.title("Knowledge Graph Explorer")
nodes_df, edges_df = load_data()

# Category & Node Selection
labels = sorted(nodes_df.label.dropna().unique())
sel_label = st.selectbox("Select Category", labels)
show_props = st.checkbox("Show full node metadata", value=False)
filt = nodes_df[nodes_df.label == sel_label]
node_opts = ["All"] + sorted(filt.name.unique())
sel_node = st.selectbox("Select Node", node_opts)
is_cat = sel_node == "All"

def show_graph(center_ids, title):
    st.markdown(title)
    G = build_graph(nodes_df, edges_df, center_ids)
    render_pyvis(G, DEFAULT_PYVIS_OPTIONS)

if is_cat:
    graph_data = "\n\n".join([
        get_node_relationships(name, nodes_df, edges_df, max_rel=3)
        for name in filt.name.head(5)
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
