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

def render_pyvis(G, options):
    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(G)
    net.force_atlas_2based()
    net.set_options(options)
    path = tempfile.mktemp(suffix=".html")
    net.save_graph(path)
    with open(path, encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=650, scrolling=True)

def format_chat_history(chat_history):
    return "\n".join(f"{r.title()}: {m}" for r, m in chat_history[-5:]) if chat_history else "No previous conversation."

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
    graph_data = context_data.get("graph_data", "No graph data provided.")
    ctx = f"""
You are an assistant analyzing a structured Knowledge Graph. You must answer ONLY based on the information provided below from the graph.
If the user's question cannot be answered using this data, respond with:
"This information is not available in the current knowledge graph."
=== KNOWLEDGE GRAPH CONTEXT ===
{graph_data}
=== CHAT HISTORY ===
{format_chat_history(context_data.get('chat_history', []))}
=== USER QUESTION ===
{question}
"""
    payload = {"inputs": ctx, "parameters": {"max_new_tokens": 1000, "temperature": 0.3}}
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

def enrich_answer_with_context(answer, nodes_df):
    bullets = []
    terms = nodes_df[["name", "label"]].drop_duplicates()
    for _, row in terms.iterrows():
        if row.name.lower() in answer.lower():
            bullets.append(f"• **{row.name}** is a **{row.label}** related to the graph.")
    return answer + "\n\n**Related Concepts from the Graph:**\n" + "\n".join(bullets) if bullets else answer

# --- Streamlit UI ---
st.set_page_config(page_title="Knowledge Graph Explorer", layout="wide")
st.title("Knowledge Graph Explorer")

try:
    nodes_df, edges_df = load_data()
except Exception as e:
    st.error(f"Database connection failed: {e}")
    st.stop()

labels = sorted(nodes_df["label"].dropna().unique())
selected_label = st.selectbox("Select Category", labels)
filtered_nodes = nodes_df[nodes_df.label == selected_label]
node_options = ["All"] + sorted(filtered_nodes["name"].unique())
selected_node = st.selectbox("Select Node", node_options)

is_cat_view = selected_node == "All"
node_sample_size = st.slider("How much of the graph should the AI see? (More = deeper answers, but slower)", 5, 50, 15)
strategy = st.selectbox("How should we pick which nodes to include for the AI?", ["Pick first few (default)", "Pick most connected ones"])

if strategy.startswith("Pick most connected"):
    connected_counts = edges_df['source_node_id'].value_counts() + edges_df['target_node_id'].value_counts()
    top_ids = connected_counts.sort_values(ascending=False).head(node_sample_size).index
    sample_nodes = nodes_df[nodes_df.node_id.isin(top_ids)]
else:
    sample_nodes = nodes_df.head(node_sample_size)

graph_data = "\n\n".join([
    get_node_relationships(row["name"], nodes_df, edges_df, max_rel=5)
    for _, row in sample_nodes.iterrows()
])

if is_cat_view:
    G = build_graph(nodes_df, edges_df, filtered_nodes["node_id"], direction="Both")
else:
    node_row = filtered_nodes[filtered_nodes["name"] == selected_node].iloc[0]
    direction_label = st.selectbox("What kind of connections do you want to see?", ["All connections", "Only shows what this connects to", "Only shows what connects to this"])
    direction_map = {
        "All connections": "Both",
        "Only shows what this connects to": "Outgoing",
        "Only shows what connects to this": "Incoming"
    }
    direction = direction_map[direction_label]
    G = build_graph(nodes_df, edges_df, [node_row.node_id], direction=direction)

render_pyvis(G, DEFAULT_PYVIS_OPTIONS)

with st.expander("🔍 What does the AI see? (LLM Context Preview)", expanded=False):
    st.markdown("""
    The AI only sees a **sample of your knowledge graph** including selected node names, types, and their relationships.
    This helps it answer graph-related questions more accurately.

    **Sample Context Sent to the AI:**
    """)
    st.code(graph_data, language="text")

# --- Chat Interface ---
st.markdown("### Ask a question about the graph")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
for role, msg in st.session_state.chat_history[-5:]:
    st.markdown(f"**{role.capitalize()}:** {msg}")

query = st.text_input("Your question")
if st.button("Send") and query.strip():
    st.session_state.chat_history.append(("user", query))
    with st.spinner("Thinking..."):
        ctx = {
            "graph_data": graph_data,
            "chat_history": st.session_state.chat_history[:-1]
        }
        raw_answer = query_llm(query, ctx)
        enriched = enrich_answer_with_context(raw_answer, nodes_df)
    st.session_state.chat_history.append(("assistant", enriched))
    st.rerun()
