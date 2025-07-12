# YOUR ORIGINAL IMPORTS - UNCHANGED
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from sqlalchemy import create_engine
import psycopg2

# NEW ADDITION: LLM integration imports for chat feature
import requests
from requests.auth import HTTPBasicAuth

# NEW ADDITION: LLM Configuration for chat functionality - UPDATED for Llama
LLM_API_URL = "https://your-llama-endpoint.com/generate"  # Updated for typical Llama endpoint
LLM_USERNAME = "your_username_here"  # Replace with actual username
LLM_PASSWORD = "your_password_here"  # Replace with actual password

# YOUR ORIGINAL DATABASE CONNECTION - UNCHANGED
# --- Connect to PostgreSQL ---
db_user = ""
db_password = ""
db_host = ""
db_port = ""
db_name = ""
table_name = "kg_application_data"

engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# --- Database Connection ---
@st.cache_resource
def get_connection():
    return psycopg2.connect(
        dbname=db_name, user=db_user, password=db_password, host=db_host,
        port=db_port
    )

# YOUR ORIGINAL NORMALIZE FUNCTION - UNCHANGED
def normalize_df(df):
    int_cols = df.select_dtypes(include=['int64']).columns
    df[int_cols] = df[int_cols].astype('int32')
    
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    return df

# YOUR ORIGINAL LOAD DATA FUNCTION - UNCHANGED
@st.cache_data
def load_data():
    conn = get_connection()
    _nodes = pd.read_sql("SELECT * FROM kg_nodes", conn)
    _edges = pd.read_sql("SELECT * FROM kg_edges", conn)
    
    nodes = normalize_df(_nodes)
    edges = normalize_df(_edges)
    
    return nodes, edges

def to_native_type(value):
    if pd.isna(value):
        return ""
    return str(value)

# YOUR ORIGINAL BUILD SUBGRAPH FUNCTION - UNCHANGED (removed "All" since it's now in label filter)
def build_subgraph(nodes_df, edges_df, selected_node_id, direction):
    G = nx.DiGraph()
    
    center = nodes_df[nodes_df.node_id == selected_node_id].iloc[0]
    G.add_node(
        to_native_type(center.node_id),
        label=to_native_type(center.label),
        group=to_native_type(center.label),
        title=f"{center['name']}"
    )
    
    # YOUR ORIGINAL LOGIC - UNCHANGED
    if direction in ("Outgoing", "Both"):
        out_edges = edges_df[edges_df.source_node_id == selected_node_id]
        for _, row in out_edges.iterrows():
            target = nodes_df[nodes_df.node_id == row.target_node_id]
            if not target.empty:
                tgt = target.iloc[0]
                G.add_node(
                    to_native_type(tgt.node_id),
                    label=to_native_type(tgt.label),
                    group=to_native_type(tgt.label),
                    title=f"{tgt['name']}"
                )
                G.add_edge(
                    to_native_type(center.node_id),
                    to_native_type(tgt.node_id),
                    label=to_native_type(row.relation_type),
                    title=f"{tgt['name']}"
                )

    # YOUR ORIGINAL LOGIC - UNCHANGED
    if direction in ("Incoming", "Both"):
        in_edges = edges_df[edges_df.target_node_id == selected_node_id]
        for _, row in in_edges.iterrows():
            source = nodes_df[nodes_df.node_id == row.source_node_id]
            if not source.empty:
                src = source.iloc[0]
                G.add_node(
                    to_native_type(src.node_id),
                    label=to_native_type(src.label),
                    group=to_native_type(src.label),
                    title=f"{src['name']}"
                )
                G.add_edge(
                    to_native_type(center.node_id),
                    to_native_type(src.node_id),
                    label=to_native_type(row.relation_type),
                    title=f"{src['name']}"
                )

    return G

# NEW ADDITION: LLM query function for chat feature - FIXED for Llama
def query_llm(question, context_data):
    """
    NEW FUNCTION: Handles LLM queries for chat functionality
    FIXED: Configured for Llama model API
    """
    try:
        # Format context including chat history
        context_text = f"""
        Knowledge Graph Context:
        - Selected Node: {context_data['selected_node']}
        - Connected Nodes: {context_data['connected_nodes']}
        - Direction Filter: {context_data['direction']}
        - Total Nodes: {context_data['total_nodes']}
        - Total Edges: {context_data['total_edges']}
        
        Previous Chat:
        {format_chat_history(context_data.get('chat_history', []))}
        
        Current Question: {question}
        """
        
        # FIXED: Simplified payload for Llama (remove OpenAI-specific fields)
        payload = {
            "inputs": context_text,        # Llama typically uses "inputs" instead of "messages"
            "parameters": {
                "max_new_tokens": 500,     # Llama parameter name
                "temperature": 0.7,
                "do_sample": True,
                "return_full_text": False  # Only return new generated text
            }
        }
        
        response = requests.post(
            LLM_API_URL,
            json=payload,
            auth=HTTPBasicAuth(LLM_USERNAME, LLM_PASSWORD),
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            # Handle different Llama response formats
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No response received")
            elif "generated_text" in result:
                return result["generated_text"]
            elif "choices" in result:  # Some Llama APIs use OpenAI format
                return result.get("choices", [{}])[0].get("message", {}).get("content", "No response received")
            else:
                return str(result)  # Fallback to show raw response
        else:
            return f"Llama API Error: {response.status_code} - {response.text}"
        
    except Exception as e:
        return f"Llama query failed: {str(e)}"

# NEW ADDITION: Helper function to format chat history for LLM context
def format_chat_history(chat_history):
    """
    NEW FUNCTION: Formats previous chat messages for LLM context
    Limits to last 5 exchanges to avoid token limits
    """
    if not chat_history:
        return "No previous conversation."
    
    formatted = []
    for role, message in chat_history[-5:]:  # Last 5 messages
        formatted.append(f"{role.title()}: {message}")
    return "\n".join(formatted)

# YOUR ORIGINAL UI LAYOUT START - CLEAN VERSION
st.set_page_config(page_title="Knowledge Graph", layout="wide")

st.title("Knowledge Graph Explorer")

nodes_df, edges_df = load_data()

label_options = ["All"] + sorted(nodes_df["label"].unique().tolist())

selected_label = st.selectbox("Select Node Label", label_options)

# Handle label selection without info messages
if selected_label == "All":
    filtered_nodes = nodes_df
else:
    filtered_nodes = nodes_df[nodes_df.label == selected_label]

# Clean node selection 
if len(filtered_nodes) > 1:
    selected_name = st.selectbox("Select Node Name", 
                                ["View All in Category"] + sorted(filtered_nodes["name"].unique()))
else:
    selected_name = st.selectbox("Select Node Name", 
                                sorted(filtered_nodes["name"].unique()))

# Category view for exploring all nodes in a label
if selected_name == "View All in Category":
    st.markdown(f"### {selected_label} Overview")
    
    # Get all nodes in this category
    category_nodes = filtered_nodes["node_id"].tolist()
    
    # Build a graph showing all category nodes and their relationships
    G = nx.DiGraph()
    
    # Add all nodes in the category with distinct styling
    for _, node in filtered_nodes.iterrows():
        G.add_node(
            to_native_type(node.node_id),
            label=to_native_type(node.label),
            group=to_native_type(node.label),
            title=f"{node['name']} ({node.label})",
            color="#FF6B6B",  # Highlight category nodes in red
            size=40           # Make category nodes larger
        )
    
    # Add edges between category nodes and their connections
    for node_id in category_nodes:
        # Outgoing edges from category nodes (e.g., Languages connecting to other things)
        out_edges = edges_df[edges_df.source_node_id == node_id]
        for _, edge in out_edges.iterrows():
            target = nodes_df[nodes_df.node_id == edge.target_node_id]
            if not target.empty:
                tgt = target.iloc[0]
                G.add_node(
                    to_native_type(tgt.node_id),
                    label=to_native_type(tgt.label),
                    group=to_native_type(tgt.label),
                    title=f"{tgt['name']} ({tgt.label})",
                    color="#ADD8E6" if tgt.label != selected_label else "#FFB6C1"  # Different colors for clarity
                )
                G.add_edge(
                    to_native_type(node_id),
                    to_native_type(tgt.node_id),
                    label=to_native_type(edge.relation_type),
                    title=f"{edge.relation_type}: {filtered_nodes[filtered_nodes.node_id == node_id]['name'].iloc[0]} → {tgt['name']}"
                )
        
        # Incoming edges to category nodes (e.g., Applications connecting to Languages)  
        in_edges = edges_df[edges_df.target_node_id == node_id]
        for _, edge in in_edges.iterrows():
            source = nodes_df[nodes_df.node_id == edge.source_node_id]
            if not source.empty:
                src = source.iloc[0]
                G.add_node(
                    to_native_type(src.node_id),
                    label=to_native_type(src.label),
                    group=to_native_type(src.label),
                    title=f"{src['name']} ({src.label})",
                    color="#90EE90" if src.label != selected_label else "#FFB6C1"  # Different colors for clarity
                )
                G.add_edge(
                    to_native_type(src.node_id),
                    to_native_type(node_id),
                    label=to_native_type(edge.relation_type),
                    title=f"{edge.relation_type}: {src['name']} → {filtered_nodes[filtered_nodes.node_id == node_id]['name'].iloc[0]}"
                )
    
    # Create and display the category overview graph
    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(G)
    net.force_atlas_2based()
    
    # Set physics options with better node visualization for category view
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.02,
          "springLength": 150,
          "springConstant": 0.05
        },
        "maxVelocity": 30,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 200}
      },
      "nodes": {
        "borderWidth": 3,
        "font": {
          "size": 14,
          "face": "Arial",
          "color": "#000000"
        },
        "shape": "dot"
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 1.5
          }
        },
        "color": {
          "inherit": false,
          "color": "#2B7CE9"
        },
        "smooth": {
          "enabled": true,
          "type": "dynamic"
        },
        "width": 2,
        "font": {
          "size": 12,
          "align": "middle"
        }
      },
      "layout": {
        "improvedLayout": true
      }
    }
    """)
    
    html_path = "graph.html"
    net.save_graph(html_path)
    
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    
    st.components.v1.html(html, height=650, scrolling=True)
    
    selected_node = None  # No specific node selected
    selected_node_id = None
    
else:
    # Original single node view
    selected_node = filtered_nodes[filtered_nodes["name"] == selected_name].iloc[0]
    selected_node_id = selected_node.node_id

# Single node view - clean layout
if selected_node is not None:
    direction = st.selectbox(
        "Show edges in which direction?",
        ["Outgoing", "Incoming", "Both"],
        index=2
    )

    st.markdown("### Node Properties")
    st.json(selected_node.properties if hasattr(selected_node, 'properties') and selected_node.properties else {})

    st.markdown("### Graph View")

    # Build subgraph and create visualization
    subgraph = build_subgraph(nodes_df, edges_df, selected_node_id, direction)

    # Create network visualization (your original pyvis setup)
    net = Network(height="600px", width="100%", directed=True)
    net.from_nx(subgraph)
    net.force_atlas_2based()

    # Set physics options for better visualization (your original settings)
    net.set_options("""
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
    """)

    html_path = "graph.html"
    net.save_graph(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # YOUR ORIGINAL HTML DISPLAY - UNCHANGED
    st.components.v1.html(html, height=650, scrolling=True)

# Chat section - clean and simple
st.markdown("---")
st.markdown("### Chat")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Display chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**AI:** {message}")

# Chat input
with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Send")
    
    if submitted and question.strip():
        # Add user message to chat history
        st.session_state.chat_history.append(("user", question))
        
        with st.spinner("Thinking..."):
            # Calculate connected nodes count for context - FIXED logic
            if direction == "Both":
                connected_count = len(edges_df[
                    (edges_df.source_node_id == selected_node_id) | 
                    (edges_df.target_node_id == selected_node_id)
                ])
            elif direction == "Outgoing":
                connected_count = len(edges_df[edges_df.source_node_id == selected_node_id])
            elif direction == "Incoming":
                connected_count = len(edges_df[edges_df.target_node_id == selected_node_id])
            else:
                connected_count = 0
            
            # Prepare context data for LLM
            context_data = {
                "selected_node": selected_name,
                "connected_nodes": connected_count,
                "direction": direction,
                "total_nodes": len(nodes_df),
                "total_edges": len(edges_df),
                "chat_history": st.session_state.chat_history[:-1]  # Previous messages only
            }
            
            # Get LLM response
            response = query_llm(question, context_data)
            
            # Add AI response to chat history
            st.session_state.chat_history.append(("ai", response))
            
            st.rerun()
