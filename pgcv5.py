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

# YOUR ORIGINAL HELPER FUNCTION - UNCHANGED
def to_native_type(value):
    if pd.isna(value):
        return ""
    return str(value)

# YOUR ORIGINAL BUILD SUBGRAPH FUNCTION - UNCHANGED
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

# NEW ADDITION: Helper function to get actual graph data for LLM context
def get_node_relationships(node_name, nodes_df, edges_df, max_relationships=10):
    """
    NEW FUNCTION: Extracts actual relationships from the knowledge graph
    Returns real data about the node and its connections
    """
    try:
        # Find the node
        node_row = nodes_df[nodes_df["name"] == node_name]
        if node_row.empty:
            return f"Node '{node_name}' not found in the knowledge graph."
        
        node = node_row.iloc[0]
        node_id = node.node_id
        
        relationships = []
        relationships.append(f"Node: {node_name} (Type: {node.label})")
        
        # Get outgoing relationships
        outgoing = edges_df[edges_df.source_node_id == node_id]
        if not outgoing.empty:
            relationships.append("\nOutgoing Relationships:")
            for _, edge in outgoing.head(max_relationships).iterrows():
                target = nodes_df[nodes_df.node_id == edge.target_node_id]
                if not target.empty:
                    target_name = target.iloc[0]['name']
                    target_label = target.iloc[0].label
                    relationships.append(f"  → {target_name} ({target_label}) via '{edge.relation_type}'")
        
        # Get incoming relationships
        incoming = edges_df[edges_df.target_node_id == node_id]
        if not incoming.empty:
            relationships.append("\nIncoming Relationships:")
            for _, edge in incoming.head(max_relationships).iterrows():
                source = nodes_df[nodes_df.node_id == edge.source_node_id]
                if not source.empty:
                    source_name = source.iloc[0]['name']
                    source_label = source.iloc[0].label
                    relationships.append(f"  ← {source_name} ({source_label}) via '{edge.relation_type}'")
        
        # Add node properties if available
        if hasattr(node, 'properties') and node.properties:
            relationships.append(f"\nNode Properties: {node.properties}")
        
        return "\n".join(relationships)
        
    except Exception as e:
        return f"Error retrieving relationships: {str(e)}"

# NEW ADDITION: Helper function for category relationships
def get_category_relationships(category_label, nodes_df, edges_df, max_nodes=5):
    """
    NEW FUNCTION: Gets actual relationships for all nodes in a category
    """
    try:
        if category_label == "All":
            category_nodes = nodes_df.head(max_nodes)  # Limit for performance
            description = "Sample nodes from entire graph"
        else:
            category_nodes = nodes_df[nodes_df.label == category_label].head(max_nodes)
            description = f"Nodes in '{category_label}' category"
        
        relationships = [f"{description}:\n"]
        
        for _, node in category_nodes.iterrows():
            node_data = get_node_relationships(node['name'], nodes_df, edges_df, max_relationships=3)
            relationships.append(node_data)
            relationships.append("-" * 40)
        
        return "\n".join(relationships)
        
    except Exception as e:
        return f"Error retrieving category data: {str(e)}"

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

# NEW ADDITION: LLM query function for chat feature - FIXED for Llama with REAL graph data
def query_llm(question, context_data):
    """
    NEW FUNCTION: Handles LLM queries with ACTUAL knowledge graph data
    FIXED: Now sends real node relationships and graph data
    """
    try:
        # Get actual graph data based on the question and context
        if "selected_node" in context_data and context_data["selected_node"]:
            if context_data.get("direction") == "Category Overview":
                # Category view - get relationships for multiple nodes
                graph_data = get_category_relationships(
                    context_data["selected_node"].replace("All nodes in '", "").replace("' category", ""),
                    context_data.get("nodes_df"), 
                    context_data.get("edges_df")
                )
            else:
                # Single node view - get specific node relationships
                graph_data = get_node_relationships(
                    context_data["selected_node"],
                    context_data.get("nodes_df"),
                    context_data.get("edges_df")
                )
        else:
            graph_data = "No specific node data available."
        
        # Format context with REAL graph data - HYBRID approach
        context_text = f"""
        You are analyzing a Knowledge Graph. Provide a hybrid response:
        
        1. FIRST: Check if question relates to data below and answer from graph
        2. THEN: Add general knowledge context to help user understand
        
        ACTUAL KNOWLEDGE GRAPH DATA:
        {graph_data}

        INSTRUCTIONS: 
        - Start with "In your knowledge graph:" if relevant data found
        - Then add "Generally:" for educational context
        - If no graph data, just provide general knowledge

        User Question: {question}
        """
        
        # Simplified payload for Llama
        payload = {
            "inputs": context_text,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.3,  # Lower temperature for more factual responses
                "do_sample": True,
                "return_full_text": False
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

# ============================================================================
# MAIN APPLICATION LOGIC - YOUR ORIGINAL UI LAYOUT STARTS HERE
# ============================================================================

# YOUR ORIGINAL UI LAYOUT START - CLEAN VERSION
st.set_page_config(page_title="Knowledge Graph", layout="wide")

st.title("Knowledge Graph Explorer")

nodes_df, edges_df = load_data()

# Clean up label options - remove empty/null labels and sort properly
valid_labels = nodes_df["label"].dropna().unique()
valid_labels = [label for label in valid_labels if str(label).strip() and str(label) != 'nan']
label_options = sorted(valid_labels)

selected_label = st.selectbox("Select Category", label_options, help="Choose a category to explore")

# Handle category selection
filtered_nodes = nodes_df[nodes_df.label == selected_label]

# Clean node selection - remove "View All in Category" if only few nodes
if len(filtered_nodes) <= 3:
    # If 3 or fewer nodes, just show individual nodes
    node_options = sorted(filtered_nodes["name"].unique())
    selected_name = st.selectbox("Select Specific Node", node_options, help="Choose a specific node to focus on")
    show_category_view = False
else:
    # If many nodes, offer category overview option
    node_options = ["📊 Category Overview"] + sorted(filtered_nodes["name"].unique())
    selected_name = st.selectbox("Select Node", node_options, help="Choose category overview or specific node")
    show_category_view = (selected_name == "📊 Category Overview")

# Category overview or single node view
if show_category_view:
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
    direction = None  # No direction filter in category view
    
else:
    # Single node view
    selected_node = filtered_nodes[filtered_nodes["name"] == selected_name].iloc[0]
    selected_node_id = selected_node.node_id

# Single node controls and display
if not show_category_view and 'selected_node' in locals():
    direction = st.selectbox(
        "Connection Direction",
        ["Both", "Outgoing", "Incoming"],
        help="Both: all connections, Outgoing: from this node, Incoming: to this node"
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

# Compact chat section
st.markdown("---")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### Chat")
with col2:
    if st.button("Clear"):
        st.session_state.chat_history = []
        st.rerun()

# Compact chat display - limit height
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Show only last 4 exchanges to save space
recent_chat = st.session_state.chat_history[-8:] if len(st.session_state.chat_history) > 8 else st.session_state.chat_history

with st.container():
    for role, message in recent_chat:
        if role == "user":
            st.markdown(f"**Q:** {message}")
        else:
            # Truncate long responses
            display_msg = message[:200] + "..." if len(message) > 200 else message
            st.markdown(f"**A:** {display_msg}")

# Compact input
question = st.text_input("Ask about your graph:", key="chat_input")
if st.button("Send") and question.strip():
    # Add user message and get response
    st.session_state.chat_history.append(("user", question))
    
    with st.spinner("Thinking..."):
        # Simplified context preparation
        if not show_category_view and 'selected_node' in locals():
            context_data = {
                "selected_node": selected_name,
                "nodes_df": nodes_df,
                "edges_df": edges_df,
                "chat_history": st.session_state.chat_history[:-1]
            }
        else:
            context_data = {
                "selected_node": f"All nodes in '{selected_label}' category",
                "nodes_df": nodes_df,
                "edges_df": edges_df,
                "chat_history": st.session_state.chat_history[:-1]
            }
        
        response = query_llm(question, context_data)
        st.session_state.chat_history.append(("ai", response))
        st.rerun()
