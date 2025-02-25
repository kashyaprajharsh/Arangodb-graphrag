import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import os
import tempfile
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
import uuid
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any, List

# Import our multi-agent system
from multi_agent_manager import MedicalGraphSupervisor

# Custom Streamlit callback handler
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler for updating Streamlit UI with intermediate steps"""
    
    def __init__(self):
        """Initialize the handler"""
        self.current_agent_name = None
        self.intermediate_steps = []
        self.debug_mode = True  # Enable debug mode
        self.final_response = None
        self.last_llm_response = None
        super().__init__()

    def clear(self):
        """Clear all intermediate steps and reset the handler state"""
        self.intermediate_steps = []
        self.final_response = None
        self.last_llm_response = None
        print("StreamlitCallbackHandler cleared")
        
    def write_agent_name(self, name: str):
        """Display agent name"""
        self.current_agent_name = name
        self.intermediate_steps.append(f"\n=== Agent: {name} ===")
        
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Display tool execution start"""
        try:
            if serialized is None:
                tool_name = "Unknown Tool"
            else:
                tool_name = serialized.get("name", "Unknown Tool")
            self.intermediate_steps.append(f"\n🔧 Using tool: {tool_name}")
        except Exception as e:
            self.intermediate_steps.append(f"\n🔧 Using tool (error getting name)")
            print(f"Error in on_tool_start: {str(e)}")

    def on_tool_end(self, output: str, **kwargs):
        """Display tool execution result"""
        try:
            self.intermediate_steps.append("\n📤 Tool output:")
            self.intermediate_steps.append("-" * 50)
            # Convert output to string if it's not already
            if not isinstance(output, str):
                try:
                    output = str(output)
                except:
                    output = "Non-string output (could not convert to string)"
            self.intermediate_steps.append(output)
            self.intermediate_steps.append("-" * 50)
        except Exception as e:
            self.intermediate_steps.append("\n📤 Tool output: [Error displaying output]")
            print(f"Error in on_tool_end: {str(e)}")

    def on_agent_action(self, action: Any, **kwargs):
        """Display agent action"""
        try:
            if hasattr(action, 'tool'):
                self.intermediate_steps.append(f"\n🎯 Action: {action.tool}")
                self.intermediate_steps.append("Input:")
                self.intermediate_steps.append("-" * 50)
                # Convert tool_input to string if it's not already
                if not isinstance(action.tool_input, str):
                    try:
                        tool_input = str(action.tool_input)
                    except:
                        tool_input = "Non-string input (could not convert to string)"
                else:
                    tool_input = action.tool_input
                self.intermediate_steps.append(tool_input)
                self.intermediate_steps.append("-" * 50)
            else:
                self.intermediate_steps.append(f"\n🎯 Action: [Unknown action type]")
        except Exception as e:
            self.intermediate_steps.append(f"\n🎯 Action: [Error displaying action]")
            print(f"Error in on_agent_action: {str(e)}")

    def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs):
        """Display when LLM starts processing"""
        try:
            self.intermediate_steps.append("\n🤔 Processing...")
        except Exception as e:
            print(f"Error in on_llm_start: {str(e)}")

    def on_llm_end(self, response, **kwargs):
        """Display final LLM response"""
        try:
            if hasattr(response, 'generations') and response.generations:
                self.intermediate_steps.append("\n Final LLM Response:")
                self.intermediate_steps.append("-" * 50)
                final_text = response.generations[0][0].text
                self.intermediate_steps.append(final_text)
                self.intermediate_steps.append("-" * 50)
                
                # Store the final response
                self.final_response = final_text
                self.last_llm_response = final_text
                
                # Debug: Log the response
                if self.debug_mode:
                    print(f"LLM Response: {final_text[:100]}...")
            else:
                self.intermediate_steps.append("\n Final LLM Response: [No generations found]")
                if self.debug_mode:
                    print("LLM Response: No generations found")
        except Exception as e:
            self.intermediate_steps.append("\n Final LLM Response: [Error displaying response]")
            print(f"Error in on_llm_end: {str(e)}")

    def get_final_response(self):
        """Return the final LLM response"""
        return self.last_llm_response or self.final_response

    def on_tool_error(self, error: str, **kwargs):
        """Display tool errors"""
        try:
            self.intermediate_steps.append(f"\n❌ Error: {error}")
        except Exception as e:
            self.intermediate_steps.append("\n❌ Error: [Error displaying error message]")
            print(f"Error in on_tool_error: {str(e)}")
    
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: list, **kwargs):
        """Handle chat model start"""
        try:
            self.intermediate_steps.append("\n💬 Starting chat model...")
            
            # Debug: Log the messages being sent to the chat model
            if self.debug_mode:
                print(f"Chat model starting with {len(messages)} messages")
                for i, msg in enumerate(messages):
                    msg_type = type(msg).__name__
                    msg_content = getattr(msg, 'content', '[No content]')
                    print(f"  Message {i}: {msg_type} - {msg_content[:50]}...")
        except Exception as e:
            print(f"Error in on_chat_model_start: {str(e)}")
        
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Handle chain start"""
        try:
            # Check if serialized is None or doesn't have 'name'
            if serialized is None:
                chain_name = "Unknown Chain"
            else:
                chain_name = serialized.get("name", "Chain")
            
            self.intermediate_steps.append(f"\n⛓️ Starting chain: {chain_name}")
        except Exception as e:
            # If anything goes wrong, just add a generic message
            self.intermediate_steps.append("\n⛓️ Starting chain (error getting name)")
            print(f"Error in on_chain_start: {str(e)}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Handle chain end"""
        try:
            self.intermediate_steps.append("\n✅ Chain completed")
            
            # Debug: Log the chain outputs
            if self.debug_mode and outputs:
                print(f"Chain completed with outputs: {list(outputs.keys())}")
                for key, value in outputs.items():
                    if isinstance(value, str):
                        print(f"  {key}: {value[:50]}...")
                    else:
                        print(f"  {key}: {type(value).__name__}")
        except Exception as e:
            print(f"Error in on_chain_end: {str(e)}")
    
    def on_text(self, text: str, **kwargs):
        """Handle text output"""
        try:
            self.intermediate_steps.append(f"\n📝 Text: {text}")
            
            # Debug: Log the text
            if self.debug_mode:
                print(f"Text output: {text[:100]}...")
        except Exception as e:
            self.intermediate_steps.append("\n📝 Text: [Error displaying text]")
            print(f"Error in on_text: {str(e)}")
    
    def on_message(self, message, **kwargs):
        """Handle message objects"""
        try:
            # Try to extract content from the message
            message_type = type(message).__name__
            
            if hasattr(message, 'content'):
                content = message.content
                self.intermediate_steps.append(f"\nMessage ({message_type}): {content}")
                
                # Debug: Log the message
                if self.debug_mode:
                    print(f"Message ({message_type}): {content[:100]}...")
            elif hasattr(message, 'tool_call_id'):
                self.intermediate_steps.append(f"\nTool Call ID: {message.tool_call_id}")
            else:
                self.intermediate_steps.append(f"\nMessage ({message_type}): {str(message)}")
                
                # Debug: Log the message
                if self.debug_mode:
                    print(f"Message ({message_type}): {str(message)[:100]}...")
        except Exception as e:
            self.intermediate_steps.append(f"\nMessage (error extracting content): {type(message).__name__}")
            print(f"Error in on_message: {str(e)}")
    
    def on_tool_message(self, message, **kwargs):
        """Specifically handle ToolMessage objects"""
        try:
            # Extract information from the ToolMessage
            message_info = f"\n🛠️ Tool Message:"
            
            if hasattr(message, 'content'):
                message_info += f"\nContent: {message.content}"
            
            if hasattr(message, 'tool_call_id'):
                message_info += f"\nTool Call ID: {message.tool_call_id}"
                
            if hasattr(message, 'name'):
                message_info += f"\nName: {message.name}"
                
            if hasattr(message, 'args'):
                message_info += f"\nArgs: {message.args}"
                
            self.intermediate_steps.append(message_info)
            
            # Debug: Log the tool message
            if self.debug_mode:
                print(f"Tool Message: {message_info}")
        except Exception as e:
            self.intermediate_steps.append(f"\n🛠️ Tool Message (error extracting info): {type(message).__name__}")
            print(f"Error in on_tool_message: {str(e)}")
        
    def get_intermediate_steps(self) -> str:
        """Get all intermediate steps as a string"""
        try:
            # Convert all items to strings first
            string_steps = []
            
            for i, step in enumerate(self.intermediate_steps):
                try:
                    if isinstance(step, str):
                        string_steps.append(step)
                    # Special handling for ToolMessage objects
                    elif hasattr(step, '__class__') and 'Message' in step.__class__.__name__:
                        # Handle different types of message objects
                        if hasattr(step, 'content') and step.content:
                            string_steps.append(f"Message Content: {step.content}")
                        elif hasattr(step, 'tool_call_id'):
                            string_steps.append(f"Tool Call ID: {step.tool_call_id}")
                        elif hasattr(step, 'name') and hasattr(step, 'args'):
                            string_steps.append(f"Tool Call: {step.name}({step.args})")
                        else:
                            string_steps.append(f"Message: {type(step).__name__}")
                    else:
                        # For any other type, try to convert to string
                        string_steps.append(str(step))
                except Exception as e:
                    # If conversion fails, add a placeholder
                    string_steps.append(f"[Item {i}: {type(step).__name__} (conversion error: {str(e)})]")
            
            # Join all strings
            try:
                return "\n".join(string_steps)
            except Exception as e:
                # If joining fails, return an error message
                return f"Error joining intermediate steps: {str(e)}"
        except Exception as e:
            # If anything goes wrong, return a generic error message
            return f"Error processing intermediate steps: {str(e)}"

# Page configuration
st.set_page_config(
    page_title="Medical Graph Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css():
    st.markdown("""
    <style>
    .main-header {
        color: #1E88E5;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #0D47A1;
        font-weight: 500;
        margin-top: 1rem;
    }
    .agent-card {
        background-color: #f7f9fc;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #1E88E5;
    }
    .agent-title {
        color: #1E88E5;
        font-size: 1.3rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        display: block;
    }
    .response-area {
        background-color: #f7f9fc;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        border-left: 5px solid #4CAF50;
    }
    .metric-card {
        background-color: #f1f7fe;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f7f9fc;
        border-radius: 5px 5px 0 0;
        padding: 1rem 2rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    .example-query {
        cursor: pointer;
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        background-color: #f1f7fe;
        transition: background-color 0.3s;
    }
    .example-query:hover {
        background-color: #d9eafa;
    }
    .agent-flow-diagram {
        background-color: #ECEFF1;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #2196F3;
        align-self: flex-end;
    }
    .assistant-message {
        background-color: #F1F8E9;
        border-left: 5px solid #8BC34A;
        align-self: flex-start;
    }
    .agent-selector {
        max-width: 250px;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 300px);
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .welcome-message {
        text-align: center;
        color: #555;
        margin: 2rem 0;
    }
    /* Remove any chat input styling to keep default Streamlit styling */
    </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'supervisor' not in st.session_state:
    st.session_state.supervisor = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'visualization_data' not in st.session_state:
    st.session_state.visualization_data = None
if 'chart_type' not in st.session_state:
    st.session_state.chart_type = 'network'
if 'agent_metrics' not in st.session_state:
    st.session_state.agent_metrics = {
        "aql_query_agent": {"count": 0, "avg_time": 0, "success_count": 0, "failure_count": 0, "success_rate": 0},
        "graph_analysis_agent": {"count": 0, "avg_time": 0, "success_count": 0, "failure_count": 0, "success_rate": 0},
        "patient_data_agent": {"count": 0, "avg_time": 0, "success_count": 0, "failure_count": 0, "success_rate": 0},
        "population_health_agent": {"count": 0, "avg_time": 0, "success_count": 0, "failure_count": 0, "success_rate": 0}
    }
if 'routing_visualization' not in st.session_state:
    st.session_state.routing_visualization = None
if 'streamlit_callback' not in st.session_state:
    st.session_state.streamlit_callback = StreamlitCallbackHandler()
if 'intermediate_steps' not in st.session_state:
    st.session_state.intermediate_steps = ""

# Helper function for extracting visualization data from responses
def extract_visualization_data(response):
    """Extract data for visualization from the response"""
    try:
        # Look for JSON data in the response
        json_pattern = r'```json\s*([\s\S]*?)\s*```'
        import re
        json_match = re.search(json_pattern, response)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            return data
        else:
            # If no JSON found, check if the response itself is JSON
            try:
                data = json.loads(response)
                return data
            except:
                return None
    except Exception as e:
        st.error(f"Error extracting visualization data: {str(e)}")
        return None

# Helper function to create network graph visualization
def create_network_visualization(data):
    if not data:
        return None
    
    try:
        # Create a new graph
        G = nx.Graph()
        
        # If data is a typical graph structure with nodes and edges
        if isinstance(data, dict) and 'nodes' in data and 'edges' in data:
            for node in data['nodes']:
                G.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
            
            for edge in data['edges']:
                G.add_edge(edge['source'], edge['target'], **{k: v for k, v in edge.items() 
                                                        if k not in ['source', 'target']})
        
        # If data is a list of nodes with relationships
        elif isinstance(data, list) and len(data) > 0 and 'connections' in data[0]:
            for item in data:
                G.add_node(item['id'], **{k: v for k, v in item.items() if k not in ['id', 'connections']})
                for conn in item['connections']:
                    G.add_edge(item['id'], conn['id'], **{k: v for k, v in conn.items() if k != 'id'})
        
        # If data is patient-related
        elif isinstance(data, dict) and 'patient' in data:
            # Add patient node
            patient_id = data['patient'].get('_key', 'patient')
            G.add_node(patient_id, type='patient', **data['patient'])
            
            # Add condition nodes and edges
            if 'conditions' in data:
                for i, condition in enumerate(data['conditions']):
                    cond_id = f"condition_{i}"
                    G.add_node(cond_id, type='condition', **condition)
                    G.add_edge(patient_id, cond_id, type='has_condition')
            
            # Add medication nodes and edges
            if 'medications' in data:
                for i, medication in enumerate(data['medications']):
                    med_id = f"medication_{i}"
                    G.add_node(med_id, type='medication', **medication)
                    G.add_edge(patient_id, med_id, type='takes_medication')
        
        # If we can't determine the structure, return None
        else:
            return None
        
        # Create figure
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes with different colors based on node type
        node_colors = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', '')
            if 'patient' in node_type.lower():
                node_colors.append('lightblue')
            elif 'condition' in node_type.lower():
                node_colors.append('salmon')
            elif 'medication' in node_type.lower():
                node_colors.append('lightgreen')
            elif 'provider' in node_type.lower():
                node_colors.append('orange')
            else:
                node_colors.append('gray')
        
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=700, font_size=8, font_weight='bold')
        
        # Save figure to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
        
    except Exception as e:
        st.error(f"Error creating network visualization: {str(e)}")
        return None

# Helper function to create chart visualization based on data
def create_chart_visualization(data, chart_type='bar'):
    if not data:
        return None
    
    try:
        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        # If it's a dictionary with lists
        elif isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
            # Find the first list value and use its length to determine number of rows
            list_key = next(k for k, v in data.items() if isinstance(v, list))
            rows = []
            for i in range(len(data[list_key])):
                row = {}
                for k, v in data.items():
                    if isinstance(v, list) and i < len(v):
                        row[k] = v[i]
                    else:
                        row[k] = v
                rows.append(row)
            df = pd.DataFrame(rows)
        else:
            return None
        
        # Create appropriate chart based on the data and requested type
        if chart_type == 'bar' and 'frequency' in df.columns:
            fig = px.bar(df, x=df.columns[0], y='frequency', 
                         title=f"Frequency Distribution", 
                         color_discrete_sequence=['#0083B8'])
            return fig
        elif chart_type == 'scatter' and len(df.columns) >= 2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                                title=f"Scatter Plot", color_discrete_sequence=['#0083B8'])
                return fig
        elif chart_type == 'pie' and 'frequency' in df.columns:
            fig = px.pie(df, names=df.columns[0], values='frequency',
                         title=f"Distribution", color_discrete_sequence=px.colors.sequential.Blues)
            return fig
        elif chart_type == 'line' and len(df.columns) >= 2:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                fig = px.line(df, x=date_cols[0], y=df.select_dtypes(include=[np.number]).columns[0],
                             title=f"Time Series", color_discrete_sequence=['#0083B8'])
                return fig
        
        # Default to a table if no specific chart can be created
        return df
        
    except Exception as e:
        st.error(f"Error creating chart visualization: {str(e)}")
        return None

# Main function to build the app
def main():
    # Apply custom CSS
    local_css()
    
    # Sidebar - System Control Panel
    st.sidebar.markdown("<h2 class='sub-header'>System Control</h2>", unsafe_allow_html=True)
    
    # System initialization
    if not st.session_state.initialized:
        if st.sidebar.button("Initialize System", key="init_button"):
            with st.sidebar:
                with st.spinner("Initializing Medical Graph System..."):
                    st.session_state.supervisor = initialize_system()
                    st.session_state.initialized = True
                st.success("System initialized successfully!")
    else:
        st.sidebar.success("System is running")
        
        # Add debug mode toggle
        if 'debug_mode' not in st.session_state:
            st.session_state.debug_mode = False
        
        debug_mode = st.sidebar.checkbox("Debug Mode", value=st.session_state.debug_mode)
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode
            if hasattr(st.session_state, 'streamlit_callback'):
                st.session_state.streamlit_callback.debug_mode = debug_mode
            st.rerun()
        
        # Reset button
        if st.sidebar.button("Reset System"):
            st.session_state.supervisor = None
            st.session_state.initialized = False
            st.session_state.chat_history = []
            st.session_state.current_response = ""
            st.session_state.visualization_data = None
            st.session_state.intermediate_steps = ""
            st.session_state.routing_visualization = None
            st.session_state.debug_mode = False
            # Reset agent metrics
            st.session_state.agent_metrics = {
                "aql_query_agent": {"count": 0, "avg_time": 0, "success_count": 0, "failure_count": 0, "success_rate": 0},
                "graph_analysis_agent": {"count": 0, "avg_time": 0, "success_count": 0, "failure_count": 0, "success_rate": 0},
                "patient_data_agent": {"count": 0, "avg_time": 0, "success_count": 0, "failure_count": 0, "success_rate": 0},
                "population_health_agent": {"count": 0, "avg_time": 0, "success_count": 0, "failure_count": 0, "success_rate": 0}
            }
            st.rerun()
    
    # Agent selection in sidebar
    if st.session_state.initialized:
        st.sidebar.markdown("<h3 class='sub-header'>Agent Selection</h3>", unsafe_allow_html=True)
        agent_type = st.sidebar.selectbox(
            "Select specific agent (optional)",
            ["Auto-select", "AQL Query", "Graph Analysis", "Patient Data", "Population Health"],
            key="agent_selector"
        )
        
        # Mapped agent types
        agent_map = {
            "Auto-select": None,
            "AQL Query": "aql_query_agent",
            "Graph Analysis": "graph_analysis_agent",
            "Patient Data": "patient_data_agent",
            "Population Health": "population_health_agent"
        }
        
        # Display agent metrics
        st.sidebar.markdown("<h3 class='sub-header'>Agent Metrics</h3>", unsafe_allow_html=True)
        
        # Create columns for metrics
        col1, col2 = st.sidebar.columns(2)
        
        # Display metrics for each agent
        for i, (agent_name, metrics) in enumerate(st.session_state.agent_metrics.items()):
            # Skip if no usage
            if metrics["count"] == 0:
                continue
                
            # Format the agent name for display
            display_name = agent_name.replace("_agent", "").replace("_", " ").title()
            
            # Alternate columns
            col = col1 if i % 2 == 0 else col2
            
            with col:
                st.markdown(f"<div class='metric-card'><span class='agent-title'>{display_name}</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-value'>{metrics['count']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-label'>Queries</div>", unsafe_allow_html=True)
                
                # Only show additional metrics if there are queries
                if metrics["count"] > 0:
                    st.markdown(f"<div class='metric-label'>Avg Time: {metrics['avg_time']:.2f}s</div>", unsafe_allow_html=True)
                    
                    # Show success rate if available
                    if "success_rate" in metrics:
                        success_color = "#4CAF50" if metrics["success_rate"] > 90 else "#FFC107" if metrics["success_rate"] > 70 else "#F44336"
                        st.markdown(f"<div class='metric-label'>Success: <span style='color:{success_color}'>{metrics['success_rate']:.1f}%</span></div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Example queries
        st.sidebar.markdown("<h3 class='sub-header'>Example Queries</h3>", unsafe_allow_html=True)
        
        example_categories = {
            "AQL Queries": [
                "What collections are available in the database?",
                "Show me a sample of patient records",
                "Count the total number of records in each collection",
                "What are the most common conditions in the database give names of that?",
                "Find all allergies for patients over 65 years old",
                "what is pagerank of the patient 7c2e78bd-52cf-1fce-acc3-0ddd93104abe",
                "Find top 5 providers with unusually high claims for expensive procedures and then detect any anomalies or fraud claims",
                "Find all the patients with having condition Stress (finding) along with thier age name and id",
                "Find the most influential providers based on referral patterns",
                "What is the average path length between primary care physicians?"
            ],
            "Graph Analysis": [
                "which node has the highest degree centrality in the graph"
            ],
            "Auto": [
                "Show me the medical history for patient 7c2e78bd-52cf-1fce-acc3-0ddd93104abe",
                "Find all conditions related to diabetes and their frequency",
                "What medications are commonly prescribed for hypertension?",
                "Find patients who share similar medication patterns and have common conditions",
                "Analyze the treatment pathway for diabetes patients across different providers"
            ]
        }
        
        # Show examples as clickable buttons
        for category, examples in example_categories.items():
            st.sidebar.markdown(f"<b>{category}</b>", unsafe_allow_html=True)
            for ex in examples:
                if st.sidebar.button(ex, key=f"ex_{category}_{ex}"):
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "user", "content": ex})
                    # Process the query
                    process_query(ex, agent_map.get(agent_type))
    
    # Main Area - Tabs
    st.markdown("<h1 class='main-header'>Medical Graph Multi-Agent System</h1>", unsafe_allow_html=True)
    
    # Only show the Chat Interface tab
    st.markdown("<h2 class='sub-header'>Medical Graph Chat</h2>", unsafe_allow_html=True)
    
    # Chat container for displaying messages
    with st.container():
        # Debug info - Remove for production
        # st.write(f"Debug: Chat history has {len(st.session_state.chat_history)} messages")
        
        # Display chat history
        if not st.session_state.chat_history:
            st.markdown("<div class='welcome-message'><h3>👋 Welcome to the Medical Graph Chat!</h3><p>Ask me anything about the medical graph database or select an example query from the sidebar.</p></div>", unsafe_allow_html=True)
        else:
            for i, message in enumerate(st.session_state.chat_history):
                try:
                    if message["role"] == "user":
                        st.markdown(f"<div class='chat-message user-message'><b>You:</b><br>{message['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='chat-message assistant-message'><b>Medical Graph System:</b><br>{message['content']}</div>", unsafe_allow_html=True)
                    
                    # Debug info for each message - Remove for production
                    # st.write(f"Debug - Message {i+1}: Role={message['role']}, Content length={len(message['content'])}")
                except Exception as msg_error:
                    st.error(f"Error displaying message {i+1}: {str(msg_error)}")
                    st.write(f"Message data: {message}")
        
        # Show intermediate steps if available
        if st.session_state.intermediate_steps:
            st.markdown("<h4 class='sub-header'>Processing Steps</h4>", unsafe_allow_html=True)
            with st.expander("Show processing steps", expanded=False):
                try:
                    # Check if intermediate_steps is a string
                    if isinstance(st.session_state.intermediate_steps, str):
                        st.code(st.session_state.intermediate_steps, language="text")
                    else:
                        # If it's not a string, try to convert it
                        try:
                            steps_str = str(st.session_state.intermediate_steps)
                            st.code(steps_str, language="text")
                        except Exception as conv_error:
                            st.error(f"Error converting steps to string: {str(conv_error)}")
                            st.write("Steps type:", type(st.session_state.intermediate_steps))
                except Exception as steps_error:
                    st.error(f"Error displaying intermediate steps: {str(steps_error)}")
                    st.write(f"Steps type: {type(st.session_state.intermediate_steps)}")
        
        # Show routing visualization if available
        # Commenting out the Query Processing Details section as requested
        # if st.session_state.routing_visualization:
        #     st.markdown("<h4 class='sub-header'>Query Processing Details</h4>", unsafe_allow_html=True)
        #     st.markdown("<div class='agent-flow-diagram'>", unsafe_allow_html=True)
        #     try:
        #         st.markdown(st.session_state.routing_visualization, unsafe_allow_html=True)
        #     except Exception as viz_error:
        #         st.error(f"Error displaying routing visualization: {str(viz_error)}")
        #     st.markdown("</div>", unsafe_allow_html=True)
        
        # Processing indicator
        if st.session_state.processing:
            with st.status("Processing your query...", expanded=True) as status:
                st.write("The medical graph agents are analyzing your query...")
                st.spinner("Please wait")
                
                # Add a placeholder for live updates
                steps_placeholder = st.empty()
                
                # This will be updated by the callback handler
                try:
                    if st.session_state.streamlit_callback.intermediate_steps:
                        steps = st.session_state.streamlit_callback.intermediate_steps
                        if len(steps) > 10:
                            steps_to_show = steps[-10:]
                        else:
                            steps_to_show = steps
                            
                        # Ensure all items are strings
                        string_steps = []
                        for step in steps_to_show:
                            if isinstance(step, str):
                                string_steps.append(step)
                            else:
                                try:
                                    string_steps.append(str(step))
                                except:
                                    string_steps.append(f"[Non-string step: {type(step).__name__}]")
                                    
                        steps_placeholder.code("\n".join(string_steps), language="text")
                except Exception as steps_error:
                    steps_placeholder.error(f"Error displaying steps: {str(steps_error)}")
        
        # Show debug information if debug mode is enabled
        if st.session_state.get('debug_mode', False):
            st.markdown("<h4 class='sub-header'>Debug Information</h4>", unsafe_allow_html=True)
            with st.expander("Show debug information", expanded=True):
                st.markdown("### System State")
                st.write("Initialized:", st.session_state.initialized)
                st.write("Processing:", st.session_state.processing)
                
                st.markdown("### Agent Metrics")
                for agent_name, metrics in st.session_state.agent_metrics.items():
                    if metrics["count"] > 0:
                        st.write(f"**{agent_name}**:", metrics)
                
                st.markdown("### Last Response")
                if hasattr(st.session_state, 'current_response'):
                    st.code(st.session_state.current_response[:500] + "..." if len(st.session_state.current_response) > 500 else st.session_state.current_response)
                
                st.markdown("### Callback Handler")
                if hasattr(st.session_state, 'streamlit_callback'):
                    st.write("Current Agent:", st.session_state.streamlit_callback.current_agent_name)
                    st.write("Steps Count:", len(st.session_state.streamlit_callback.intermediate_steps))
                
                st.markdown("### Session State Keys")
                st.write(list(st.session_state.keys()))
    
    # Chat input
    if st.session_state.initialized:
        # Use default Streamlit chat input without custom styling
        query = st.chat_input("Ask something about the medical graph data...")
        if query:
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            # Get the selected agent type
            agent_type = agent_map.get(st.session_state.agent_selector)
            # Process the query
            process_query(query, agent_type)
    else:
        st.info("Please initialize the system using the button in the sidebar to start chatting.")

# Function to initialize the system
def initialize_system():
    """Initialize the Medical Graph Multi-Agent System"""
    try:
        # Create supervisor with Streamlit callback
        from multi_agent_manager import MedicalGraphSupervisor
        
        # Set debug mode on the callback handler
        if 'debug_mode' in st.session_state:
            st.session_state.streamlit_callback.debug_mode = st.session_state.debug_mode
        
        # Create a custom supervisor that uses our Streamlit callback
        class StreamlitMedicalGraphSupervisor(MedicalGraphSupervisor):
            def __init__(self):
                # Initialize with default callback first
                super().__init__()
                # Then replace the callback handler with our Streamlit callback
                self.callback_handler = st.session_state.streamlit_callback
                # Recreate the workflow with our callback
                try:
                    self.workflow = self._create_workflow()
                except Exception as workflow_error:
                    st.error(f"Error creating workflow: {str(workflow_error)}")
                    print(f"Error creating workflow: {str(workflow_error)}")
            
            # Override process_query to handle errors better
            def process_query(self, query: str, agent_type: str = None) -> str:
                """
                Process a user query through the multi-agent workflow with better error handling.
                
                Args:
                    query: User's natural language query
                    agent_type: Type of agent to process the query
                    
                Returns:
                    Final response
                """
                try:
                    # Log the query and agent type
                    print(f"\n{'='*50}")
                    print(f"StreamlitMedicalGraphSupervisor processing query: '{query}'")
                    print(f"Using agent_type: {agent_type}")
                    print(f"{'='*50}")
                    
                    # Use the parent class's process_query method with direct agent invocation
                    response = super().process_query(query, agent_type)
                    
                    # Debug the response
                    print(f"\n{'='*50}")
                    print(f"Response type: {type(response)}")
                    if isinstance(response, str):
                        print(f"Response content (first 100 chars): {response[:100]}...")
                        if not response:
                            print("WARNING: Empty string response received")
                            return "No response was generated. Please try a different query or agent."
                    else:
                        print(f"Non-string response: {response}")
                        # Try to convert non-string response to string
                        try:
                            response = str(response)
                        except Exception as conv_error:
                            print(f"Error converting response to string: {str(conv_error)}")
                            return "Error: Received a response that couldn't be displayed. Please try again."
                    print(f"{'='*50}")
                    
                    return response
                except Exception as e:
                    error_msg = f"Error in supervisor process_query: {str(e)}"
                    print(error_msg)
                    import traceback
                    print(traceback.format_exc())
                    return error_msg
        
        supervisor = StreamlitMedicalGraphSupervisor()
        return supervisor
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        print(f"Error initializing system: {str(e)}")
        return None

# Function to process a query
def process_query(query, agent_type=None):
    """Process a query and update the session state with the response"""
    if not st.session_state.initialized or not st.session_state.supervisor:
        st.error("System not initialized. Please initialize the system first.")
        return
    
    try:
        # Set processing flag
        st.session_state.processing = True
        st.session_state.current_response = ""
        st.session_state.visualization_data = None
        
        # Clear previous intermediate steps
        if hasattr(st.session_state.streamlit_callback, 'clear'):
            st.session_state.streamlit_callback.clear()
        else:
            # If clear method doesn't exist, create a new instance
            st.session_state.streamlit_callback = StreamlitCallbackHandler()
            
        st.session_state.intermediate_steps = ""
        
        # Create routing visualization 
        agent_map = {
            "aql_query_agent": "AQL Query Agent",
            "graph_analysis_agent": "Graph Analysis Agent", 
            "patient_data_agent": "Patient Data Agent",
            "population_health_agent": "Population Health Agent"
        }
        
        # Capture start time
        start_time = time.time()
        
        # Process the query
        try:
            # Process the query and get the response, passing the agent_type if specified
            print(f"Sending query to supervisor: '{query}' with agent_type: {agent_type}")
            response = st.session_state.supervisor.process_query(query, agent_type)
            
            # Debug the response
            print(f"Response received from supervisor: {type(response)}")
            if isinstance(response, str):
                print(f"Response content (first 100 chars): {response[:100] if response else 'Empty string'}")
                if not response.strip():
                    print("WARNING: Empty or whitespace-only response received")
                    response = "No response was generated. Please try a different query or agent."
            else:
                print(f"Non-string response: {response}")
                # Try to convert non-string response to string
                try:
                    response = str(response)
                except Exception as conv_error:
                    print(f"Error converting response to string: {str(conv_error)}")
                    response = "Error: Received a response that couldn't be displayed. Please try again."
            
            # Get intermediate steps - with extra error handling
            try:
                # Use our safe method to get intermediate steps
                st.session_state.intermediate_steps = get_safe_intermediate_steps(st.session_state.streamlit_callback)
            except Exception as steps_error:
                error_msg = f"Error capturing intermediate steps: {str(steps_error)}"
                print(error_msg)
                st.session_state.intermediate_steps = error_msg
            
        except Exception as query_error:
            error_msg = f"Error processing query: {str(query_error)}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            response = error_msg
            # Try to get any intermediate steps that were captured before the error
            try:
                # Use our safe method to get intermediate steps
                st.session_state.intermediate_steps = get_safe_intermediate_steps(st.session_state.streamlit_callback)
                st.session_state.intermediate_steps += f"\n\n❌ ERROR: {str(query_error)}"
            except Exception as steps_error:
                st.session_state.intermediate_steps = f"Error occurred: {str(query_error)}\nFailed to capture steps: {str(steps_error)}"
        
        # Capture end time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Extract routing decisions for visualization
        routing_info = None
        if hasattr(st.session_state.supervisor, 'workflow') and hasattr(st.session_state.supervisor.workflow, 'state'):
            if 'context' in st.session_state.supervisor.workflow.state:
                routing_info = st.session_state.supervisor.workflow.state['context'].get('query_classification')
        
        # Generate routing visualization
        if agent_type:
            # If direct agent invocation was used
            agent_display_name = agent_map.get(agent_type, agent_type)
            st.session_state.routing_visualization = f"""
            <h4>Direct Agent Invocation</h4>
            <p><strong>Agent Used:</strong> {agent_display_name}</p>
            <p><strong>Processing Time:</strong> {processing_time:.2f} seconds</p>
            """
            
            # Update agent metrics
            if agent_type in st.session_state.agent_metrics:
                metrics = st.session_state.agent_metrics[agent_type]
                metrics["count"] += 1
                # Update the average processing time
                metrics["avg_time"] = (metrics["avg_time"] * (metrics["count"] - 1) + processing_time) / metrics["count"]
                # Add success/failure tracking
                if "success_count" not in metrics:
                    metrics["success_count"] = 0
                    metrics["failure_count"] = 0
                
                # Consider it a success if no error in response
                if not response.startswith("Error"):
                    metrics["success_count"] += 1
                else:
                    metrics["failure_count"] += 1
                
                # Calculate success rate
                metrics["success_rate"] = (metrics["success_count"] / metrics["count"]) * 100
        
        elif routing_info:
            # If workflow with query analyzer was used
            primary_agent = routing_info.get('next_action', 'unknown')
            requires_multi = routing_info.get('requires_multi_agent', False)
            additional_agents = routing_info.get('additional_agents', [])
            
            agent_display_name = agent_map.get(primary_agent, primary_agent)
            
            # Create a simple HTML visualization
            st.session_state.routing_visualization = f"""
            <h4>Query Routing Decision</h4>
            <p><strong>Primary Agent:</strong> {agent_display_name}</p>
            <p><strong>Multi-Agent Analysis:</strong> {'Yes' if requires_multi else 'No'}</p>
            <p><strong>Processing Time:</strong> {processing_time:.2f} seconds</p>
            """
            
            if requires_multi and additional_agents:
                st.session_state.routing_visualization += "<p><strong>Additional Agents:</strong></p><ul>"
                for agent in additional_agents:
                    st.session_state.routing_visualization += f"<li>{agent_map.get(agent, agent)}</li>"
                st.session_state.routing_visualization += "</ul>"
            
            # Update agent metrics for the primary agent
            if primary_agent in st.session_state.agent_metrics:
                metrics = st.session_state.agent_metrics[primary_agent]
                metrics["count"] += 1
                # Update the average processing time
                metrics["avg_time"] = (metrics["avg_time"] * (metrics["count"] - 1) + processing_time) / metrics["count"]
                
                # Add success/failure tracking
                if "success_count" not in metrics:
                    metrics["success_count"] = 0
                    metrics["failure_count"] = 0
                
                # Consider it a success if no error in response
                if not response.startswith("Error"):
                    metrics["success_count"] += 1
                else:
                    metrics["failure_count"] += 1
                
                # Calculate success rate
                metrics["success_rate"] = (metrics["success_count"] / metrics["count"]) * 100
        
        # Extract potential visualization data from response
        try:
            st.session_state.visualization_data = extract_visualization_data(response)
            if st.session_state.visualization_data:
                if 'nodes' in st.session_state.visualization_data and 'edges' in st.session_state.visualization_data:
                    st.session_state.chart_type = 'network'
                elif 'values' in st.session_state.visualization_data:
                    data_type = st.session_state.visualization_data.get('type', 'bar')
                    st.session_state.chart_type = data_type
        except Exception as viz_error:
            print(f"Error extracting visualization data: {str(viz_error)}")
        
        # Update the response
        st.session_state.current_response = response
        
        # Add response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear processing flag before rerun
        st.session_state.processing = False
        
        # Rerun to update UI immediately
        st.rerun()
        
    except Exception as e:
        st.error(f"Error in process_query function: {str(e)}")
        # Add error to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": f"Error processing query: {str(e)}"})
    finally:
        # Clear processing flag
        st.session_state.processing = False

# Function to safely get intermediate steps
def get_safe_intermediate_steps(callback_handler):
    """Safely get intermediate steps from the callback handler"""
    if callback_handler is None:
        return "No callback handler available"
        
    try:
        # First try the normal method
        print("Attempting to get intermediate steps from callback handler")
        steps = callback_handler.get_intermediate_steps()
        print(f"Successfully retrieved {len(steps) if isinstance(steps, str) else 'non-string'} steps")
        return steps
    except Exception as e:
        # If that fails, try a more direct approach
        print(f"Error getting intermediate steps: {str(e)}")
        try:
            # Check if intermediate_steps exists
            if not hasattr(callback_handler, 'intermediate_steps'):
                return f"Callback handler has no intermediate_steps attribute: {str(e)}"
                
            # Log the number of steps
            num_steps = len(callback_handler.intermediate_steps)
            print(f"Found {num_steps} intermediate steps")
            
            # Convert each step individually
            steps_list = []
            for i, step in enumerate(callback_handler.intermediate_steps):
                try:
                    if isinstance(step, str):
                        steps_list.append(step)
                    # Special handling for ToolMessage objects
                    elif hasattr(step, '__class__') and 'Message' in step.__class__.__name__:
                        if hasattr(step, 'content') and step.content:
                            steps_list.append(f"Message {i}: {step.content}")
                        elif hasattr(step, 'tool_call_id'):
                            steps_list.append(f"Tool Call {i}: {step.tool_call_id}")
                        else:
                            steps_list.append(f"Message {i}: {type(step).__name__}")
                    else:
                        steps_list.append(f"Step {i}: {str(step)}")
                except Exception as step_error:
                    steps_list.append(f"Step {i}: [conversion error: {str(step_error)}]")
            
            # Join the steps
            try:
                result = "\n".join(steps_list)
                print(f"Successfully joined {len(steps_list)} steps")
                return result
            except Exception as join_error:
                return f"Error joining steps: {str(join_error)}"
        except Exception as fallback_error:
            # If all else fails, just report the error
            return f"Error getting intermediate steps: {str(e)}, fallback error: {str(fallback_error)}"

if __name__ == "__main__":
    main() 