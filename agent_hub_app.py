import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import asyncio
from datetime import datetime
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO
import uuid
from typing import Dict, Any, List, TypedDict
from mem0 import Memory
from evaluation import *
# Set page configuration
st.set_page_config(
    page_title="MediGraph Consilium",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import graph cache first to ensure it's initialized before other modules
from graph_cache import get_graph, get_arango_graph, get_db

# Pre-load the graph in the background when the app starts
if 'graph_initialized' not in st.session_state:
    with st.spinner("Initializing medical graph database..."):
        # Initialize all cached objects
        db = get_db()
        G = get_graph()
        arango_g = get_arango_graph()
        st.session_state.graph_initialized = True
        st.success("Graph database initialized successfully!")

# Import our agents
from aql_agent import run_aql_agent, run_aql_agent_with_stream
from graph_agent import run_graph_analysis_agent, run_graph_analysis_agent_with_stream
from patient_data_agent import run_patient_data_agent, run_patient_data_agent_with_stream
from population_health_agent import run_population_health_agent, run_population_health_agent_with_stream
from supervisor_agent import run_supervisor, run_supervisor_async
from callback import CustomConsoleCallbackHandler

class AgentState(TypedDict):
    messages: list  
    context: Dict
    internal_state: Dict
    callback: Any
    mem0_user_id: str
    agent_id:str

# Custom Streamlit callback handler
class StreamlitCallbackHandler(CustomConsoleCallbackHandler):
    """Callback handler for updating Streamlit UI with intermediate steps"""
    
    def __init__(self, agent_id=None):
        """Initialize the handler"""
        self.current_agent_name = None
        self.intermediate_steps = []
        self.debug_mode = True  # Enable debug mode
        self.final_response = None
        self.last_llm_response = None
        self.agent_id = agent_id
        super().__init__()

    def clear(self):
        """Clear all intermediate steps and reset the handler state"""
        self.intermediate_steps = []
        self.final_response = None
        self.last_llm_response = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log when LLM starts"""
        self.intermediate_steps.append(f"\n🧠 LLM thinking...")
        if self.agent_id and self.agent_id in st.session_state:
            st.session_state[f"debug_{self.agent_id}"] += f"\n🧠 LLM thinking..."
        super().on_llm_start(serialized, prompts, **kwargs)
    
    def on_llm_end(self, response, **kwargs):
        """Log when LLM ends"""
        content = response.generations[0][0].text if hasattr(response, 'generations') else str(response)
        self.last_llm_response = content
        self.intermediate_steps.append(f"\n🤔 Processing...\n\n Final LLM Response:\n--------------------------------------------------\n{content}\n--------------------------------------------------\n")
        if self.agent_id and self.agent_id in st.session_state:
            st.session_state[f"debug_{self.agent_id}"] += f"\n🤔 Processing...\n\n Final LLM Response:\n--------------------------------------------------\n{content}\n--------------------------------------------------\n"
        super().on_llm_end(response, **kwargs)
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Log when tool starts"""
        tool_name = serialized.get("name", "unknown_tool")
        self.intermediate_steps.append(f"\n🔧 Using tool: {tool_name}")
        if self.agent_id and self.agent_id in st.session_state:
            st.session_state[f"debug_{self.agent_id}"] += f"\n🔧 Using tool: {tool_name}"
        super().on_tool_start(serialized, input_str, **kwargs)
    
    def on_tool_end(self, output, **kwargs):
        """Log when tool ends"""
        self.intermediate_steps.append(f"\n📤 Tool output:\n--------------------------------------------------\n{output}\n--------------------------------------------------\n")
        if self.agent_id and self.agent_id in st.session_state:
            st.session_state[f"debug_{self.agent_id}"] += f"\n📤 Tool output:\n--------------------------------------------------\n{output}\n--------------------------------------------------\n"
        super().on_tool_end(output, **kwargs)
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Log when chain starts"""
        chain_name = serialized.get("name", "unknown_chain")
        self.intermediate_steps.append(f"\n> Entering new {chain_name} chain...")
        if self.agent_id and self.agent_id in st.session_state:
            st.session_state[f"debug_{self.agent_id}"] += f"\n> Entering new {chain_name} chain..."
        super().on_chain_start(serialized, inputs, **kwargs)
    
    def on_chain_end(self, outputs, **kwargs):
        """Log when chain ends"""
        self.intermediate_steps.append(f"\n> Finished chain.")
        if self.agent_id and self.agent_id in st.session_state:
            st.session_state[f"debug_{self.agent_id}"] += f"\n> Finished chain."
        super().on_chain_end(outputs, **kwargs)
        
    def write_agent_name(self, name: str):
        """Display agent name"""
        self.current_agent_name = name
        self.intermediate_steps.append(f"\n=== Agent: {name} ===")
        if self.agent_id and self.agent_id in st.session_state:
            st.session_state[f"debug_{self.agent_id}"] += f"\n=== Agent: {name} ==="
    
    def get_intermediate_steps(self) -> str:
        """Get all intermediate steps as a single string"""
        return "\n".join(self.intermediate_steps)

# Apply custom CSS
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f5f7f9;
        }
        /* Keep main content area wide but not sidebar */
        .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1rem;
        }
        /* Don't modify sidebar */
        .css-1d391kg, .css-1544g2n {
            padding-top: unset;
        }
        /* Don't modify chat input */
        .stChatInput, .stChatInputContainer {
            max-width: unset !important;
        }
        /* Agent cards styling */
        .agent-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
            height: 100%;
        }
        .agent-card:hover {
            transform: translateY(-5px);
        }
        .agent-title {
            color: #1E88E5;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .agent-description {
            color: #424242;
            font-size: 16px;
            margin-bottom: 15px;
        }
        .agent-capabilities {
            color: #616161;
            font-size: 14px;
            margin-bottom: 20px;
        }
        /* Chat messages styling */
        .user-message {
            background-color: #E3F2FD;
            padding: 10px 15px;
            border-radius: 15px 15px 0 15px;
            margin: 10px 0;
            max-width: 80%;
            align-self: flex-end;
            margin-left: auto;
        }
        .agent-message {
            background-color: #F5F5F5;
            padding: 10px 15px;
            border-radius: 15px 15px 15px 0;
            margin: 10px 0;
            max-width: 80%;
        }
        .message-container {
            display: flex;
            flex-direction: column;
            margin-bottom: 10px;
        }
        .agent-icon {
            font-size: 40px;
            margin-bottom: 10px;
        }
        .back-button {
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# Agent information
agent_info = {
    "aql": {
        "name": "AQL Query Agent",
        "icon": "🔍",
        "description": "Expert medical data analyst specializing in ArangoDB graph queries.",
        "capabilities": [
            "Convert natural language medical questions into precise AQL queries",
            "Execute queries against the SYNTHEA_P100 database",
            "Interpret results in a medically meaningful way",
            "Provide clear, structured responses for medical professionals"
        ],
        "example_queries": [
            "How many encounters does patient 7c2e78bd-52cf-1fce-acc3-0ddd93104abe have?",
            "What is the distribution of healthcare expenses across all patients?",
            "How many patients have Otitis media as a condition?",
            "What is the breakdown of patients by race and ethnicity?",
            "List all patients with healthcare expenses over $3,000",
            "What is the average age of patients in the database?",
            "What are the most common conditions in the database?",
            "How many patients have medication reviews?",
            "What is the distribution of patient ages by gender?",
            "Which medications are most commonly prescribed?"
        ],
        "function": run_aql_agent,
        "stream_function": run_aql_agent_with_stream,
        "color": "#1E88E5"
    },
    "graph": {
        "name": "Graph Analysis Agent",
        "icon": "📊",
        "description": "Expert graph analyst specializing in medical network analysis.",
        "capabilities": [
            "Apply advanced graph algorithms to medical networks",
            "Identify key patterns, hubs, and influences in the medical graph",
            "Discover hidden relationships and pathways",
            "Extract actionable insights from complex network structures"
        ],
        "example_queries": [
            "Who is the most influential node?",
            "Which node has highest degree of centrality in graph?",
            "What is the shortest path between the most influential node and the rest of the graph?",
            "Fetch the highest pagerank node and value",
            "Which nodes are the most connected?",
            "Which node has the highest betweenness centrality score? Use a k value of 10",
            "Which node has the highest closeness centrality score? Use a k value of 10",
            "Which node has the highest eigenvector centrality score? Use a k value of 10",
            "How strongly connected is the network? Used connected components",
            
            
        ],
        "function": run_graph_analysis_agent,
        "stream_function": run_graph_analysis_agent_with_stream,
        "color": "#43A047"
    },
    "patient": {
        "name": "Patient Data Agent",
        "icon": "👤",
        "description": "Clinical data specialist focused on individual patient analysis.",
        "capabilities": [
            "Analyze comprehensive patient medical histories",
            "Identify patterns in individual patient care",
            "Evaluate treatment effectiveness for specific patients",
            "Find similar patients for comparative analysis"
        ],
        "example_queries": [
            "Show me the complete medical history for patient Shila857 Kshlerin58 (ID: 7c2e78bd-52cf-1fce-acc3-0ddd93104abe)",
            "What medications have been prescribed to this patient during their medication reviews?",
            "Compare this patient's healthcare expenses ($3,672.68) with similar patients",
            "What is the timeline of encounters and conditions for this patient?",
            "Show me all encounters and procedures for this patient",
            "Analyze the complete medical history and risk factors for patient 7c2e78bd-52cf-1fce-acc3-0ddd93104abe",
            "What treatments has this patient received for their chronic conditions?",
            "Compare this patient's care plan with similar Hispanic patients in Massachusetts",
            "What are the potential risk factors for this patient?",
            "Summarize the patient's medication history and effectiveness"
        ],
        "function": run_patient_data_agent,
        "stream_function": run_patient_data_agent_with_stream,
        "color": "#FB8C00"
    },
    "population": {
        "name": "Population Health Agent",
        "icon": "👥",
        "description": "Population health analyst specializing in medical data trends.",
        "capabilities": [
            "Identify patterns and trends across large patient populations",
            "Analyze condition prevalence and distribution",
            "Evaluate treatment effectiveness at scale",
            "Discover correlations between medical factors"
        ],
        "example_queries": [
            "What are the most common treatment patterns for medication review patients?",
            "What is the average healthcare expense and coverage ratio across all patients?",
            "Analyze the cost-effectiveness of different treatment approaches for Otitis media",
            "What is the average healthcare expense and coverage ratio across all patients?",
            "How many Hispanic patients are there in database?",
            "What is the distribution of healthcare costs by race and ethnicity?",
            "Compare healthcare coverage between different ZIP codes in database",
            "What percentage of patients require regular medication reviews?",
            "Which conditions account for the highest percentage of total healthcare costs?",
            "Identify opportunities to reduce medication costs through alternative treatments",
            "What are the demographic trends in condition prevalence?"
        ],
        "function": run_population_health_agent,
        "stream_function": run_population_health_agent_with_stream,
        "color": "#8E24AA"
    },
    "supervisor": {
        "name": "Supervisor Agent",
        "icon": "🤖",
        "description": "Medical data analysis supervisor coordinating specialized agents.",
        "capabilities": [
            "Analyze medical data questions and route to appropriate specialists",
            "Coordinate between multiple specialized agents",
            "Synthesize information from different medical perspectives",
            "Provide comprehensive answers to complex medical queries"
        ],
        "example_queries": [
            "Who is the most influential node?",
            "Which node has highest degree of centrality in graph?",
            "What are the most common treatment patterns for medication review patients?",
            "What is the average healthcare expense and coverage ratio across all patients?",
            "How many encounters does patient 7c2e78bd-52cf-1fce-acc3-0ddd93104abe have?",
            "What is the distribution of healthcare expenses across all patients?",
            "Show me the complete medical history for patient Shila857 Kshlerin58 (ID: 7c2e78bd-52cf-1fce-acc3-0ddd93104abe)",
            "What medications have been prescribed to this patient during their medication reviews?",

        ],
        "function": run_supervisor,
        "stream_function": run_supervisor_async,
        "color": "#F44336"
    }
}

# Initialize session state
def init_session_state():
    # Initialize page state
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    # Initialize chat history for each agent
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {agent_id: [] for agent_id in agent_info.keys()}
    
    # Initialize debug info for each agent
    for agent_id in agent_info.keys():
        if f"debug_{agent_id}" not in st.session_state:
            st.session_state[f"debug_{agent_id}"] = ""
        if f"processing_{agent_id}" not in st.session_state:
            st.session_state[f"processing_{agent_id}"] = False
        # Initialize evaluation state for each agent
        if f"evaluation_{agent_id}" not in st.session_state:
            st.session_state[f"evaluation_{agent_id}"] = None
    
    # Initialize current agent
    if "current_agent" not in st.session_state:
        st.session_state.current_agent = None
    
    # Initialize memory
    if "memory" not in st.session_state:
        st.session_state.memory = Memory.from_config(config_dict={
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                }
            },
            "custom_prompt": """
                Please extract entities containing healthcare-related information such as patient details, diagnoses, treatments, medications, medical encounters, and any relevant healthcare events. Additionally, please handle analytical queries regarding healthcare statistics, such as averages, percentages, and cost breakdowns. Each input should be processed to return structured data that can be stored in a system or used to answer specific queries.

                Here are some few-shot examples:

                Input: Please fetch the medical history for patient ID 12345.
                Output: {{"facts" : ["Medical history for patient ID 12345", "Diagnosis: Hypertension, Type 2 Diabetes", "Treatments: Metformin, Lisinopril"], "action": "fetch"}}

                Input: Fetch the latest prescription details for patient ID 67890.
                Output: {{"facts" : ["Prescription details for patient ID 67890", "Medications: 50mg Metformin, 10mg Lisinopril"], "action": "fetch"}}

                Input: How much did patients with hypertension spend on healthcare on average last year?
                Output: {{"facts" : ["Average healthcare expense for patients with hypertension: $4000"], "action": "fetch"}}

                Input: What is the overall patient satisfaction score across all healthcare facilities?
                Output: {{"facts" : ["Overall patient satisfaction score: 85%"], "action": "fetch"}}

                Return the facts and the corresponding action ("store" or "fetch") in a JSON format as shown above.
                """
        })
    
    # Initialize graph
    if "graph_initialized" not in st.session_state:
        st.session_state.graph_initialized = False

# Navigation functions
def navigate_to_agent(agent_id):
    st.session_state.page = "agent"
    st.session_state.current_agent = agent_id
    # Remove any processing flags that might be set
    if f"processing_{agent_id}" in st.session_state:
        del st.session_state[f"processing_{agent_id}"]
    st.rerun()

def navigate_to_home():
    st.session_state.page = "home"
    st.session_state.current_agent = None
    # Clear any processing flags
    for agent_id in agent_info:
        if f"processing_{agent_id}" in st.session_state:
            del st.session_state[f"processing_{agent_id}"]
    st.rerun()

def clear_chat_history(agent_id=None):
    if agent_id:
        st.session_state.chat_history[agent_id] = []
        st.session_state[f"debug_{agent_id}"] = ""
        # Clear evaluation results for the agent
        if f"evaluation_{agent_id}" in st.session_state:
            del st.session_state[f"evaluation_{agent_id}"]
    else:
        for agent_id in agent_info.keys():
            st.session_state.chat_history[agent_id] = []
            st.session_state[f"debug_{agent_id}"] = ""
            # Clear evaluation results for all agents
            if f"evaluation_{agent_id}" in st.session_state:
                del st.session_state[f"evaluation_{agent_id}"]
    st.rerun()

# Home page with agent cards
def show_home_page():
    st.title("MediGraph Consilium")
    st.markdown("#### *An Intelligent Medical Graph Analysis System*")
    st.markdown("### Select an agent to start a conversation")
    
    # Sidebar for home page
    st.sidebar.title("MediGraph Consilium")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **MediGraph Consilium** is an advanced medical graph analysis platform that leverages AI agents to extract insights from healthcare data.
    
    The platform features a **Supervisor Agent** that coordinates between four specialized agents, each with unique capabilities for analyzing different aspects of medical data.
    
    Select an agent card to start a conversation and explore the medical graph.
    """)
    
    # Clear all chats button
    if st.sidebar.button("Clear All Chat Histories"):
        clear_chat_history()
    
    # Add graph cache management section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Graph Cache Management")
    
    # Display when the graph was last loaded
    if st.session_state.graph_initialized:
        from graph_cache import _last_loaded
        if _last_loaded:
            last_loaded_time = datetime.fromtimestamp(_last_loaded).strftime('%Y-%m-%d %H:%M:%S')
            st.sidebar.info(f"Graph last loaded: {last_loaded_time}")
        else:
            st.sidebar.info("Graph cache status: Not initialized")
    
    # Add refresh button
    if st.sidebar.button("Refresh Graph Cache"):
        with st.spinner("Refreshing graph cache..."):
            from graph_cache import clear_cache
            clear_cache()
            db = get_db()
            G = get_graph()
            arango_g = get_arango_graph()
            st.session_state.graph_initialized = True
            st.sidebar.success("Graph cache refreshed successfully!")
            time.sleep(1)  # Give user time to see the success message
            st.rerun()
    
    # Add a supervisor agent section at the top
    st.markdown(f"""
    <div class="agent-card" onclick="none" style="border-left: 5px solid {agent_info['supervisor']['color']}; margin-bottom: 30px;">
        <div class="agent-icon" style="font-size: 50px;">{agent_info['supervisor']['icon']}</div>
        <div class="agent-title" style="font-size: 28px;">{agent_info['supervisor']['name']}</div>
        <div class="agent-description" style="font-size: 18px;">{agent_info['supervisor']['description']}</div>
        <div class="agent-capabilities">
            <strong>Capabilities:</strong><br>
            {"<br>".join([f"• {cap}" for cap in agent_info['supervisor']['capabilities']])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Chat with Supervisor Agent", key="btn_supervisor", type="primary"):
        navigate_to_agent("supervisor")
    
    st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("### Specialized Agents")
    
    # Use a 2x2 grid for better space utilization
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # AQL Agent Card
        with st.container():
            st.markdown(f"""
            <div class="agent-card" onclick="none" style="border-left: 5px solid {agent_info['aql']['color']}">
                <div class="agent-icon">{agent_info['aql']['icon']}</div>
                <div class="agent-title">{agent_info['aql']['name']}</div>
                <div class="agent-description">{agent_info['aql']['description']}</div>
                <div class="agent-capabilities">
                    <strong>Capabilities:</strong><br>
                    {"<br>".join([f"• {cap}" for cap in agent_info['aql']['capabilities']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Chat with AQL Agent", key="btn_aql"):
                navigate_to_agent("aql")
        
        # Patient Data Agent Card
        with st.container():
            st.markdown(f"""
            <div class="agent-card" onclick="none" style="border-left: 5px solid {agent_info['patient']['color']}">
                <div class="agent-icon">{agent_info['patient']['icon']}</div>
                <div class="agent-title">{agent_info['patient']['name']}</div>
                <div class="agent-description">{agent_info['patient']['description']}</div>
                <div class="agent-capabilities">
                    <strong>Capabilities:</strong><br>
                    {"<br>".join([f"• {cap}" for cap in agent_info['patient']['capabilities']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Chat with Patient Data Agent", key="btn_patient"):
                navigate_to_agent("patient")
    
    with col2:
        # Graph Analysis Agent Card
        with st.container():
            st.markdown(f"""
            <div class="agent-card" onclick="none" style="border-left: 5px solid {agent_info['graph']['color']}">
                <div class="agent-icon">{agent_info['graph']['icon']}</div>
                <div class="agent-title">{agent_info['graph']['name']}</div>
                <div class="agent-description">{agent_info['graph']['description']}</div>
                <div class="agent-capabilities">
                    <strong>Capabilities:</strong><br>
                    {"<br>".join([f"• {cap}" for cap in agent_info['graph']['capabilities']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Chat with Graph Analysis Agent", key="btn_graph"):
                navigate_to_agent("graph")
        
        # Population Health Agent Card
        with st.container():
            st.markdown(f"""
            <div class="agent-card" onclick="none" style="border-left: 5px solid {agent_info['population']['color']}">
                <div class="agent-icon">{agent_info['population']['icon']}</div>
                <div class="agent-title">{agent_info['population']['name']}</div>
                <div class="agent-description">{agent_info['population']['description']}</div>
                <div class="agent-capabilities">
                    <strong>Capabilities:</strong><br>
                    {"<br>".join([f"• {cap}" for cap in agent_info['population']['capabilities']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Chat with Population Health Agent", key="btn_population"):
                navigate_to_agent("population")

# Agent chat page
def show_agent_page(agent_state):
    agent_id = st.session_state.current_agent
    agent = agent_info[agent_id]
    
    # Initialize needs_processing state if not exists
    if 'needs_processing' not in st.session_state:
        st.session_state.needs_processing = False
    
    # Agent header
    st.title(f"{agent['icon']} {agent['name']}")
    st.write(agent['description'])
    
    # Sidebar content
    st.sidebar.title(f"{agent['icon']} {agent['name']}")
    
    # Back button in sidebar
    if st.sidebar.button("← Back to Agent Hub", key="back_button"):
        navigate_to_home()
    
    # Capabilities section with expander
    with st.sidebar.expander("### Capabilities", expanded=False):
        for cap in agent['capabilities']:
            st.markdown(f"- {cap}")

    st.sidebar.markdown("### Memory Information")
    memory_data = st.session_state.memory.get_all(
        user_id="User", 
        agent_id="MediGraph Consilium"
    )['results']

    with st.sidebar.expander("View Memory Details", expanded=False):
        # Check if memory_data is empty
        if not memory_data:
            st.markdown("### No Memory Data Available")
            st.markdown("There is no memory data to display at the moment.")
        else:
            # Sidebar dropdown for selecting memory
            memory_ids = [entry['id'] for entry in memory_data]
            selected_memory_id = st.selectbox('Select Memory', memory_ids)

            # Find the selected memory
            selected_memory = next(item for item in memory_data if item['id'] == selected_memory_id)

            # Display the memory and created_at in the sidebar with a custom formatted layout
            st.markdown(f"**Memory ID:** {selected_memory_id}")
            st.markdown(f"**Memory:**\n{selected_memory['memory']}")
            st.markdown(f"**Created At:** {selected_memory['created_at']}")
            st.markdown(f"**User ID:** {selected_memory['user_id']}")
            st.markdown(f"**Agent ID:** {selected_memory['agent_id']}")
    
    # Example Queries Section
    st.sidebar.markdown("### Example Queries")
    st.sidebar.markdown("Select an example query from the dropdown:")
    
    # Create a dropdown for example queries
    selected_query = st.sidebar.selectbox(
        "Choose an example query",
        agent['example_queries'],
        key=f"example_dropdown_{agent_id}",
        label_visibility="collapsed"
    )
    
    # Button to use the selected query
    if st.sidebar.button("Use Selected Query", use_container_width=True):
        if not st.session_state.get(f"processing_{agent_id}"):
            st.session_state.chat_history[agent_id].append({"role": "user", "content": selected_query})
            st.session_state.needs_processing = True
            st.rerun()  # This will trigger a rerun to show the message first
    
    # Clear chat button
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat History"):
        clear_chat_history(agent_id)
    
    # Add graph cache management section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Graph Cache Management")
    
    # Display when the graph was last loaded
    if st.session_state.graph_initialized:
        from graph_cache import _last_loaded
        if _last_loaded:
            last_loaded_time = datetime.fromtimestamp(_last_loaded).strftime('%Y-%m-%d %H:%M:%S')
            st.sidebar.info(f"Graph last loaded: {last_loaded_time}")
        else:
            st.sidebar.info("Graph cache status: Not initialized")
    
    # Add refresh button
    if st.sidebar.button("Refresh Graph Cache", key="refresh_graph_agent"):
        with st.spinner("Refreshing graph cache..."):
            from graph_cache import clear_cache
            clear_cache()
            db = get_db()
            G = get_graph()
            arango_g = get_arango_graph()
            st.session_state.graph_initialized = True
            st.sidebar.success("Graph cache refreshed successfully!")
            time.sleep(1)  # Give user time to see the success message
            st.rerun()
    
    # Create containers first
    chat_container = st.container()
    evaluation_container = st.container()  # New container for evaluation
    steps_container = st.container()
    
    # Get user input before displaying chat history
    user_input = st.chat_input(f"Ask {agent['name']} a question...")
    
    # Handle new user input immediately
    if user_input and not st.session_state.get(f"processing_{agent_id}"):
        # Add user message to chat history and set needs_processing
        st.session_state.chat_history[agent_id].append({"role": "user", "content": user_input})
        st.session_state.needs_processing = True
        st.rerun()
    
    # Display chat history in the chat container
    with chat_container:
        for message in st.session_state.chat_history[agent_id]:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant", avatar=agent['icon']).write(message["content"])
    
    # Display evaluation if available
    with evaluation_container:
        if st.session_state.get(f"evaluation_{agent_id}"):
            evaluation_results = st.session_state[f"evaluation_{agent_id}"]
            st.markdown("### Response Evaluation")
            metrics_cols = st.columns(4)
            with metrics_cols[0]:
                st.metric("Combined Score", f"{evaluation_results['combined_score']:.2f}")
            with metrics_cols[1]:
                if 'average_query_score' in evaluation_results:
                    st.metric("Query Score", f"{evaluation_results['average_query_score']:.2f}")
            with metrics_cols[2]:
                st.metric("Interpretation Score", f"{evaluation_results['average_interpretation_score']:.2f}")
            with metrics_cols[3]:
                quality = "✅ Good" if evaluation_results['final_binary'] == 1 else "❌ Needs Improvement"
                st.metric("Quality", quality)
            
            # Add a visual indicator based on the scores
            if evaluation_results['combined_score'] >= 80:
                st.success("This response meets high-quality standards! 🌟")
            elif evaluation_results['combined_score'] >= 50:
                st.info("This response is acceptable but could be improved. 📝")
            else:
                st.warning("This response needs significant improvement. ⚠️")
    
    # Display debug info
    with steps_container:
        debug_placeholder = st.expander("Agent Thinking Process", expanded=False)
        if not st.session_state.get(f"debug_{agent_id}"):
            debug_placeholder.info("No steps to show yet.")
        else:
            debug_placeholder.code(st.session_state[f"debug_{agent_id}"])
    
    # Process the message if needed
    if st.session_state.needs_processing:
        st.session_state.needs_processing = False  # Reset the flag
        st.session_state[f"debug_{agent_id}"] = "Processing your query...\n"
        st.session_state[f"processing_{agent_id}"] = True
        st.rerun()  # This will trigger the processing
    
    # Process the message if needed
    if (st.session_state.chat_history[agent_id] and 
        st.session_state.chat_history[agent_id][-1]["role"] == "user" and 
        st.session_state.get(f"processing_{agent_id}")):
        
        try:
            with st.spinner(f"{agent['name']} is thinking..."):
                # Create a callback handler to capture intermediate steps
                callback_handler = StreamlitCallbackHandler(agent_id)
                
                # Get the last user message
                user_message = st.session_state.chat_history[agent_id][-1]["content"]
                
                # Handle supervisor agent differently
                if agent_id == "supervisor":
                    # Add to debug info
                    st.session_state[f"debug_{agent_id}"] += "\nUsing supervisor agent to coordinate between specialized agents...\n"
                    
                    # Use async version with timeout for better user experience
                    use_async = True
                    
                    if use_async:
                        # Use async version with timeout
                        st.session_state[f"debug_{agent_id}"] += "\nUsing async supervisor with timeout...\n"
                        
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Define async function to run with progress updates
                        async def run_with_progress():
                            # Generate a unique thread ID
                            thread_id = f"streamlit_{agent_id}_{uuid.uuid4()}"
                            agent_state['messages'].append(user_message)
                            # Start the supervisor agent
                            task = asyncio.create_task(
                                run_supervisor_async(
                                    agent_state, 
                                    st.session_state.memory,
                                    user_message,
                                    timeout=300,
                                    thread_id=thread_id
                                )
                            )
                            
                            # Update progress while waiting
                            start_time = time.time()
                            max_wait = 300  # Same as timeout
                            
                            while not task.done():
                                elapsed = time.time() - start_time
                                progress = min(elapsed / max_wait, 0.95)  # Cap at 95%
                                progress_bar.progress(progress)
                                status_text.text(f"Processing... {int(progress * 100)}%")
                                await asyncio.sleep(0.5)
                            
                            # Get the result
                            result = await task
                            
                            # Complete the progress bar
                            progress_bar.progress(1.0)
                            status_text.text("Complete!")
                            
                            return result
                        
                        # Run the async function
                        full_response = asyncio.run(run_with_progress())
                        
                        # Clean up progress indicators
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        # Use basic synchronous version
                        st.session_state[f"debug_{agent_id}"] += "\nUsing basic supervisor...\n"
                        
                        # Generate a unique thread ID
                        thread_id = f"streamlit_{agent_id}_{uuid.uuid4()}"
                        
                        agent_state['messages'].append(user_message)
                        # Run the supervisor
                        result = run_supervisor(agent_state, st.session_state.memory,user_message, thread_id=thread_id)
                        
                        # Extract the final response from the result
                        if result and "messages" in result and len(result["messages"]) > 0:
                            last_message = result["messages"][-1]
                            # Handle different message types
                            if hasattr(last_message, 'content'):  # AIMessage or similar object
                                full_response = last_message.content
                            elif isinstance(last_message, dict) and 'content' in last_message:
                                full_response = last_message['content']
                            else:
                                full_response = str(last_message)
                        else:
                            full_response = "No response generated"
                        
                        # Add the raw result to debug info
                        st.session_state[f"debug_{agent_id}"] += f"\nRaw result: {str(result)[:500]}...\n"
                else:
                    # Try to use the streaming function first to capture intermediate steps
                    try:
                        # First update the debug info
                        st.session_state[f"debug_{agent_id}"] += "\nAttempting to use streaming function to capture steps...\n"
                        
                        # Try to stream with basic error handling
                        full_response = None
                        chunks = []
                        agent_state['messages'].append(user_message)
                        # Pass the callback handler to the streaming function if it accepts it
                        # Based on the example, the streaming functions can accept a callback
                        for chunk in agent["stream_function"](agent_state, st.session_state.memory, user_message):
                            chunks.append(chunk)
                            
                            # Extract information from the chunk
                            if isinstance(chunk, dict):
                                # Add raw chunk to debug info
                                st.session_state[f"debug_{agent_id}"] += f"\nChunk: {str(chunk)}\n"
                                
                                # Try to extract agent messages
                                if 'agent' in chunk and 'messages' in chunk['agent']:
                                    for msg in chunk['agent']['messages']:
                                        if hasattr(msg, 'content') and msg.content:
                                            st.session_state[f"debug_{agent_id}"] += f"\n🤔 Processing...\n\n Final LLM Response:\n--------------------------------------------------\n{msg.content}\n--------------------------------------------------\n"
                                            full_response = msg.content
                                
                                # Try to extract tool messages
                                if 'tools' in chunk and 'messages' in chunk['tools']:
                                    for msg in chunk['tools']['messages']:
                                        if hasattr(msg, 'content') and msg.content:
                                            st.session_state[f"debug_{agent_id}"] += f"\n📤 Tool output:\n--------------------------------------------------\n{msg.content}\n--------------------------------------------------\n"
                            # Update the debug display
                            with steps_container:
                                debug_placeholder = st.expander("Agent Thinking Process", expanded=True)
                                debug_placeholder.code(st.session_state[f"debug_{agent_id}"])
                        # If we didn't get a response, fall back to the basic function
                        if not full_response:
                            st.session_state[f"debug_{agent_id}"] += "\nNo final response from streaming. Using basic function...\n"
                            
                            # Use the basic function with the callback handler
                            st.session_state[f"debug_{agent_id}"] += "\n🔍 Running basic function...\n"
                            agent_state['messages'].append(user_message)
                            response_tuple = agent["function"](agent_state, st.session_state.memory, user_message)
                            if isinstance(response_tuple, tuple) and len(response_tuple) == 2:
                                full_response, aql_query = response_tuple
                            else:
                                full_response = response_tuple
                    except Exception as e:
                        # Log the error
                        st.session_state[f"debug_{agent_id}"] += f"\nError in streaming: {str(e)}\nFalling back to basic function...\n"
                        
                        # Fall back to the basic function
                        st.session_state[f"debug_{agent_id}"] += "\n🔍 Running basic function...\n"
                        agent_state['messages'].append(user_message)
                        response_tuple = agent["function"](agent_state, st.session_state.memory, user_message)
                        if isinstance(response_tuple, tuple) and len(response_tuple) == 2:
                            full_response, aql_query = response_tuple
                        else:
                            full_response = response_tuple
                
                # Update debug info
                st.session_state[f"debug_{agent_id}"] += "\nAgent response generated successfully."
                
                # Add agent response to chat history
                st.session_state.chat_history[agent_id].append({"role": "agent", "content": full_response})

                # Run evaluation on the existing response
                evaluation_results = evaluate_with_llm_judge(user_message, (full_response, aql_query) if 'aql_query' in locals() else full_response, agent_type=agent_id)

                # Add evaluation results to debug info
                if evaluation_results:
                    st.session_state[f"debug_{agent_id}"] += "\n\n=== Response Evaluation ===\n"
                    st.session_state[f"debug_{agent_id}"] += f"Combined Score: {evaluation_results['combined_score']:.2f}\n"
                    if 'average_query_score' in evaluation_results:
                        st.session_state[f"debug_{agent_id}"] += f"Query Score: {evaluation_results['average_query_score']:.2f}\n"
                    st.session_state[f"debug_{agent_id}"] += f"Interpretation Score: {evaluation_results['average_interpretation_score']:.2f}\n"
                    st.session_state[f"debug_{agent_id}"] += f"Final Binary: {evaluation_results['final_binary']}\n"

                    # Store evaluation results in session state for display in the dedicated container
                    st.session_state[f"evaluation_{agent_id}"] = evaluation_results

        except Exception as e:
            # Handle any errors
            error_message = f"Error: {str(e)}"
            st.error(error_message)
            st.session_state[f"debug_{agent_id}"] += f"\n{error_message}"
            
            # Add error message to chat history
            st.session_state.chat_history[agent_id].append(
                {"role": "agent", "content": f"I encountered an error: {error_message}"}
            )
        
        finally:
            # Reset processing flag
            st.session_state[f"processing_{agent_id}"] = False
            # Force a rerun to display the agent's response immediately
            st.rerun()

# Main app
def main():
    agent_state = AgentState(
    messages=[],
    context={},
    internal_state={},
    callback=None,
    mem0_user_id= "User",
    agent_id="MediGraph Consilium"
    )
    # Apply custom CSS
    local_css()
    
    # Initialize session state
    init_session_state()
    
    # Show appropriate page based on session state
    if st.session_state.page == "home":
        show_home_page()
    elif st.session_state.page == "agent":
        show_agent_page(agent_state)

if __name__ == "__main__":
    main() 