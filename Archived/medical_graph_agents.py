"""
Medical Graph Multi-Agent System for SYNTHEA Healthcare Data Analysis

This module implements a multi-agent architecture for analyzing the SYNTHEA_P100 medical graph
stored in ArangoDB. It provides specialized agents for different analytical tasks such as:
- AQL query generation and execution
- NetworkX algorithm-based graph analysis
- Patient-specific data analysis
- Population health analysis
"""

import os
import json
import time
import re
import networkx as nx
import nx_arangodb as nxadb
import pandas as pd
import matplotlib.pyplot as plt
from typing import Annotated, Dict, TypedDict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel

# LangChain and LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# ArangoDB imports
from arango import ArangoClient

# Configuration
os.environ["OPENAI_API_KEY"] = "sk-svcacct-RovxkVn4B5EldoWTMrmDT3BlbkFJ7G2xSJfecYIFPCt7V5iD"

# Initialize ArangoDB connection
db = ArangoClient(hosts="https://86525f194b77.arangodb.cloud:8529").db(
    '_system', username="root", password="biafgOk988PlafsbNMC3", verify=True
)

# Initialize NetworkX graph with GPU support if available
try:
    
    nx.config.backends.arangodb.use_gpu = True
    print("GPU acceleration enabled for graph operations")
except ImportError:
    nx.config.backends.arangodb.use_gpu = False
    print("GPU acceleration disabled, using CPU for graph operations")

G_adb = nxadb.DiGraph(name="SYNTHEA_P100", db=db)

# Create ArangoDB graph object for graph QA
from langchain_community.graphs import ArangoGraph
arango_graph = ArangoGraph(db)

#############################################################################
# Schema Extraction and Management Functions
#############################################################################

def explore_collections():
    """List all collections in the database"""
    collections = db.collections()
    return [col['name'] for col in collections if not col['name'].startswith('_')]

def explore_collection_schema(collection_name: str, limit: int = 5):
    """Sample documents to understand collection schema"""
    aql = """
    FOR doc IN @@collection
    LIMIT @limit
    RETURN doc
    """
    cursor = db.aql.execute(aql, bind_vars={
        '@collection': collection_name,
        'limit': limit
    })
    return list(cursor)

def get_schema_details():
    """Generate detailed schema information from the database"""
    collections = explore_collections()
    schema_info = {
        "collections": {},
        "sample_docs": {},
        "edge_collections": [],
        "vertex_collections": []
    }

    for collection in collections:
        # Get sample documents
        samples = explore_collection_schema(collection)
        if samples:
            # Store collection properties
            schema_info["collections"][collection] = list(samples[0].keys())
            schema_info["sample_docs"][collection] = samples[0]

            # Identify if collection is edge or vertex
            if '_from' in samples[0] and '_to' in samples[0]:
                schema_info["edge_collections"].append(collection)
            else:
                schema_info["vertex_collections"].append(collection)

    return schema_info

# Get schema details once for reuse
graph_schema = get_schema_details()

#############################################################################
# Tool Definitions
#############################################################################

@tool
def text_to_aql_to_text(query: str):
    """This tool translates natural language queries about the SYNTHEA_P100 graph into AQL,
    executes the query, and returns the results in natural language.
    """
    try:
        # Get actual schema details from database
        schema_details = get_schema_details()

        # Initialize LLM
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4o"
        )

        # Create custom prompts with our schema information
        from langchain.prompts import PromptTemplate

        aql_generation_template = """You are an expert at converting natural language questions into AQL (ArangoDB Query Language) queries.

        GRAPH DETAILS:
        Name: 'SYNTHEA_P100'
        Nodes: 145,514
        Edges: 311,701
        
        DATABASE STRUCTURE:
        Vertex Collections: {vertex_collections}
        Edge Collections: {edge_collections}
        
        COLLECTION SCHEMAS:
        {collection_schemas}

        QUERY GUIDELINES:
        1. Use proper AQL syntax and collection names as specified above
        2. Use FILTER for specific attribute queries
        3. Use LIMIT 10 for result size control
        4. Use proper attribute names as shown in the sample documents
        5. Include only relevant attributes in the RETURN statement

        Human: Convert this question into an AQL query: {user_input}
        Assistant: Based on the database structure, here's the AQL query to answer your question:

        ```aql
        """

        aql_qa_template = """You are an expert at interpreting AQL query results in the context of medical data.

        DATABASE STRUCTURE:
        Vertex Collections: {vertex_collections}
        Edge Collections: {edge_collections}

        Original question: {user_input}
        AQL query used: {aql_query}
        Query result: {aql_result}

        Please provide a clear, natural language response to the original question based on the query results.
        Human: What does this result mean?
        Assistant: """

        # Initialize chain with default prompts
        from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
        
        chain = ArangoGraphQAChain.from_llm(
            llm=llm,
            graph=arango_graph,
            verbose=True,
            allow_dangerous_requests=True,
            top_k=10,
            return_aql_query=True,
            return_aql_result=True,
            max_aql_generation_attempts=3
        )

        # Add comprehensive examples with correct AQL syntax
        chain.aql_examples = """
        # Count total records in all collections
        RETURN {
            vertex_collections: {
                patients: LENGTH(patients),
                encounters: LENGTH(encounters),
                conditions: LENGTH(conditions),
                medications: LENGTH(medications),
                allergies: LENGTH(allergies),
                immunizations: LENGTH(immunizations),
                observations: LENGTH(observations),
                procedures: LENGTH(procedures),
                devices: LENGTH(devices),
                imaging_studies: LENGTH(imaging_studies),
                careplans: LENGTH(careplans),
                supplies: LENGTH(supplies),
                organizations: LENGTH(organizations),
                providers: LENGTH(providers),
                payers: LENGTH(payers)
            },
            edge_collections: {
                patients_to_medications: LENGTH(patients_to_medications),
                encounters_to_immunizations: LENGTH(encounters_to_immunizations),
                patients_to_conditions: LENGTH(patients_to_conditions),
                patients_to_allergies: LENGTH(patients_to_allergies),
                encounters_to_medications: LENGTH(encounters_to_medications)
            }
        }

        # Find patients with specific condition
        FOR patient IN patients
            FOR condition IN OUTBOUND patient patients_to_conditions
                FILTER condition.description == @condition_name
                RETURN {
                    patient_id: patient._key,
                    condition: condition.description,
                    onset: condition.start
                }

        # Count encounters by provider
        FOR encounter IN encounters
            FOR provider IN INBOUND encounter providers_to_encounters
                COLLECT provider_id = provider._key
                WITH COUNT INTO count
                RETURN {
                    provider: provider_id,
                    encounter_count: count
                }

        # Find allergies with patient details
        FOR allergy IN allergies
            FOR patient IN INBOUND allergy patients_to_allergies
                RETURN {
                    patient_id: patient._key,
                    allergy: allergy.description,
                    severity: allergy.severity
                }

        # Complex query: Patient history with conditions and medications
        FOR patient IN patients
            LET conditions = (
                FOR condition IN OUTBOUND patient patients_to_conditions
                    RETURN {
                        description: condition.description,
                        onset: condition.start
                    }
            )
            LET medications = (
                FOR medication IN OUTBOUND patient patients_to_medications
                    RETURN {
                        drug: medication.description,
                        start: medication.start
                    }
            )
            RETURN {
                patient_id: patient._key,
                conditions: conditions,
                medications: medications
            }
        """

        # Execute query
        result = chain.invoke({
            "query": query
        })

        response = f"""
          Answer: {result['result']}

          Generated AQL Query:
          ```aql
          {result['aql_query']}
          ```

          Query Results:
          ```json
          {json.dumps(result['aql_result'], indent=2)}
          ```
          """
        return response

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}\nPlease try rephrasing your question."
        print(f"Error details: {str(e)}")  # For debugging
        return error_msg

@tool
def text_to_nx_algorithm_to_text(query: str):
    """This tool is available to invoke a NetworkX Algorithm on
    the ArangoDB Graph. You are responsible for accepting the
    Natural Language Query, establishing which algorithm needs to
    be executed, executing the algorithm, and translating the results back
    to Natural Language, with respect to the original query.
    """
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o")

    ######################
    print("1) Generating NetworkX code")

    text_to_nx = llm.invoke(f"""
    I have a NetworkX Graph called `G_adb`. It has the following schema: {json.dumps(graph_schema, indent=2)}

    I have the following graph analysis query: {query}.
    Instructions : 

          Follow this structured reasoning process to generate optimal graph analysis code:

          1. Graph Understanding Phase:
            - Schema Analysis: First analyze the provided schema
            - Query Decomposition: Break down the query into atomic operations
            - Identify relevant node types and relationships
            - Document your reasoning with inline comments

          2. Algorithm Selection Phase:
            Let's think step by step:
            a) What is the core graph operation needed? (path finding/centrality/community/etc.)
            b) Which NetworkX algorithms match these requirements?
            c) What is the computational complexity of each option?
            d) Choose the most efficient algorithm and justify in comments

          3. Optimization Planning Phase:
            Reason through these optimization steps:
            a) Can we reduce the initial search space?
            b) Where can we apply early filtering?
            c) What intermediate results can we cache?
            d) How can we minimize memory usage?

          4. Implementation Strategy:
            Walk through the solution:
            a) Start with the smallest possible subgraph
            b) Apply filters before expensive operations
            c) Use generators and iterators for large collections
            d) Implement early termination conditions

          5. Result Validation Phase:
            Consider these aspects:
            a) Will the result scale with graph size?
            b) Is the output format optimized?
            c) Are edge cases handled?
            d) Is error handling comprehensive?

          Generate Python code that:
          - Follows this reasoning process in comments
          - Uses only NetworkX and standard Python libraries
          - Stores the final result in `FINAL_RESULT`
          - Implements all optimizations identified above
          - Avoids full graph traversals
          - Handles errors gracefully

    MUST FOLLOW :
        1. Must include reasoning as inline comments
        2. Must use efficient NetworkX algorithms
        3. Must avoid full graph traversals
        4. Must filter data early
        5. Must set FINAL_RESULT as output
        6. Must include only standard Python and NetworkX imports
        7. Must be directly executable via exec()
        8. ONLY PROVIDE THE CODE BLOCK WITH NO OTHER TEXT:

    Generate the Python Code required to answer the query using the `G_adb` object.
    Be very precise on the NetworkX algorithm you select to answer this query. Think step by step.
    Only assume that networkx is installed, and other base python dependencies.
    - Use efficient filtering for specific node types
    do filtering properly and as graph is very large so for loop over entire garaph will be very time consuming and inefficient.


    Always set the last variable as `FINAL_RESULT`, which represents the answer to the original query.

    Only provide python code that I can directly execute via `exec()`. Do not provide any instructions.

    Make sure that `FINAL_RESULT` stores a short & consice answer. Avoid setting this variable to a long sequence.

    Your code:
    """).content

    # Clean up the generated code
    text_to_nx_cleaned = re.sub(r"^```python\n|```$", "", text_to_nx, flags=re.MULTILINE).strip()

    print('-'*10)
    print(text_to_nx_cleaned)
    print('-'*10)

    ######################
    print("\n2) Executing NetworkX code")
    global_vars = {"G_adb": G_adb, "nx": nx}
    local_vars = {}

    try:
        exec(text_to_nx_cleaned, global_vars, local_vars)
        FINAL_RESULT = local_vars.get("FINAL_RESULT", "No result generated")
    except Exception as e:
        FINAL_RESULT = f"Error executing NetworkX code: {str(e)}"

    print('-'*10)
    print(f"FINAL_RESULT: {FINAL_RESULT}")
    print('-'*10)

    ######################
    print("3) Formulating final answer")

    nx_to_text = llm.invoke(f"""
        I have a NetworkX Graph called `G_adb`. It has the following schema: {json.dumps(graph_schema, indent=2)}

        I have the following graph analysis query: {query}.

        I have executed the following python code to help me answer my query:

        ---
        {text_to_nx_cleaned}
        ---

        The `FINAL_RESULT` variable is set to the following: {FINAL_RESULT}.

        Based on my original Query and FINAL_RESULT, generate a short and concise response to
        answer my query.

        Your response:
    """).content

    return nx_to_text

# Step 3: Define base models for queries
class PatientQuery(BaseModel):
    patient_id: str

class ConditionQuery(BaseModel):
    search_term: str = None
    code: str = None

@tool
def query_patient_history(query: PatientQuery) -> str:
    """Query patient's medical history"""
    try:
        aql = """
        LET patient = DOCUMENT('patients', @patient_id)
        
        LET conditions = (
            FOR c IN conditions
            FILTER c.PATIENT == @patient_id
            RETURN {
                type: 'condition',
                date: c.START,
                description: c.DESCRIPTION,
                code: c.CODE
            }
        )
        
        LET medications = (
            FOR m IN medications
            FILTER m.PATIENT == @patient_id
            RETURN {
                type: 'medication',
                start: m.START,
                stop: m.STOP,
                description: m.DESCRIPTION,
                code: m.CODE,
                cost: m.BASE_COST
            }
        )
        
        LET encounters = (
            FOR e IN encounters
            FILTER e.PATIENT == @patient_id
            SORT e.START DESC
            RETURN {
                date: e.START,
                type: e.ENCOUNTERCLASS,
                description: e.DESCRIPTION,
                cost: e.BASE_ENCOUNTER_COST
            }
        )
        
        RETURN {
            patient: patient,
            conditions: conditions,
            medications: medications,
            encounters: encounters
        }
        """
        
        cursor = db.aql.execute(
            aql,
            bind_vars={'patient_id': query.patient_id}
        )
        return json.dumps(list(cursor)[0])
    except Exception as e:
        return f"Error querying patient history: {str(e)}"

@tool
def search_conditions(query: ConditionQuery) -> str:
    """Search conditions and their frequencies"""
    try:
        aql = """
        FOR c IN conditions
            FILTER @search_term == null OR 
                   LOWER(c.DESCRIPTION) LIKE CONCAT('%', LOWER(@search_term), '%')
            FILTER @code == null OR c.CODE == @code
            
            COLLECT description = c.DESCRIPTION, code = c.CODE
            WITH COUNT INTO frequency
            
            SORT frequency DESC
            LIMIT 10
            
            RETURN {
                description: description,
                code: code,
                frequency: frequency
            }
        """
        
        cursor = db.aql.execute(
            aql,
            bind_vars={
                'search_term': query.search_term,
                'code': query.code
            }
        )
        return json.dumps(list(cursor))
    except Exception as e:
        return f"Error searching conditions: {str(e)}"

@tool
def analyze_medications(condition_desc: str) -> str:
    """Analyze medications prescribed for a condition"""
    try:
        aql = """
        LET condition_encounters = (
            FOR c IN conditions
            FILTER LOWER(c.DESCRIPTION) LIKE CONCAT('%', LOWER(@condition), '%')
            RETURN c.ENCOUNTER
        )
        
        FOR m IN medications
        FILTER m.ENCOUNTER IN condition_encounters
        
        COLLECT description = m.DESCRIPTION, code = m.CODE
        WITH COUNT INTO frequency
        
        SORT frequency DESC
        LIMIT 10
        
        RETURN {
            medication: description,
            code: code,
            frequency: frequency,
            avg_cost: AVERAGE(
                FOR med IN medications
                FILTER med.CODE == code
                RETURN med.BASE_COST
            )
        }
        """
        
        cursor = db.aql.execute(
            aql,
            bind_vars={'condition': condition_desc}
        )
        return json.dumps(list(cursor))
    except Exception as e:
        return f"Error analyzing medications: {str(e)}"


@tool
def find_similar_patients(patient_id: str, max_distance: int = 2) -> str:
    """Find patients with similar medical profiles using graph traversal"""
    try:
        aql = """
        // First get conditions for our target patient
        LET patient_conditions = (
            FOR c IN conditions
            FILTER c.PATIENT == @patient_id
            COLLECT condition_code = c.CODE
            RETURN condition_code
        )
        
        // Find patients with similar conditions
        LET similar_patients = (
            FOR c IN conditions
            FILTER c.CODE IN patient_conditions AND c.PATIENT != @patient_id
            COLLECT patient = c.PATIENT, 
                    matches = COUNT(c.CODE)
            SORT matches DESC
            LIMIT 10
            RETURN {
                patient_id: patient,
                condition_matches: matches,
                similarity_score: matches / LENGTH(patient_conditions)
            }
        )
        
        RETURN {
            target_patient: @patient_id,
            condition_count: LENGTH(patient_conditions),
            similar_patients: similar_patients
        }
        """
        
        cursor = db.aql.execute(
            aql,
            bind_vars={'patient_id': patient_id}
        )
        return json.dumps(list(cursor)[0])
    except Exception as e:
        return f"Error finding similar patients: {str(e)}"

@tool
def analyze_treatment_pathways(condition_code: str) -> str:
    """Analyze common treatment pathways for a specific condition"""
    try:
        aql = """
        // First identify patients with this condition
        LET condition_patients = (
            FOR c IN conditions
            FILTER c.CODE == @condition_code
            RETURN DISTINCT c.PATIENT
        )
        
        // For those patients, find the sequence of medications
        LET medication_pathways = (
            FOR patient_id IN condition_patients
                LET patient_meds = (
                    FOR m IN medications
                    FILTER m.PATIENT == patient_id
                    SORT m.START
                    RETURN {
                        medication: m.DESCRIPTION,
                        code: m.CODE,
                        start: m.START
                    }
                )
                
                RETURN {
                    patient: patient_id,
                    medication_sequence: patient_meds
                }
        )
        
        // Analyze common medication patterns
        LET common_first_medications = (
            FOR pathway IN medication_pathways
                FILTER LENGTH(pathway.medication_sequence) > 0
                COLLECT medication = pathway.medication_sequence[0].medication
                WITH COUNT INTO frequency
                SORT frequency DESC
                LIMIT 5
                RETURN {
                    medication: medication,
                    frequency: frequency,
                    percentage: ROUND(frequency * 100 / LENGTH(medication_pathways), 2)
                }
        )
        
        RETURN {
            condition_code: @condition_code,
            patient_count: LENGTH(condition_patients),
            common_first_medications: common_first_medications,
            sample_pathways: (
                FOR p IN medication_pathways
                LIMIT 3
                RETURN p
            )
        }
        """
        
        cursor = db.aql.execute(
            aql,
            bind_vars={'condition_code': condition_code}
        )
        return json.dumps(list(cursor)[0])
    except Exception as e:
        return f"Error analyzing treatment pathways: {str(e)}"

#############################################################################
# Callback Handler for Agent Execution
#############################################################################

class CustomConsoleCallbackHandler(BaseCallbackHandler):

    def __init__(self):
        """Initialize the handler"""
        self.current_agent_name = None
        super().__init__()

    def write_agent_name(self, name: str):
        """Display agent name"""
        self.current_agent_name = name
        print(f"\n=== Agent: {name} ===")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Display tool execution start"""
        print(f"\n🔧 Using tool: {serialized['name']}")

    def on_tool_end(self, output: str, **kwargs):
        """Display tool execution result"""
        print("\n📤 Tool output:")
        print("-" * 50)
        print(output)
        print("-" * 50)

    def on_agent_action(self, action: Any, **kwargs):
        """Display agent action"""
        if hasattr(action, 'tool'):
            print(f"\n🎯 Action: {action.tool}")
            print("Input:")
            print("-" * 50)
            print(action.tool_input)
            print("-" * 50)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: list[str], **kwargs):
        """Display when LLM starts processing"""
        print("\n🤔 Processing...")

    def on_llm_end(self, response, **kwargs):
        """Display final LLM response"""
        if hasattr(response, 'generations') and response.generations:
            print("\n Final LLM Response:")
            print("-" * 50)
            print(response.generations[0][0].text)
            print("-" * 50)

    def on_tool_error(self, error: str, **kwargs):
        """Display tool errors"""
        print(f"\n❌ Error: {error}")

#############################################################################
# Agent State Type
#############################################################################

class AgentState(TypedDict):
    messages: list
    current_date: str
    context: Dict
    internal_state: Dict
    callback: Any

#############################################################################
# Specialized Agent Definitions
#############################################################################

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str, max_iterations: int = 2, max_execution_time: int = 120):
    """
    Creates a LangGraph react agent using the specified ChatOpenAI model, tools, and system prompt.
    
    Args:
        llm: LLM to be used to create the agent
        tools: List of tools to be given to the agent
        system_prompt: System prompt to be used in the agent
        max_iterations: Maximum number of iterations (will be converted to recursion_limit)
        max_execution_time: Maximum execution time in seconds
    """
    def _modify_state_messages(state: AgentState):
        # Add system prompt and keep existing messages
        return [("system", system_prompt)] + state["messages"]
    
    # Create the react agent
    agent = create_react_agent(llm, tools, prompt=_modify_state_messages)
    
    # Set recursion limit (LangGraph uses 2 steps per iteration + 1)
    agent.recursion_limit = 2 * max_iterations + 1
    
    # Set timeout for each step
    agent.step_timeout = max_execution_time
    
    return agent

def create_llm():
    """Create a standard LLM for all agents"""
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.1
    )

#############################################################################
# 1. AQL Query Agent - For natural language to AQL queries
#############################################################################

def aql_query_agent(state: AgentState):
    """
    Agent that specializes in converting natural language to AQL queries and executing them.
    Focused on complex database queries and relationship exploration.
    """
    try:
        system_prompt = f"""You are an expert medical data analyst specializing in ArangoDB graph queries.
        
        Your role is to:
        1. Convert natural language medical questions into precise AQL queries
        2. Execute these queries against the SYNTHEA_P100 database
        3. Interpret the results in a medically meaningful way
        4. Provide clear, structured responses that medical professionals can use
        
        Database schema:
        {json.dumps(graph_schema, indent=2)}
        
        You should consider:
        - Temporal relationships between medical events
        - Patient-provider-organization relationships
        - Treatment patterns and pathways
        - Cost analysis and optimization
        
        Always use proper medical terminology and provide context for your findings.
        """

        # Tools specific to this agent
        agent_tools = [text_to_aql_to_text]
        
        # Initialize LLM
        llm = create_llm()
        
        # Create the agent
        agent = create_agent(
            llm, 
            agent_tools, 
            system_prompt,
            max_iterations=3,
            max_execution_time=180
        )
        
        # Set agent name in callback
        state["callback"].write_agent_name("AQL Query Expert 📊")
        
        # Stream mode will give us intermediate steps
        chunks = []
        for chunk in agent.stream(
            {"messages": state["messages"]},
            {"callbacks": [state["callback"]]},
            stream_mode="updates"  # This gives us intermediate steps
        ):
            chunks.append(chunk)
            
        # Get final state from chunks
        final_messages = chunks[-1].get("messages", []) if chunks else []
        
        # Debug: Print the final messages
        print(f"AQL Query Agent received {len(final_messages)} messages from agent")
        for i, msg in enumerate(final_messages):
            print(f"Message {i}: type={type(msg).__name__}")
            if hasattr(msg, 'content'):
                print(f"  Content preview: {msg.content[:50]}...")
        
        # Update state with agent's response
        if final_messages:
            state["messages"] = final_messages
        else:
            # If no messages were returned, check the callback for the final response
            if hasattr(state["callback"], "intermediate_steps") and state["callback"].intermediate_steps:
                # Look for the final LLM response in the intermediate steps
                final_response = None
                for step in reversed(state["callback"].intermediate_steps):
                    if isinstance(step, str) and "Final LLM Response:" in step:
                        # Extract the response content
                        response_parts = step.split("-" * 50)
                        if len(response_parts) >= 2:
                            final_response = response_parts[1].strip()
                            break
                
                if final_response:
                    print(f"Extracted final response from intermediate steps: {final_response[:100]}...")
                    # Add the response as an AI message
                    state["messages"].append(AIMessage(content=final_response))
                else:
                    # Fallback: create a generic response
                    state["messages"].append(
                        AIMessage(content="I've analyzed the database but couldn't generate a specific response. Please try a more specific query.")
                    )
        
        # Store internal state
        state["internal_state"]["aql_agent_executed"] = True
        
        # Make sure we have a last response to store
        if state["messages"] and any(isinstance(msg, AIMessage) for msg in state["messages"]):
            # Find the last AI message
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    state["internal_state"]["last_aql_agent_response"] = msg.content
                    break
        else:
            state["internal_state"]["last_aql_agent_response"] = "No response generated"
        
    except Exception as e:
        # Handle errors
        error_msg = f"AQL Query Agent encountered an error: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        state["messages"].append(
            AIMessage(content=error_msg)
        )
    
    # Debug: Print the final state messages
    print(f"AQL Query Agent returning {len(state['messages'])} messages")
    for i, msg in enumerate(state["messages"]):
        print(f"Message {i}: type={type(msg).__name__}")
        if hasattr(msg, 'content'):
            print(f"  Content preview: {msg.content[:50]}...")
    
    return state

#############################################################################
# 2. Graph Analysis Agent - For NetworkX algorithm-based analysis
#############################################################################

def graph_analysis_agent(state: AgentState):
    """
    Agent that specializes in complex graph analysis using NetworkX algorithms.
    Focused on network metrics, pathfinding, and structural analysis.
    """
    try:
        system_prompt = f"""You are an expert graph analyst specializing in medical network analysis.
        
        Your role is to:
        1. Apply advanced graph algorithms to medical networks
        2. Identify key patterns, hubs, and influences in the medical graph
        3. Discover hidden relationships and pathways
        4. Extract actionable insights from complex network structures
        
        Database schema:
        {json.dumps(graph_schema, indent=2)}
        
        You should analyze:
        - Centrality and influence of providers and organizations
        - Community detection to find related medical entities
        - Path analysis for treatment flows
        - Structural patterns that reveal medical practice behaviors
        
        Always focus on the graph-theoretical implications and what they mean for healthcare.
        """

        # Tools specific to this agent
        agent_tools = [text_to_nx_algorithm_to_text]
        
        # Initialize LLM
        llm = create_llm()
        
        # Create the agent
        agent = create_agent(
            llm, 
            agent_tools, 
            system_prompt,
            max_iterations=3,
            max_execution_time=180
        )
        
        # Set agent name in callback
        state["callback"].write_agent_name("Graph Analysis Expert 🕸️")
        
        # Stream mode will give us intermediate steps
        chunks = []
        for chunk in agent.stream(
            {"messages": state["messages"]},
            {"callbacks": [state["callback"]]},
            stream_mode="updates"  # This gives us intermediate steps
        ):
            chunks.append(chunk)
            
        # Get final state from chunks
        final_messages = chunks[-1].get("messages", []) if chunks else []
        
        # Update state with agent's response
        state["messages"] = final_messages
        
        # Store internal state
        state["internal_state"]["graph_agent_executed"] = True
        state["internal_state"]["last_graph_agent_response"] = final_messages[-1].content if final_messages else ""
        
    except Exception as e:
        # Handle errors
        state["messages"].append(
            AIMessage(content=f"Graph Analysis Agent encountered an error: {str(e)}")
        )
    
    return state

#############################################################################
# 3. Patient Data Agent - For patient-specific analysis
#############################################################################

def patient_data_agent(state: AgentState):
    """
    Agent that specializes in analyzing individual patient data.
    Focused on comprehensive patient history, treatment timelines, and personalized insights.
    """
    try:
        system_prompt = f"""You are a clinical data specialist focused on individual patient analysis.
        
        Your role is to:
        1. Analyze comprehensive patient medical histories
        2. Identify patterns in individual patient care
        3. Evaluate treatment effectiveness for specific patients
        4. Find similar patients for comparative analysis
        
        Database schema:
        {json.dumps(graph_schema, indent=2)}
        
        You should consider:
        - Longitudinal analysis of patient conditions and treatments
        - Treatment response patterns
        - Care coordination across providers
        - Cost-effectiveness of personalized treatments
        
        Always maintain clinical relevance and prioritize insights that could improve patient care.
        """

        # Tools specific to this agent
        agent_tools = [
            query_patient_history,
            find_similar_patients
        ]
        
        # Initialize LLM
        llm = create_llm()
        
        # Create the agent
        agent = create_agent(
            llm, 
            agent_tools, 
            system_prompt,
            max_iterations=3,
            max_execution_time=180
        )
        
        # Set agent name in callback
        state["callback"].write_agent_name("Patient Analysis Expert 👨‍⚕️")
        
        # Stream mode will give us intermediate steps
        chunks = []
        for chunk in agent.stream(
            {"messages": state["messages"]},
            {"callbacks": [state["callback"]]},
            stream_mode="updates"  # This gives us intermediate steps
        ):
            chunks.append(chunk)
            
        # Get final state from chunks
        final_messages = chunks[-1].get("messages", []) if chunks else []
        
        # Debug: Print the final messages
        print(f"Patient Data Agent received {len(final_messages)} messages from agent")
        for i, msg in enumerate(final_messages):
            print(f"Message {i}: type={type(msg).__name__}")
            if hasattr(msg, 'content'):
                print(f"  Content preview: {msg.content[:50]}...")
        
        # Update state with agent's response
        if final_messages:
            state["messages"] = final_messages
        else:
            # If no messages were returned, check the callback for the final response
            if hasattr(state["callback"], "intermediate_steps") and state["callback"].intermediate_steps:
                # Look for the final LLM response in the intermediate steps
                final_response = None
                for step in reversed(state["callback"].intermediate_steps):
                    if isinstance(step, str) and "Final LLM Response:" in step:
                        # Extract the response content
                        response_parts = step.split("-" * 50)
                        if len(response_parts) >= 2:
                            final_response = response_parts[1].strip()
                            break
                
                if final_response:
                    print(f"Extracted final response from intermediate steps: {final_response[:100]}...")
                    # Add the response as an AI message
                    state["messages"].append(AIMessage(content=final_response))
                else:
                    # Fallback: create a generic response
                    state["messages"].append(
                        AIMessage(content="I've analyzed the patient data but couldn't generate a specific response. Please try a more specific query.")
                    )
        
        # Store internal state
        state["internal_state"]["patient_agent_executed"] = True
        
        # Make sure we have a last response to store
        if state["messages"] and any(isinstance(msg, AIMessage) for msg in state["messages"]):
            # Find the last AI message
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    state["internal_state"]["last_patient_agent_response"] = msg.content
                    break
        else:
            state["internal_state"]["last_patient_agent_response"] = "No response generated"
        
    except Exception as e:
        # Handle errors
        error_msg = f"Patient Data Agent encountered an error: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        state["messages"].append(
            AIMessage(content=error_msg)
        )
    
    # Debug: Print the final state messages
    print(f"Patient Data Agent returning {len(state['messages'])} messages")
    for i, msg in enumerate(state["messages"]):
        print(f"Message {i}: type={type(msg).__name__}")
        if hasattr(msg, 'content'):
            print(f"  Content preview: {msg.content[:50]}...")
    
    return state

#############################################################################
# 4. Population Health Agent - For population-level analysis
#############################################################################

def population_health_agent(state: AgentState):
    """
    Agent that specializes in population-level health analysis.
    Focused on trends, patterns, and insights across the entire patient population.
    """
    try:
        system_prompt = f"""You are a population health analyst specializing in medical data trends.
        
        Your role is to:
        1. Identify patterns and trends across large patient populations
        2. Analyze condition prevalence and distribution
        3. Evaluate treatment effectiveness at scale
        4. Discover correlations between medical factors
        
        Database schema:
        {json.dumps(graph_schema, indent=2)}
        
        You should analyze:
        - Disease prevalence and distribution
        - Treatment pathways and outcomes
        - Provider practice patterns
        - Cost and resource utilization trends
        
        Always consider statistical significance and focus on actionable population health insights.
        """

        # Tools specific to this agent
        agent_tools = [
            search_conditions,
            analyze_medications,
            analyze_treatment_pathways
        ]
        
        # Initialize LLM
        llm = create_llm()
        
        # Create the agent
        agent = create_agent(
            llm, 
            agent_tools, 
            system_prompt,
            max_iterations=3,
            max_execution_time=180
        )
        
        # Set agent name in callback
        state["callback"].write_agent_name("Population Health Expert 📈")
        
        # Stream mode will give us intermediate steps
        chunks = []
        for chunk in agent.stream(
            {"messages": state["messages"]},
            {"callbacks": [state["callback"]]},
            stream_mode="updates"  # This gives us intermediate steps
        ):
            chunks.append(chunk)
            
        # Get final state from chunks
        final_messages = chunks[-1].get("messages", []) if chunks else []
        
        # Update state with agent's response
        state["messages"] = final_messages
        
        # Store internal state
        state["internal_state"]["population_agent_executed"] = True
        state["internal_state"]["last_population_agent_response"] = final_messages[-1].content if final_messages else ""
        
    except Exception as e:
        # Handle errors
        state["messages"].append(
            AIMessage(content=f"Population Health Agent encountered an error: {str(e)}")
        )
    
    return state

#############################################################################
# Utility Functions for Testing
#############################################################################

def run_agent_test(agent_func, query: str):
    """Test a specific agent with a query"""
    # Create callback handler
    callback = CustomConsoleCallbackHandler()
    
    # Create initial state
    state = AgentState(
        messages=[HumanMessage(content=query)],
        current_date=datetime.now().isoformat(),
        context={},
        internal_state={},
        callback=callback
    )
    
    # Run the agent
    result_state = agent_func(state)
    
    # Get the response
    if result_state["messages"]:
        response = result_state["messages"][-1].content
    else:
        response = "No response generated"
    
    return response

def test_all_agents():
    """Run tests for all agents with appropriate queries"""
    test_queries = {
        "aql_query_agent": [
            "Find all patients who have been diagnosed with hypertension and are on beta blockers",
            "What are the most common medications prescribed for diabetes patients?"
        ],
        "graph_analysis_agent": [
            "Which providers have the highest centrality in the referral network?",
            "Find communities of providers who frequently collaborate on patient care"
        ],
        "patient_data_agent": [
            "Give me a complete medical history for patient with ID 'f4640c72-6ea6-db89-e996-91c90af95544'",
            "Find patients similar to patient 'f4640c72-6ea6-db89-e996-91c90af95544' based on their conditions"
        ],
        "population_health_agent": [
            "What are the most common conditions in the database and their frequencies?",
            "Analyze the typical treatment pathway for patients with COPD"
        ]
    }
    
    results = {}
    
    # Test AQL Query Agent
    print("\n" + "="*80)
    print("TESTING AQL QUERY AGENT")
    print("="*80)
    for query in test_queries["aql_query_agent"]:
        print(f"\nQuery: {query}")
        result = run_agent_test(aql_query_agent, query)
        results[query] = result
    
    # Test Graph Analysis Agent
    print("\n" + "="*80)
    print("TESTING GRAPH ANALYSIS AGENT")
    print("="*80)
    for query in test_queries["graph_analysis_agent"]:
        print(f"\nQuery: {query}")
        result = run_agent_test(graph_analysis_agent, query)
        results[query] = result
    
    # Test Patient Data Agent
    print("\n" + "="*80)
    print("TESTING PATIENT DATA AGENT")
    print("="*80)
    for query in test_queries["patient_data_agent"]:
        print(f"\nQuery: {query}")
        result = run_agent_test(patient_data_agent, query)
        results[query] = result
    
    # Test Population Health Agent
    print("\n" + "="*80)
    print("TESTING POPULATION HEALTH AGENT")
    print("="*80)
    for query in test_queries["population_health_agent"]:
        print(f"\nQuery: {query}")
        result = run_agent_test(population_health_agent, query)
        results[query] = result
    
    return results

# This allows running the test when the file is executed directly
if __name__ == "__main__":
    print("Testing Medical Graph Multi-Agent System...")
    test_results = test_all_agents()
    print("\nAll tests completed!") 