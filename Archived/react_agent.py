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



from tools import *
from callback import *


# Tools for each specialized agent
AQL_QUERY_TOOLS = [text_to_aql_to_text]
GRAPH_ANALYSIS_TOOLS = [text_to_nx_algorithm_to_text]
PATIENT_DATA_TOOLS = [query_patient_history, search_conditions]
POPULATION_HEALTH_TOOLS = [analyze_medications, analyze_treatment_pathways]

#############################################################################
# Agent State Type
#############################################################################

class AgentState(TypedDict):
    messages: list  
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

def aql_query_node(state: AgentState):
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
        aql_query_tools = [text_to_aql_to_text]
        
        # Initialize LLM
        llm = create_llm()
        
        # Create the agent
        aql_query_agent = create_agent(
            llm, 
            aql_query_tools, 
            system_prompt,
            max_iterations=3,
            max_execution_time=180
        )
        
        state["callback"].write_agent_name("AQL Query Agent 📊")
        
        # Convert messages to the correct format if they're tuples
        formatted_messages = []
        for msg in state["messages"]:
            if isinstance(msg, tuple):
                role, content = msg
                if role == "human":
                    formatted_messages.append(HumanMessage(content=content))
                elif role == "system":
                    formatted_messages.append(SystemMessage(content=content))
                elif role == "ai":
                    formatted_messages.append(AIMessage(content=content))
            else:
                formatted_messages.append(msg)
                
        # Update state with formatted messages
        state["messages"] = formatted_messages
        
        # Stream mode will give us intermediate steps
        chunks = []
        for chunk in aql_query_agent.stream(
            {"messages": state["messages"]},
            {"callbacks": [state["callback"]]},
            stream_mode="updates"  # This gives us intermediate steps
        ):
            chunks.append(chunk)
            
        # Get final state from chunks
        final_messages = chunks[-1].get("messages", []) if chunks else []
        
        # Get the response from the callback
        ai_response = state["callback"].get_final_response()
        
        if ai_response:
            # If we have a response from the callback, use it
            state["messages"].append(AIMessage(content=ai_response))
        elif final_messages:
            # If we have final messages, use those
            state["messages"] = final_messages
        else:
            # Try to extract the last AI message from chunks
            for chunk in reversed(chunks):
                if "agent" in chunk and "messages" in chunk["agent"]:
                    for msg in reversed(chunk["agent"]["messages"]):
                        if isinstance(msg, AIMessage):
                            state["messages"].append(msg)
                            break
                    if len(state["messages"]) > len(formatted_messages):
                        break
            
            # If still no response, create a fallback
            if len(state["messages"]) == len(formatted_messages):
                state["messages"].append(
                    AIMessage(content="I processed your query but couldn't generate a complete response.")
                )
        
        # Store internal state
        if "aql_query_agent_internal_state" not in state:
            state["aql_query_agent_internal_state"] = {}
            
        state["aql_query_agent_internal_state"]["agent_executor_tools"] = {
            tool.name: 0 for tool in aql_query_tools
        }
        state["aql_query_agent_internal_state"]["full_response"] = {
            "messages": state["messages"],
            "intermediate_steps": chunks[:-1] if chunks else []  # All but last chunk are intermediate steps
        }
        
    except TimeoutError:
        # Handle timeout
        state["messages"].append(
            AIMessage(content="Agent stopped due to timeout.")
        )
    except Exception as e:
        # Handle other errors
        state["messages"].append(
            AIMessage(content=f"Agent encountered an error: {str(e)}")
        )
    
    return state


# #############################################################################
# #############################################################################
# # 2. Graph Analysis Agent - For NetworkX algorithm-based analysis
# #############################################################################

# def graph_analysis_agent(state: AgentState):
#     """
#     Agent that specializes in complex graph analysis using NetworkX algorithms.
#     Focused on network metrics, pathfinding, and structural analysis.
#     """
#     try:
#         system_prompt = f"""You are an expert graph analyst specializing in medical network analysis.
        
#         Your role is to:
#         1. Apply advanced graph algorithms to medical networks
#         2. Identify key patterns, hubs, and influences in the medical graph
#         3. Discover hidden relationships and pathways
#         4. Extract actionable insights from complex network structures
        
#         Database schema:
#         {json.dumps(graph_schema, indent=2)}
        
#         You should analyze:
#         - Centrality and influence of providers and organizations
#         - Community detection to find related medical entities
#         - Path analysis for treatment flows
#         - Structural patterns that reveal medical practice behaviors
        
#         Always focus on the graph-theoretical implications and what they mean for healthcare.
#         """

#         # Tools specific to this agent
#         agent_tools = [text_to_nx_algorithm_to_text]
        
#         # Initialize LLM
#         llm = create_llm()
        
#         # Create the agent
#         agent = create_agent(
#             llm, 
#             agent_tools, 
#             system_prompt,
#             max_iterations=3,
#             max_execution_time=180
#         )
        
#         # Set agent name in callback
#         state["callback"].write_agent_name("Graph Analysis Expert 🕸️")
        
#         # Stream mode will give us intermediate steps
#         chunks = []
#         print(f"\n=== Agent: Graph Analysis Expert 🕸️ ===")
#         print(f"Processing query with {len(state['messages'])} messages")
        
#         # Debug: Print message types
#         for i, msg in enumerate(state["messages"]):
#             print(f"  Message {i}: {type(msg).__name__}")
            
#         for chunk in agent.stream(
#             {"messages": state["messages"]},
#             {"callbacks": [state["callback"]]},
#             stream_mode="updates"  # This gives us intermediate steps
#         ):
#             chunks.append(chunk)
            
#         # Get final state from chunks
#         final_messages = chunks[-1].get("messages", []) if chunks else []
        
#         # Debug: Print final messages
#         print(f"Agent returned {len(final_messages)} messages")
        
#         # Update state with agent's response
#         state["messages"] = final_messages
        
#         # If no messages were returned, try to extract the final response from intermediate steps
#         if not final_messages or len(final_messages) == 0:
#             print("No messages found in final state, checking intermediate steps")
#             # Try to find the last AIMessage in the chunks
#             for chunk in reversed(chunks):
#                 if "messages" in chunk and chunk["messages"]:
#                     for msg in reversed(chunk["messages"]):
#                         if isinstance(msg, AIMessage):
#                             print(f"Found AI message in intermediate steps")
#                             state["messages"].append(msg)
#                             break
#                     if state["messages"]:
#                         break
            
#             # If still no messages, create a fallback response
#             if not state["messages"]:
#                 print("No messages found in intermediate steps, creating fallback response")
#                 state["messages"].append(
#                     AIMessage(content="I processed your query but couldn't generate a complete response. Please try rephrasing your question.")
#                 )
        
#         # Store internal state
#         state["internal_state"]["graph_agent_executed"] = True
#         if state["messages"] and isinstance(state["messages"][-1], AIMessage):
#             state["internal_state"]["last_graph_agent_response"] = state["messages"][-1].content
#         else:
#             print("No AI message found in final messages")
#             state["internal_state"]["last_graph_agent_response"] = "No response generated"
        
#         # Print final state of messages
#         print(f"Final state has {len(state['messages'])} messages")
#         for i, msg in enumerate(state["messages"]):
#             print(f"  Message {i}: {type(msg).__name__}")
        
#     except Exception as e:
#         # Handle errors
#         error_msg = f"Graph Analysis Agent encountered an error: {str(e)}"
#         print(f"ERROR: {error_msg}")
#         state["messages"].append(
#             AIMessage(content=error_msg)
#         )
#         state["internal_state"]["graph_agent_error"] = str(e)
    
#     return state

# #############################################################################
# # 3. Patient Data Agent - For patient-specific analysis
# #############################################################################

# def patient_data_agent(state: AgentState):
#     """
#     Agent that specializes in analyzing individual patient data.
#     Focused on comprehensive patient history, treatment timelines, and personalized insights.
#     """
#     try:
#         system_prompt = f"""You are a clinical data specialist focused on individual patient analysis.
        
#         Your role is to:
#         1. Analyze comprehensive patient medical histories
#         2. Identify patterns in individual patient care
#         3. Evaluate treatment effectiveness for specific patients
#         4. Find similar patients for comparative analysis
        
#         Database schema:
#         {json.dumps(graph_schema, indent=2)}
        
#         You should consider:
#         - Longitudinal analysis of patient conditions and treatments
#         - Treatment response patterns
#         - Care coordination across providers
#         - Cost-effectiveness of personalized treatments
        
#         Always maintain clinical relevance and prioritize insights that could improve patient care.
#         """

#         # Tools specific to this agent
#         agent_tools = [
#             query_patient_history,
#             find_similar_patients
#         ]
        
#         # Initialize LLM
#         llm = create_llm()
        
#         # Create the agent
#         agent = create_agent(
#             llm, 
#             agent_tools, 
#             system_prompt,
#             max_iterations=3,
#             max_execution_time=180
#         )
        
#         # Set agent name in callback
#         state["callback"].write_agent_name("Patient Analysis Expert 👨‍⚕️")
        
#         # Stream mode will give us intermediate steps
#         chunks = []
#         print(f"\n=== Agent: Patient Analysis Expert 👨‍⚕️ ===")
#         print(f"Processing query with {len(state['messages'])} messages")
        
#         # Debug: Print message types
#         for i, msg in enumerate(state["messages"]):
#             print(f"  Message {i}: {type(msg).__name__}")
        
#         # Save original messages
#         original_messages = state["messages"].copy()
            
#         for chunk in agent.stream(
#             {"messages": state["messages"]},
#             {"callbacks": [state["callback"]]},
#             stream_mode="updates"  # This gives us intermediate steps
#         ):
#             chunks.append(chunk)
            
#         # Get final state from chunks
#         final_messages = chunks[-1].get("messages", []) if chunks else []
        
#         # Debug: Print final messages
#         print(f"Agent returned {len(final_messages)} messages")
        
#         # Get the response from the callback
#         ai_response = state["callback"].get_final_response()
        
#         if ai_response:
#             print(f"Found response in callback: {ai_response[:50]}...")
#             # Store the response in internal state
#             state["internal_state"]["last_patient_agent_response"] = ai_response
            
#             # Keep original messages and append the AI response
#             state["messages"] = original_messages
#             state["messages"].append(AIMessage(content=ai_response))
#         else:
#             # If no response in callback, try to extract from chunks
#             if final_messages and len(final_messages) > 0:
#                 # Update state with agent's response
#                 state["messages"] = final_messages
                
#                 # Try to extract the AI message
#                 for msg in reversed(final_messages):
#                     if isinstance(msg, AIMessage):
#                         state["internal_state"]["last_patient_agent_response"] = msg.content
#                         break
#             else:
#                 print("No messages found in final state, checking intermediate steps")
#                 # Try to find the last AIMessage in the chunks
#                 for chunk in reversed(chunks):
#                     if "messages" in chunk and chunk["messages"]:
#                         for msg in reversed(chunk["messages"]):
#                             if isinstance(msg, AIMessage):
#                                 print(f"Found AI message in intermediate steps")
#                                 # Keep original messages and append the AI response
#                                 state["messages"] = original_messages
#                                 state["messages"].append(msg)
#                                 state["internal_state"]["last_patient_agent_response"] = msg.content
#                                 break
#                         if len(state["messages"]) > len(original_messages):
#                             break
                
#                 # If still no messages, create a fallback response
#                 if len(state["messages"]) == len(original_messages):
#                     print("No messages found in intermediate steps, creating fallback response")
#                     fallback_msg = "I processed your query but couldn't generate a complete response. Please try rephrasing your question."
#                     state["messages"] = original_messages
#                     state["messages"].append(AIMessage(content=fallback_msg))
#                     state["internal_state"]["last_patient_agent_response"] = fallback_msg
        
#         # Print final state of messages
#         print(f"Final state has {len(state['messages'])} messages")
#         for i, msg in enumerate(state["messages"]):
#             print(f"  Message {i}: {type(msg).__name__}")
        
#     except Exception as e:
#         # Handle errors
#         error_msg = f"Patient Data Agent encountered an error: {str(e)}"
#         print(f"ERROR: {error_msg}")
#         state["messages"].append(AIMessage(content=error_msg))
#         state["internal_state"]["patient_agent_error"] = str(e)
#         state["internal_state"]["last_patient_agent_response"] = error_msg
    
#     return state

# #############################################################################
# # 4. Population Health Agent - For population-level analysis
# #############################################################################

# def population_health_agent(state: AgentState):
#     """
#     Agent that specializes in population-level health analysis.
#     Focused on trends, patterns, and insights across the entire patient population.
#     """
#     try:
#         system_prompt = f"""You are a population health analyst specializing in medical data trends.
        
#         Your role is to:
#         1. Identify patterns and trends across large patient populations
#         2. Analyze condition prevalence and distribution
#         3. Evaluate treatment effectiveness at scale
#         4. Discover correlations between medical factors
        
#         Database schema:
#         {json.dumps(graph_schema, indent=2)}
        
#         You should analyze:
#         - Disease prevalence and distribution
#         - Treatment pathways and outcomes
#         - Provider practice patterns
#         - Cost and resource utilization trends
        
#         Always consider statistical significance and focus on actionable population health insights.
#         """

#         # Tools specific to this agent
#         agent_tools = [
#             search_conditions,
#             analyze_medications,
#             analyze_treatment_pathways
#         ]
        
#         # Initialize LLM
#         llm = create_llm()
        
#         # Create the agent
#         agent = create_agent(
#             llm, 
#             agent_tools, 
#             system_prompt,
#             max_iterations=3,
#             max_execution_time=180
#         )
        
#         # Set agent name in callback
#         state["callback"].write_agent_name("Population Health Expert 📈")
        
#         # Stream mode will give us intermediate steps
#         chunks = []
#         print(f"\n=== Agent: Population Health Expert 📈 ===")
#         print(f"Processing query with {len(state['messages'])} messages")
        
#         # Debug: Print message types
#         for i, msg in enumerate(state["messages"]):
#             print(f"  Message {i}: {type(msg).__name__}")
        
#         # Save original messages
#         original_messages = state["messages"].copy()
            
#         for chunk in agent.stream(
#             {"messages": state["messages"]},
#             {"callbacks": [state["callback"]]},
#             stream_mode="updates"  # This gives us intermediate steps
#         ):
#             chunks.append(chunk)
            
#         # Get final state from chunks
#         final_messages = chunks[-1].get("messages", []) if chunks else []
        
#         # Debug: Print final messages
#         print(f"Agent returned {len(final_messages)} messages")
        
#         # Get the response from the callback
#         ai_response = state["callback"].get_final_response()
        
#         if ai_response:
#             print(f"Found response in callback: {ai_response[:50]}...")
#             # Store the response in internal state
#             state["internal_state"]["last_population_agent_response"] = ai_response
            
#             # Keep original messages and append the AI response
#             state["messages"] = original_messages
#             state["messages"].append(AIMessage(content=ai_response))
#         else:
#             # If no response in callback, try to extract from chunks
#             if final_messages and len(final_messages) > 0:
#                 # Update state with agent's response
#                 state["messages"] = final_messages
                
#                 # Try to extract the AI message
#                 for msg in reversed(final_messages):
#                     if isinstance(msg, AIMessage):
#                         state["internal_state"]["last_population_agent_response"] = msg.content
#                         break
#             else:
#                 print("No messages found in final state, checking intermediate steps")
#                 # Try to find the last AIMessage in the chunks
#                 for chunk in reversed(chunks):
#                     if "messages" in chunk and chunk["messages"]:
#                         for msg in reversed(chunk["messages"]):
#                             if isinstance(msg, AIMessage):
#                                 print(f"Found AI message in intermediate steps")
#                                 # Keep original messages and append the AI response
#                                 state["messages"] = original_messages
#                                 state["messages"].append(msg)
#                                 state["internal_state"]["last_population_agent_response"] = msg.content
#                                 break
#                         if len(state["messages"]) > len(original_messages):
#                             break
                
#                 # If still no messages, create a fallback response
#                 if len(state["messages"]) == len(original_messages):
#                     print("No messages found in intermediate steps, creating fallback response")
#                     fallback_msg = "I processed your query but couldn't generate a complete response. Please try rephrasing your question."
#                     state["messages"] = original_messages
#                     state["messages"].append(AIMessage(content=fallback_msg))
#                     state["internal_state"]["last_population_agent_response"] = fallback_msg
        
#         # Print final state of messages
#         print(f"Final state has {len(state['messages'])} messages")
#         for i, msg in enumerate(state["messages"]):
#             print(f"  Message {i}: {type(msg).__name__}")
        
#     except Exception as e:
#         # Handle errors
#         error_msg = f"Population Health Agent encountered an error: {str(e)}"
#         print(f"ERROR: {error_msg}")
#         state["messages"].append(AIMessage(content=error_msg))
#         state["internal_state"]["population_agent_error"] = str(e)
#         state["internal_state"]["last_population_agent_response"] = error_msg
    
#     return state




# # Tools for each specialized agent
# AQL_QUERY_TOOLS = [text_to_aql_to_text]
# GRAPH_ANALYSIS_TOOLS = [text_to_nx_algorithm_to_text]
# PATIENT_DATA_TOOLS = [query_patient_history, search_conditions]
# POPULATION_HEALTH_TOOLS = [analyze_medications, analyze_treatment_pathways]


# def test_all_agents():
#     """Run tests for all agents with appropriate queries"""
#     test_queries = {
#         "aql_query_agent": [
#             "Find all patients who have been diagnosed with hypertension and are on beta blockers",
#             "What are the most common medications prescribed for diabetes patients?"
#         ],
#         "graph_analysis_agent": [
#             "Which providers have the highest centrality in the referral network?",
#             "Find communities of providers who frequently collaborate on patient care"
#         ],
#         "patient_data_agent": [
#             "Give me a complete medical history for patient with ID '7c2e78bd-52cf-1fce-acc3-0ddd93104abe'",
#             "Find patients similar to patient '7c2e78bd-52cf-1fce-acc3-0ddd93104abe' based on their conditions"
#         ],
#         "population_health_agent": [
#             "What are the most common conditions in the database and their frequencies?",
#             "Analyze the typical treatment pathway for patients with COPD"
#         ]
#     }
    
#     results = {}
    
#     # Map agent names to their functions
#     agent_functions = {
#         "aql_query_agent": aql_query_agent,
#        # "graph_analysis_agent": graph_analysis_agent,
#         "patient_data_agent": patient_data_agent,
#         "population_health_agent": population_health_agent
#     }
    
#     # Test each agent
#     for agent_name, agent_func in agent_functions.items():
#         print("\n" + "="*80)
#         print(f"TESTING {agent_name.upper()}")
#         print("="*80)
        
#         for query in test_queries[agent_name]:
#             print(f"\nQuery: {query}")
            
#             # Create callback handler
#             callback = CustomConsoleCallbackHandler()
            
#             # Create initial state
#             state = AgentState(
#                 messages=[HumanMessage(content=query)],
#                 current_date=datetime.now().isoformat(),
#                 context={},
#                 internal_state={},
#                 callback=callback
#             )
            
#             # Run the agent
#             result_state = agent_func(state)
            
#             # Get the response from callback first
#             response = callback.get_final_response()
            
#             # If no response from callback, try getting from messages
#             if not response and result_state["messages"]:
#                 for msg in reversed(result_state["messages"]):
#                     if isinstance(msg, AIMessage):
#                         response = msg.content
#                         break
            
#             # If still no response, check internal state
#             if not response:
#                 internal_state_key = f"last_{agent_name.split('_')[0]}_agent_response"
#                 if internal_state_key in result_state["internal_state"]:
#                     response = result_state["internal_state"][internal_state_key]
            
#             # Store the result
#             results[f"{agent_name}_{query}"] = response
            
#             # Print the result
#             print("\nRESULT:")
#             print("-"*80)
#             print(response if response else "No response generated")
#             print("-"*80)
    
#     return results


def run_aql_agent(question: str):
    """
    Runs the AQL agent with the given question.
    
    Args:
        question: The user's question
    """
    # Create callback handler
    callback = CustomConsoleCallbackHandler()
    
    initial_state = {
        "messages": [HumanMessage(content=question)], 
        "context": {},
        "internal_state": {},
        "callback": callback,
        "aql_query_agent_internal_state": {}
    }
    
    try:
        # Single invocation
        result = aql_query_node(initial_state)
        
        # Extract the response directly from the messages
        if result["messages"]:
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    return msg.content
        
        # If no AI message found in messages, try the internal state
        if result["aql_query_agent_internal_state"].get("full_response", {}).get("messages"):
            messages = result["aql_query_agent_internal_state"]["full_response"]["messages"]
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    return msg.content
        
        # If still no response, check the final LLM response in the console output
        return "No response generated. Please check the console output for details."
        
    except Exception as e:
        return f"Error running agent: {str(e)}"



# Example usage:
if __name__ == "__main__":
    # Basic usage
    question = "What is the average age of patients in the database?"
    result = run_aql_agent(question)
    print(f"Basic result: {result}")
    
