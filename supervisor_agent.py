import os
import json
import time
import re
import asyncio
from typing import Dict, List, Any, TypedDict, Optional
from datetime import datetime
from pydantic import BaseModel

# LangChain and LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# Import our specialized agents
from aql_agent import create_agent as create_aql_agent, AQL_QUERY_TOOLS, system_prompt as aql_system_prompt
from graph_agent import create_agent as create_graph_agent, GRAPH_ANALYSIS_TOOLS, system_prompt as graph_system_prompt
from patient_data_agent import create_agent as create_patient_agent, PATIENT_DATA_TOOLS, system_prompt as patient_system_prompt
from population_health_agent import create_agent as create_population_agent, POPULATION_HEALTH_TOOLS, system_prompt as population_system_prompt
from callback import CustomConsoleCallbackHandler
from tools import graph_schema
from settings import *

class AgentState(TypedDict):
    messages: list  
    context: Dict
    internal_state: Dict
    callback: Any

def create_llm(model_name="gpt-4o-mini", temperature=0.1):
    """Create a standard LLM for all agents"""
    return ChatOpenAI(
        model=model_name,
        temperature=temperature
    )

# Create the specialized agents
def create_specialized_agents(llm):
    """Create all specialized agents using the provided LLM"""
    
    # AQL Query Agent
    aql_agent = create_aql_agent(
        llm=llm,
        tools=AQL_QUERY_TOOLS,
        system_prompt=aql_system_prompt,
        max_iterations=2,
        max_execution_time=120
    )
    aql_agent.name = "aql_agent"  # Add name to the agent
    
    # Graph Analysis Agent
    graph_agent = create_graph_agent(
        llm=llm,
        tools=GRAPH_ANALYSIS_TOOLS,
        system_prompt=graph_system_prompt,
        max_iterations=2,
        max_execution_time=120
    )
    graph_agent.name = "graph_agent"  # Add name to the agent
    
    # Patient Data Agent
    patient_agent = create_patient_agent(
        llm=llm,
        tools=PATIENT_DATA_TOOLS,
        system_prompt=patient_system_prompt,
        max_iterations=2,
        max_execution_time=120
    )
    patient_agent.name = "patient_agent"  # Add name to the agent
    
    # Population Health Agent
    population_agent = create_population_agent(
        llm=llm,
        tools=POPULATION_HEALTH_TOOLS,
        system_prompt=population_system_prompt,
        max_iterations=2,
        max_execution_time=120
    )
    population_agent.name = "population_agent"  # Add name to the agent
    
    return {
        "aql_agent": aql_agent,
        "graph_agent": graph_agent,
        "patient_agent": patient_agent,
        "population_agent": population_agent
    }

def create_supervisor_workflow(agents, llm=None, memory=True):
    """
    Create a supervisor workflow that coordinates between specialized agents.
    
    Args:
        agents: Dictionary of specialized agents
        llm: LLM to use for the supervisor (if None, will create a new one)
        memory: Whether to use memory for the supervisor
    
    Returns:
        Compiled supervisor workflow
    """
    if llm is None:
        llm = create_llm(model_name="gpt-4o", temperature=0.1)
    
    # Create a list of agents for the supervisor
    agent_list = [
        agents["aql_agent"],
        agents["graph_agent"],
        agents["patient_agent"],
        agents["population_agent"]
    ]
    
    # Create the supervisor prompt
    supervisor_prompt = f"""
    You are a medical data analysis supervisor coordinating a team of specialized agents.
    
    Your team consists of:
    
    1. AQL Query Agent: Expert in converting natural language to AQL queries and executing them against the medical database.
       - Use for: Database queries, data retrieval, complex data filtering
       - Capabilities: Converts natural language to AQL, executes queries, interprets results
       - To delegate: Use the handoff keyword "aql_agent"
    
    2. Graph Analysis Agent: Expert in medical network analysis using graph algorithms.
       - Use for: Network analysis, finding patterns in connected data, identifying key nodes/relationships
       - Capabilities: Applies graph algorithms, identifies patterns, discovers relationships
       - To delegate: Use the handoff keyword "graph_agent"
    
    3. Patient Data Agent: Clinical specialist focused on individual patient analysis.
       - Use for: Individual patient history, treatment patterns for specific patients
       - Capabilities: Analyzes patient histories, evaluates treatments for specific patients
       - To delegate: Use the handoff keyword "patient_agent"
    
    4. Population Health Agent: Expert in analyzing trends across patient populations.
       - Use for: Population-level trends, condition prevalence, treatment effectiveness at scale
       - Capabilities: Identifies patterns across populations, analyzes condition prevalence
       - To delegate: Use the handoff keyword "population_agent"
    
    Database schema:
    {json.dumps(graph_schema, indent=2)}
    
    Your job is to:
    1. Analyze the user's medical data question
    2. Select the most appropriate agent(s) to handle the task
    3. Delegate to the selected agent by using their handoff keyword (e.g., "aql_agent", "graph_agent", etc.)
    4. When control returns to you, analyze the agent's response and either:
       a) Provide it to the user if it fully answers their question
       b) Delegate to another agent if more analysis is needed
       c) Ask for clarification if the response is unclear
    
    Guidelines for agent selection:
    - For database queries and data retrieval: Delegate to "aql_agent"
    - For network/graph analysis and relationship discovery: Delegate to "graph_agent"
    - For individual patient analysis: Delegate to "patient_agent"
    - For population-level trends and statistics: Delegate to "population_agent"
    
    Always select the most appropriate agent for the task. Multiple agents may be needed for complex queries.
    When delegating to an agent, use their exact handoff keyword as specified above.
    
    When receiving results from an agent, analyze them carefully before deciding next steps.
    Your final response to the user should synthesize all the information gathered from the agents into a clear, comprehensive answer.
    """
    
    # Create the supervisor workflow
    workflow = create_supervisor(
        agents=agent_list,
        model=llm,
        prompt=supervisor_prompt,
        add_handoff_back_messages=True,
        output_mode="full_history",  # Include full message history
        supervisor_name="medical_supervisor"  # Give a descriptive name to the supervisor
    )
    
    # Compile with memory if requested
    if memory:
        checkpointer = MemorySaver()
        store = InMemoryStore()
        return workflow.compile(checkpointer=checkpointer, store=store)
    else:
        return workflow.compile()

def run_supervisor(question: str, current_date: str = None, thread_id: str = "default_thread"):
    """
    Run the supervisor agent with the given question.
    
    Args:
        question: The user's question
        current_date: Optional date context
        thread_id: Thread ID for the checkpointer
    
    Returns:
        The response from the selected agent
    """
    # Initialize date
    if current_date and isinstance(current_date, str):
        current_date = datetime.strptime(current_date, "%Y-%m-%d")
    else:
        current_date = datetime.now()
    
    # Create LLM
    llm = create_llm(model_name="gpt-4o-mini", temperature=0.1)
    
    # Create specialized agents
    agents = create_specialized_agents(llm)
    
    # Create supervisor workflow
    supervisor = create_supervisor_workflow(agents, llm)
    
    # Create config with required configurable parameters
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the supervisor
    result = supervisor.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        },
        config=config
    )
    return result

# # Extract the final response
#     if result and "messages" in result and len(result["messages"]) > 0:
#         last_message = result["messages"][-1]
#         # Handle different message types
#         if hasattr(last_message, 'content'):  # AIMessage or similar object
#             return last_message.content
#         elif isinstance(last_message, dict) and 'content' in last_message:
#             return last_message['content']
#         else:
#             return str(last_message)
#     else:
#         return "No response generated"


async def run_supervisor_async(question: str, current_date: str = None, timeout: int = 300, thread_id: str = "default_thread"):
    """
    Run the supervisor agent asynchronously with a total timeout.
    
    Args:
        question: The user's question
        current_date: Optional date context
        timeout: Total timeout in seconds
        thread_id: Thread ID for the checkpointer
    
    Returns:
        The response from the selected agent
    """
    # Initialize date
    if current_date and isinstance(current_date, str):
        current_date = datetime.strptime(current_date, "%Y-%m-%d")
    else:
        current_date = datetime.now()
    
    # Create LLM
    llm = create_llm(model_name="gpt-4o-mini", temperature=0.1)
    
    # Create specialized agents
    agents = create_specialized_agents(llm)
    
    # Create supervisor workflow
    supervisor = create_supervisor_workflow(agents, llm)
    
    # Create config with required configurable parameters
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Run with overall timeout
        task = asyncio.create_task(
            supervisor.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                },
                config=config
            )
        )
        result = await asyncio.wait_for(task, timeout=timeout)
        
        # Extract the final response
        if result and "messages" in result and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            # Handle different message types
            if hasattr(last_message, 'content'):  # AIMessage or similar object
                return last_message.content
            elif isinstance(last_message, dict) and 'content' in last_message:
                return last_message['content']
            else:
                return str(last_message)
        else:
            return "No response generated"
        
    except asyncio.TimeoutError:
        return "Agent stopped due to overall timeout."
    except Exception as e:
        return f"Error running agent: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Basic usage
    question = "What are the most common conditions for patients"
    result = run_supervisor(question, thread_id="example_thread_1")
    print(f"Basic result: {result}")
    
    # Async usage with timeout
    print("\nAsync result:")
    async def main():
        result = await run_supervisor_async(question, timeout=300, thread_id="example_thread_2")
        print(f"Async result: {result}")
    
    asyncio.run(main())

   