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
from langchain.agents import (
    AgentExecutor,
    create_openai_tools_agent,
)
import asyncio
# ArangoDB imports
from arango import ArangoClient
from tools import *
from callback import *
from settings import *
from memory import get_memory_tools, get_memory_store

# Tools for each specialized agent
GRAPH_ANALYSIS_TOOLS = [text_to_nx_algorithm_to_text]


class AgentState(TypedDict):
    messages: list  
    context: Dict
    internal_state: Dict
    callback: Any


def create_llm():
    """Create a standard LLM for all agents"""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1
    )



def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str, max_iterations: int = 2, max_execution_time: int = 120) -> AgentExecutor:
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
    
    # Add memory tools to the existing tools
    memory_tools = get_memory_tools("graph_analysis")
    all_tools = tools + memory_tools
    
    # Create the react agent with memory capabilities
    agent = create_react_agent(
        llm, 
        all_tools, 
        prompt=_modify_state_messages,
        store=get_memory_store()
    )
    
    # Set recursion limit (LangGraph uses 2 steps per iteration + 1)
    agent.recursion_limit = 2 * max_iterations + 1
    
    # Set timeout for each step
    agent.step_timeout = max_execution_time
    
    return agent


system_prompt = f"""You are an expert graph analyst specializing in medical network analysis.
        
        Your role is to:
        1. Apply advanced graph algorithms to medical networks
        2. Identify key patterns, hubs, and influences in the medical graph
        3. Discover hidden relationships and pathways
        4. Extract actionable insights from complex network structures
        Remember I am using Directed Graphs so in code where required use "directed = True".
        Database schema:
        {json.dumps(graph_schema, indent=2)}
        
        You should analyze:
        - Centrality and influence of providers and organizations
        - Community detection to find related medical entities
        - Path analysis for treatment flows
        - Structural patterns that reveal medical practice behaviors
        
        Always focus on the graph-theoretical implications and what they mean for healthcare.
        """



llm = create_llm()

def graph_analysis_node(state):
    """
    Handles fundamental analysis and financial metrics using tools from tools.py
    """
 
    
    
    try:
        graph_analysis_agent = create_agent(
            llm, 
            GRAPH_ANALYSIS_TOOLS, 
            system_prompt,
            max_iterations=2,
            max_execution_time=120
        )
        
        state["callback"].write_agent_name("Graph Analysis Agent 📊")
        
        # Stream mode will give us intermediate steps
        chunks = []
        for chunk in graph_analysis_agent.stream(
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
        state["graph_analysis_agent_internal_state"]["agent_executor_tools"] = {
            tool.name: 0 for tool in GRAPH_ANALYSIS_TOOLS
        }
        state["graph_analysis_agent_internal_state"]["full_response"] = {
            "messages": final_messages,
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


def run_graph_analysis_agent(question: str, current_date: str = None):
    """
    Runs the financial agent with the given question.
    
    Args:
        question: The user's question
        current_date: Optional date context
    """
    # Initialize state with datetime object
    if current_date and isinstance(current_date, str):
        current_date = datetime.strptime(current_date, "%Y-%m-%d")
    else:
        current_date = datetime.now()

    initial_state = {
        "messages": [("human", question)],
        "user_input": question,
        "callback": CustomConsoleCallbackHandler(),
        "graph_analysis_agent_internal_state": {}
    }
    
    # Basic usage
    try:
        # Create agent directly like in the other methods
        graph_analysis_agent = create_agent(
            llm,
            GRAPH_ANALYSIS_TOOLS,
            system_prompt,
            max_iterations=2,
            max_execution_time=120
        )
        
        # Use invoke instead of aql_query_node
        result = graph_analysis_agent.invoke(
            {"messages": initial_state["messages"]},
            {"callbacks": [initial_state["callback"]]}
        )
        
        # Get the last message content
        if result["messages"] and len(result["messages"]) > 0:
            return result["messages"][-1].content
        else:
            return "No response generated"
        
    except Exception as e:
        return f"Error running agent: {str(e)}"

def run_graph_analysis_agent_with_stream(question: str, current_date: str = None):
    """
    Runs the financial agent with streaming output.
    
    Args:
        question: The user's question
        current_date: Optional date context
        personality: Optional personality configuration
    """
    # Initialize state with datetime object
    if current_date and isinstance(current_date, str):
        current_date = datetime.strptime(current_date, "%Y-%m-%d")
    else:
        current_date = datetime.now()

    initial_state = {
        "messages": [("human", question)],
        "user_input": question,
        "callback": CustomConsoleCallbackHandler(),
        "graph_analysis_agent_internal_state": {}
    }
    
    # Create agent
   
    
    graph_analysis_agent = create_agent(
        llm,
        GRAPH_ANALYSIS_TOOLS,
        system_prompt,
        max_iterations=2,
        max_execution_time=120
    )
    
    # Stream with intermediate steps
    try:
        for chunk in graph_analysis_agent.stream(
            {"messages": initial_state["messages"]},
            {"callbacks": [initial_state["callback"]]},
            stream_mode="updates"
        ):
            yield chunk
            
    except Exception as e:
        yield {"error": f"Error streaming agent: {str(e)}"}

async def run_graph_analysis_agent_async(question: str, current_date: str = None, timeout: int = 300):
    """
    Runs the financial agent asynchronously with a total timeout.
    
    Args:
        question: The user's question
        current_date: Optional date context
        timeout: Total timeout in seconds
    """
    # Initialize state with datetime object
    if current_date and isinstance(current_date, str):
        current_date = datetime.strptime(current_date, "%Y-%m-%d")
    else:
        current_date = datetime.now()

    initial_state = {
        "messages": [("human", question)],
        "user_input": question,
        "callback": CustomConsoleCallbackHandler(),
        "aql_query_agent_internal_state": {}
    }

    
    graph_analysis_agent = create_agent(
        llm,
        GRAPH_ANALYSIS_TOOLS,
        system_prompt,
        max_iterations=2,
        max_execution_time=120
    )
    
    try:
        # Run with overall timeout
        task = asyncio.create_task(
            graph_analysis_agent.ainvoke(
                {"messages": initial_state["messages"]},
                {"callbacks": [initial_state["callback"]]}
            )
        )
        result = await asyncio.wait_for(task, timeout=timeout)
        return result["messages"][-1].content
        
    except asyncio.TimeoutError:
        return "Agent stopped due to overall timeout."
    except Exception as e:
        return f"Error running agent: {str(e)}"


# Example usage:
if __name__ == "__main__":
    # Basic usage
    question = "find node with highest betweenness centrality in the graph"
    result = run_graph_analysis_agent(question)
    print(f"Basic result: {result}")
    
    #Streaming usage
    print("\nStreaming results:")
    for chunk in run_graph_analysis_agent_with_stream(question):
        print(f"Chunk: {chunk}")
    
    # Async usage with timeout
    print("\nAsync result:")
    async def main():
        result = await run_graph_analysis_agent_async(question, timeout=300)
        print(f"Async result: {result}")
    
    asyncio.run(main())