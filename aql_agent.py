import os
import json
import time
import re
import networkx as nx
# import nx_arangodb as nxadb
import pandas as pd
import matplotlib.pyplot as plt
from typing import Annotated, Dict, TypedDict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel
import atexit
import pickle

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

# Memory imports
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

# ArangoDB imports
from arango import ArangoClient
from tools import *
from callback import *
from settings import *
from dotenv import load_dotenv
from memory import get_memory_tools, get_memory_store
load_dotenv()


store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)


# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGSMITH_PROJECT"] = "arangodb-cugraph"
# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")




# Tools for each specialized agent
AQL_QUERY_TOOLS = [text_to_aql_to_text]


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



def create_agent(state, memory, llm: ChatOpenAI, tools: list, system_prompt: str, max_iterations: int = 2, max_execution_time: int = 120) -> AgentExecutor:
    """
    Creates a LangGraph react agent using the specified ChatOpenAI model, tools, and system prompt.
    
    Args:
        llm: LLM to be used to create the agent
        tools: List of tools to be given to the agent
        system_prompt: System prompt to be used in the agent
        max_iterations: Maximum number of iterations (will be converted to recursion_limit)
        max_execution_time: Maximum execution time in seconds
    """
    def _modify_state_messages(state, memory):
        # Retrieve relevant memories
        memories = memory.search(state["messages"][-1], user_id=state["mem0_user_id"])
        context = "Relevant information from previous conversations:\n"
        for memory__ in memories["results"]:
            context += f"- {memory__['memory']}\n"
        # Add system prompt and keep existing messages
        full_prompt = system_prompt + "\n" + "Use the provided context to personalize your responses and remember user preferences and past interactions." + "\n" + context + "User-", " ".join(state["messages"])
        return full_prompt[0]
    
    # Create the react agent
        # Add memory tools to the existing tools
    memory_tools = get_memory_tools("aql_query")
    all_tools = tools + memory_tools
    
    # Create the react agent with memory capabilities
    agent = create_react_agent(
        llm, 
        all_tools, 
        prompt=_modify_state_messages(state, memory),
        store=get_memory_store()
    )
    
    # Set recursion limit (LangGraph uses 2 steps per iteration + 1)
    agent.recursion_limit = 2 * max_iterations + 1
    
    # Set timeout for each step
    agent.step_timeout = max_execution_time
    
    return agent

system_prompt = f"""
You are an expert medical data analyst specializing in ArangoDB graph queries.
        
Your role is to:
1. Convert natural language medical questions into precise AQL queries AND use the provided AQL_QUERY_TOOLS tools for query execution
2. Execute these queries against the SYNTHEA_P100 database
3. Interpret the results in a medically meaningful way
4. Provide clear, structured responses that medical professionals can use
5. Maintain and utilize memory of past interactions to provide context-aware responses

Memory Management:
1. Store detailed information from each interaction, including:
   - Patient demographics (age, gender, conditions) when discussed
   - Specific medical conditions or diagnoses mentioned
   - Query patterns and their results
   - Statistical findings and trends
   - Treatment outcomes and medications discussed
   - Any temporal patterns or longitudinal data
   
2. When storing information, always include:
   - The context of the query
   - Key numerical findings
   - Relevant patient cohort characteristics
   - Date and scope of the analysis

3. Search and reference past interactions to:
   - Compare current findings with previous analyses
   - Identify trends across multiple queries
   - Build upon previous insights
   - Maintain consistency in analysis

4. Actively use stored information to:
   - Provide historical context for similar queries
   - Track changes in patient populations over time
   - Build comprehensive patient profiles
   - Identify recurring patterns in medical data

Database schema:
{json.dumps(graph_schema, indent=2)}
        
Important guidelines:
1. ALWAYS utilize the provided AQL_QUERY_TOOLS tools for query execution
2. Before executing any query, validate that all required tools are accessible
3. Use memory tools to store and retrieve relevant context
4. Maintain consistency in responses by referencing past interactions
        
Always use proper medical terminology and provide context for your findings.
"""
llm = create_llm()

def aql_query_node(state):
    """
    Handles fundamental analysis and financial metrics using tools from tools.py
    """
 
    
    
    try:
        aql_agent = create_agent(
            llm, 
            AQL_QUERY_TOOLS, 
            system_prompt,
            max_iterations=2,
            max_execution_time=120
        )
        
        state["callback"].write_agent_name("AQL Query Agent 📊")
        
        # Stream mode will give us intermediate steps
        chunks = []
        for chunk in aql_agent.stream(
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
        state["aql_query_agent_internal_state"]["agent_executor_tools"] = {
            tool.name: 0 for tool in AQL_QUERY_TOOLS
        }
        state["aql_query_agent_internal_state"]["full_response"] = {
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


def run_aql_agent(state, memory, question: str, current_date: str = None):
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
        "aql_query_agent_internal_state": {}
    }
    
    # Basic usage
    try:
        # Create agent directly like in the other methods
        aql_agent = create_agent(
            state,
            memory,
            llm,
            AQL_QUERY_TOOLS,
            system_prompt,
            max_iterations=2,
            max_execution_time=120
        )
        
        # Use invoke instead of aql_query_node
        result = aql_agent.invoke(
            {"messages": initial_state["messages"]},
            {"callbacks": [initial_state["callback"]]}
        )
        memory.add(f"User: {state['messages'][-1]}\nAssistant: {result}", user_id=state['mem0_user_id'], agent_id=state['agent_id'])
        print("memory -:", memory.get_all(user_id=state['mem0_user_id'], agent_id=state['agent_id']))
        # Get the last message content and AQL query if available
        if result["messages"] and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            if isinstance(last_message, tuple) and len(last_message) == 2:
                return last_message  # Return the tuple directly
            else:
                return last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            return "No response generated"
        
    except Exception as e:
        return f"Error running agent: {str(e)}"

def run_aql_agent_with_stream(state, memory, question: str, current_date: str = None):
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
        "aql_query_agent_internal_state": {}
    }
    
    # Create agent
    aql_agent = create_agent(
        state,
        memory,
        llm,
        AQL_QUERY_TOOLS,
        system_prompt,
        max_iterations=2,
        max_execution_time=120
    )
    
    # Stream with intermediate steps
    try:
        for chunk in aql_agent.stream(
            {"messages": initial_state["messages"]},
            {"callbacks": [initial_state["callback"]]},
            stream_mode="updates"
        ):
            # If the chunk contains a tuple response, yield both parts
            if isinstance(chunk, dict) and 'messages' in chunk:
                last_message = chunk['messages'][-1] if chunk['messages'] else None
                if isinstance(last_message, tuple) and len(last_message) == 2:
                    yield {"response": last_message[0], "aql_query": last_message[1]}
                else:
                    yield chunk
            else:
                yield chunk
        memory.add(f"User: {state['messages'][-1]}\nAssistant: {chunk}", user_id=state['mem0_user_id'], agent_id=state['agent_id'])
        print("memory -:", memory.get_all(user_id=state['mem0_user_id'], agent_id=state['agent_id']))
    except Exception as e:
        yield {"error": f"Error streaming agent: {str(e)}"}

async def run_aql_agent_async(question: str, current_date: str = None, timeout: int = 300):
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

    
    aql_agent = create_agent(
        llm,
        AQL_QUERY_TOOLS,
        system_prompt,
        max_iterations=2,
        max_execution_time=120
    )
    
    try:
        # Run with overall timeout
        task = asyncio.create_task(
            aql_agent.ainvoke(
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
    question = "What is the average age of patients in the database?"
    result = run_aql_agent(question)
    print(f"Basic result: {result}")
    
    #Streaming usage
    print("\nStreaming results:")
    for chunk in run_aql_agent_with_stream(question):
        print(f"Chunk: {chunk}")
    
    # Async usage with timeout
    print("\nAsync result:")
    async def main():
        result = await run_aql_agent_async(question, timeout=300)
        print(f"Async result: {result}")
    
    asyncio.run(main())