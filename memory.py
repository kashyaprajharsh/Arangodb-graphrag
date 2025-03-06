from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from settings import *
# Create memory store with OpenAI embeddings
memory_store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

def get_memory_tools(namespace: str):
    """
    Creates memory tools with a specific namespace for each agent.
    
    Args:
        namespace: The namespace to use for this agent's memory
    
    Returns:
        List of memory tools configured for the namespace
    """
    return [
        create_manage_memory_tool(namespace=(namespace,)),
        create_search_memory_tool(namespace=(namespace,))
    ]

def get_memory_store():
    """
    Returns the global memory store instance.
    """
    return memory_store 