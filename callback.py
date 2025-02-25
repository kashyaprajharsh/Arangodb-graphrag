#############################################################################
# Callback Handler for Agent Execution
#############################################################################


from typing import Annotated, Dict, TypedDict, Any, Optional, List
from langchain.callbacks.base import BaseCallbackHandler


class CustomConsoleCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming output to console"""
    
    def __init__(self):
        """Initialize with empty buffers"""
        self.current_agent_name = None
        self.current_step = None
        self.current_tool = None
        self.final_response = None
        self.last_llm_response = None
        
    def write_agent_name(self, name: str):
        """Write the agent name to the console"""
        self.current_agent_name = name
        print(f"\n=== Agent: {name} ===")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Run when LLM starts running"""
        print("\n🤔 Processing...")
    
    def on_llm_end(self, response, **kwargs):
        """Run when LLM ends running"""
        if hasattr(response, 'generations') and response.generations:
            # Extract the final response text
            final_text = response.generations[0][0].text if response.generations[0] else ""
            self.final_response = final_text
            self.last_llm_response = final_text  # Store the most recent response
            
            # Print the final response
            print("\n Final LLM Response:")
            print("-" * 50)
            print(final_text)
            print("-" * 50)
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Run when tool starts running"""
        tool_name = kwargs.get("name", "unknown_tool")
        self.current_tool = tool_name
        print(f"\n🔧 Using tool: {tool_name}")
    
    def on_tool_end(self, output, **kwargs):
        """Run when tool ends running"""
        print("\n📤 Tool output:")
        print("-" * 50)
        print(output)
        print("-" * 50)
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Run when chain starts running"""
        pass
    
    def on_chain_end(self, outputs, **kwargs):
        """Run when chain ends running"""
        # If this is the final chain output, capture it
        if outputs and isinstance(outputs, dict) and "output" in outputs:
            self.final_response = outputs["output"]
    
    def on_text(self, text, **kwargs):
        """Run on arbitrary text"""
        pass
    
    def get_final_response(self):
        """Return the final LLM response"""
        return self.last_llm_response or self.final_response
    
    def on_agent_action(self, action: Any, **kwargs):
        """Display agent action"""
        if hasattr(action, 'tool'):
            print(f"\n🎯 Action: {action.tool}")
            print("Input:")
            print("-" * 50)
            print(action.tool_input)
            print("-" * 50)

    def on_tool_error(self, error: str, **kwargs):
        """Display tool errors"""
        print(f"\n❌ Error: {error}")