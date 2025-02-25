"""
Medical Graph Multi-Agent Manager

This module implements the supervisor/orchestrator for the specialized medical graph agents.
It coordinates agent execution, manages state transitions, and provides a unified interface 
for querying the medical graph database.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

# LangChain and LangGraph imports
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Import agent definitions
from react_agent import (
    aql_query_agent,
    graph_analysis_agent,
    patient_data_agent, 
    population_health_agent,
    AgentState
)

from tools import *
from callback import CustomConsoleCallbackHandler

# Define structured schema for routing decisions
class RouteDecision(BaseModel):
    """Schema for supervisor routing decisions"""
    next_action: str = Field(description="The next agent or action to route to")
    task_description: str = Field(description="Detailed description of the task for the agent")
    expected_output: str = Field(description="Expected outputs from the agent")
    validation_criteria: str = Field(description="Criteria to validate the agent's response")
    query_type: str = Field(description="Type of query: 'aql_query', 'graph_analysis', 'patient_data', 'population_health'")
    requires_multi_agent: bool = Field(description="Whether multiple agents are needed for this query")
    additional_agents: List[str] = Field(default_factory=list, description="Additional agents to consult if multi-agent is required")
    reasoning: str = Field(description="Reasoning behind this routing decision")

class MedicalGraphSupervisor:
    """
    Orchestrates the specialized medical graph agents, deciding which agent to route queries to
    and how to combine their outputs for comprehensive analysis.
    """
    
    def __init__(self):
        """Initialize the medical graph supervisor."""
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.callback_handler = CustomConsoleCallbackHandler()
        
        # Initialize the workflow
        self.workflow = self._create_workflow()
        
    def _create_workflow(self) -> StateGraph:
        """
        Create the workflow for orchestrating the specialized agents.
        
        The workflow is a directed graph that routes queries to the appropriate agent(s)
        based on query analysis, and combines their outputs.
        """
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("query_analyzer", self._query_analyzer)
        workflow.add_node("aql_query_agent", aql_query_agent)
        workflow.add_node("graph_analysis_agent", graph_analysis_agent)
        workflow.add_node("patient_data_agent", patient_data_agent)
        workflow.add_node("population_health_agent", population_health_agent)
        workflow.add_node("response_synthesizer", self._response_synthesizer)
        workflow.add_node("check_if_multi_agent_needed", self._check_if_multi_agent_needed)
        
        # Define the workflow
        workflow.add_edge(START, "query_analyzer")
        
        # Add conditional edges from query analyzer to agents
        workflow.add_conditional_edges(
            "query_analyzer",
            self._route_to_agents,
            {
                "aql_query_agent": "aql_query_agent",
                "graph_analysis_agent": "graph_analysis_agent",
                "patient_data_agent": "patient_data_agent",
                "population_health_agent": "population_health_agent",
                "all": "aql_query_agent",  # Default path for comprehensive analysis
            }
        )
        
        # Add edges from agents to check_if_multi_agent_needed
        workflow.add_edge("aql_query_agent", "check_if_multi_agent_needed")
        workflow.add_edge("graph_analysis_agent", "check_if_multi_agent_needed")
        workflow.add_edge("patient_data_agent", "check_if_multi_agent_needed")
        workflow.add_edge("population_health_agent", "check_if_multi_agent_needed")
        
        # When multi-agent path is needed
        workflow.add_conditional_edges(
            "check_if_multi_agent_needed",
            self._determine_next_step,
            {
                "end": "response_synthesizer",
                "graph_analysis": "graph_analysis_agent",
                "patient_data": "patient_data_agent",
                "population_health": "population_health_agent",
            }
        )
        
        # Final edge
        workflow.add_edge("response_synthesizer", END)
        
        # Compile workflow
        return workflow.compile()
    
    def _query_analyzer(self, state: AgentState) -> AgentState:
        """
        Analyze the user query to determine which agent(s) should handle it.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with routing decision
        """
        # Check if we should bypass the query analyzer
        if state["internal_state"].get("bypass_query_analyzer", False):
            state["callback"].write_agent_name("Query Analyzer 🔍 (Bypassed)")
            print("\n" + "="*50)
            print("�� QUERY ANALYZER (BYPASSED)")
            print(f"Using explicitly selected agent: {state['internal_state']['route_decision']}")
            print("="*50)
            return state
        
        state["callback"].write_agent_name("Query Analyzer 🔍")
        
        # Get the query from the state
        query = state["messages"][-1].content
        
        print("\n" + "="*50)
        print("🔍 QUERY ANALYZER")
        print(f"Current Input: {query}")
        print(f"Analysis Date: {state.get('current_date', 'Not specified')}")
        
        # Use the LLM to classify the query
        response = self.llm.with_structured_output(RouteDecision).invoke(f"""
        I need to determine which specialized medical graph agent(s) should handle this query:
        "{query}"
        
        Current Date: {state.get('current_date', 'Not specified')}
        
        The available agents are:
        1. AQL Query Agent: Converts natural language to AQL queries and executes them against the database. Best for data retrieval, aggregation, and statistical analysis. This agent should handle queries about anomaly patterns, claim behaviors, and frequency analysis.
        2. Graph Analysis Agent: Uses NetworkX algorithms for complex graph analysis of the medical network. Best for network structure, relationships, and connectivity analysis.
        3. Patient Data Agent: Specializes in analyzing individual patient data and histories. Best for patient-specific queries.
        4. Population Health Agent: Analyzes trends and patterns across the entire patient population. Best for population-level health trends and outcomes.
        
        Please analyze this query and determine which agent would be most appropriate.
        
        For simple queries that clearly match one agent's specialty, select just that agent.
        For complex queries that might benefit from multiple perspectives, you can select multiple agents.
        
        IMPORTANT GUIDELINES:
        - Queries about anomaly patterns, claim behaviors, or frequency analysis should be routed to the AQL Query Agent
        - Queries about network structure or relationships should be routed to the Graph Analysis Agent
        - Queries about specific patients should be routed to the Patient Data Agent
        - Queries about population-level trends should be routed to the Population Health Agent
        
        You must provide a structured response with these fields:
        - next_action: The name of the primary agent to handle this query
        - task_description: Detailed description of what the agent should do
        - expected_output: Clear description of what output is expected
        - validation_criteria: How to validate the agent's response
        - query_type: Classification of the query (e.g., "aql_query", "graph_analysis", etc.)
        - requires_multi_agent: Boolean indicating if multiple agents should be involved
        - additional_agents: List of additional agents to consult (if applicable)
        - reasoning: Explanation of your selection
        
        The agent names should be one of: "aql_query_agent", "graph_analysis_agent", "patient_data_agent", "population_health_agent", or "FINISH" for non-medical queries.
        """)
        
        try:
            # Store the classification in the state
            state["context"]["query_classification"] = response.dict()
            state["context"]["original_query"] = query
            state["internal_state"]["route_decision"] = response.next_action
            state["internal_state"]["requires_multi_agent"] = response.requires_multi_agent
            state["internal_state"]["additional_agents"] = response.additional_agents
            state["internal_state"]["current_task"] = {
                "description": response.task_description,
                "expected_output": response.expected_output,
                "validation_criteria": response.validation_criteria,
                "query_type": response.query_type
            }
            
            # Log the decision
            decision_explanation = f"""
            Query classification:
            - Primary agent: {response.next_action}
            - Reasoning: {response.reasoning}
            - Requires multi-agent: {response.requires_multi_agent}
            - Additional agents: {response.additional_agents}
            - Query type: {response.query_type}
            - Task: {response.task_description}
            """
            print(decision_explanation)
            
        except Exception as e:
            # Handle parsing errors
            print(f"Error parsing query analyzer response: {str(e)}")
            state["internal_state"]["route_decision"] = "all"  # Default to comprehensive analysis
            state["internal_state"]["requires_multi_agent"] = True
            state["internal_state"]["additional_agents"] = ["graph_analysis_agent", "patient_data_agent", "population_health_agent"]
        
        return state
    
    def _route_to_agents(self, state: AgentState) -> str:
        """
        Determine which agent to route to based on the query classification.
        
        Args:
            state: Current agent state
            
        Returns:
            Name of the agent to route to
        """
        return state["internal_state"]["route_decision"]
    
    def _check_if_multi_agent_needed(self, state: AgentState) -> AgentState:
        """
        Check if additional agents need to be consulted after the primary agent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        # Keep track of which agents have been executed
        if "executed_agents" not in state["internal_state"]:
            state["internal_state"]["executed_agents"] = []
        
        # Store the last response from each agent
        for agent_name in ["aql_query_agent", "graph_analysis_agent", "patient_data_agent", "population_health_agent"]:
            # Check if this agent was the last one to respond (based on the workflow)
            # Modified condition to check the current agent based on the route decision
            if (agent_name == state["internal_state"].get("route_decision") or 
                agent_name in state["internal_state"].get("additional_agents", [])) and \
                agent_name not in state["internal_state"]["executed_agents"]:
                
                # Store the agent name
                state["internal_state"]["executed_agents"].append(agent_name)
                
                # Store the agent's response
                if state["messages"] and isinstance(state["messages"][-1], AIMessage):
                    # Store with agent name to make it easier to reference
                    state["internal_state"][f"last_{agent_name}_response"] = state["messages"][-1].content
                    print(f"✓ Storing response from {agent_name}")
                    
                    # Also store as a named message for better tracking
                    agent_display_names = {
                        "aql_query_agent": "AQLQuery",
                        "graph_analysis_agent": "GraphAnalysis",
                        "patient_data_agent": "PatientData",
                        "population_health_agent": "PopulationHealth"
                    }
                    
                    # Replace the unnamed message with a named one
                    named_message = AIMessage(
                        content=state["messages"][-1].content,
                        name=agent_display_names.get(agent_name, agent_name)
                    )
                    state["messages"][-1] = named_message
                break
        
        return state
    
    def _determine_next_step(self, state: AgentState) -> str:
        """
        Determine the next step after an agent has been executed.
        
        Args:
            state: Current agent state
            
        Returns:
            Name of the next step
        """
        requires_multi_agent = state["internal_state"].get("requires_multi_agent", False)
        executed_agents = state["internal_state"].get("executed_agents", [])
        additional_agents = state["internal_state"].get("additional_agents", [])
        
        # Ensure the primary agent's response is captured
        primary_agent = state["internal_state"].get("route_decision")
        if primary_agent and primary_agent not in executed_agents and state["messages"] and isinstance(state["messages"][-1], AIMessage):
            # This is a safeguard to ensure the primary agent's response is captured
            state["internal_state"][f"last_{primary_agent}_response"] = state["messages"][-1].content
            executed_agents.append(primary_agent)
            print(f"✓ Captured response from primary agent: {primary_agent}")
        
        # If multi-agent analysis is not required or all agents have been executed
        if not requires_multi_agent or all(agent in executed_agents for agent in additional_agents):
            return "end"
        
        # Otherwise, select the next agent to execute
        for agent in additional_agents:
            if agent not in executed_agents:
                return agent
        
        # Default to end if no more agents to execute
        return "end"
    
    def _response_synthesizer(self, state: AgentState) -> AgentState:
        """
        Synthesize the responses from multiple agents into a coherent final response.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with synthesized response
        """
        state["callback"].write_agent_name("Response Synthesizer 📝")
        print("\nStarting Medical Analysis Synthesis...")
        
        # Get the original query
        original_query = state["context"].get("original_query", "")
        
        # Debug information about internal state
        print(f"Internal state keys: {list(state['internal_state'].keys())}")
        print(f"Executed agents: {state['internal_state'].get('executed_agents', [])}")
        
        # Collect responses from agents that have been executed
        agent_responses = {}
        for agent_name in ["aql_query_agent", "graph_analysis_agent", "patient_data_agent", "population_health_agent"]:
            response_key = f"last_{agent_name}_response"
            if response_key in state["internal_state"] and state["internal_state"][response_key]:
                agent_responses[agent_name] = state["internal_state"][response_key]
                print(f"✓ {agent_name} data extracted")
            else:
                print(f"✗ No data found for {agent_name}")
        
        # Extract named messages as fallback
        if not agent_responses:
            print("No agent responses found in internal state, trying to extract from named messages...")
            agent_display_names = {
                "AQLQuery": "aql_query_agent",
                "GraphAnalysis": "graph_analysis_agent",
                "PatientData": "patient_data_agent",
                "PopulationHealth": "population_health_agent"
            }
            
            for i, msg in enumerate(state["messages"]):
                print(f"Message {i}: type={type(msg)}, name={getattr(msg, 'name', 'unnamed')}")
                if isinstance(msg, AIMessage) and hasattr(msg, 'name') and msg.name in agent_display_names:
                    agent_name = agent_display_names[msg.name]
                    agent_responses[agent_name] = msg.content
                    print(f"✓ {agent_name} extracted from named message")
        
        # If no responses have been collected, provide a fallback
        if not agent_responses:
            print("WARNING: No agent responses could be collected. Using fallback response.")
            state["messages"].append(
                AIMessage(content="I'm sorry, but I wasn't able to generate a response to your query. "
                                   "Please try rephrasing or being more specific.",
                           name="FinalSynthesis")
            )
            return state
        
        # If only one agent has responded, use its response directly
        if len(agent_responses) == 1:
            agent_name = list(agent_responses.keys())[0]
            response = agent_responses[agent_name]
            print(f"Using direct response from {agent_name}")
            
            # Append the response to the message history
            state["messages"].append(AIMessage(content=response, name="FinalSynthesis"))
            return state
        
        # For multiple responses, synthesize them
        synthesis_prompt = f"""
        I need to synthesize responses from multiple specialized medical agents to answer this query:
        "{original_query}"
        
        The agents have provided the following insights:
        
        {self._format_agent_responses(agent_responses)}
        
        Please create a comprehensive, unified response that:
        1. Directly answers the original query
        2. Integrates the insights from all agents in a coherent way
        3. Highlights key findings and their implications
        4. Maintains medical accuracy and appropriate terminology
        5. Organizes information in a clear, structured format
        6. Provides a balanced view that considers all perspectives
        
        The response should be well-structured with appropriate headings and should not explicitly 
        mention which agent provided which insight.
        
        For visualizations and data tables:
        1. Clearly indicate where network graphs would be useful
        2. Format tabular data in markdown tables
        3. Suggest chart types for numerical data (bar, line, pie, etc.)
        
        Your final synthesis should be comprehensive yet focused on answering the specific query.
        """
        
        # Generate the synthesized response
        response = self.llm.invoke(synthesis_prompt)
        
        # Append the synthesized response to the message history with a name
        state["messages"].append(AIMessage(content=response.content, name="FinalSynthesis"))
        
        return state
    
    def _format_agent_responses(self, agent_responses: Dict[str, str]) -> str:
        """
        Format agent responses for the synthesis prompt.
        
        Args:
            agent_responses: Dictionary mapping agent names to their responses
            
        Returns:
            Formatted string of agent responses
        """
        formatted_responses = []
        
        agent_display_names = {
            "aql_query_agent": "AQL Database Analysis",
            "graph_analysis_agent": "Network Graph Analysis",
            "patient_data_agent": "Patient-Specific Analysis",
            "population_health_agent": "Population Health Analysis"
        }
        
        for agent_name, response in agent_responses.items():
            display_name = agent_display_names.get(agent_name, agent_name)
            formatted_responses.append(f"=== {display_name} ===\n{response}\n")
        
        return "\n".join(formatted_responses)
    
    def _extract_json(self, text: str) -> Dict:
        """
        Extract JSON from text, handling various formats.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON as a dictionary
        """
        import re
        import json
        
        # Try to find JSON pattern (handles both ```json and plain JSON)
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*\})"
        match = re.search(json_pattern, text)
        
        if match:
            json_str = match.group(1) or match.group(2)
            return json.loads(json_str)
        
        # If no match with the pattern, try parsing the whole text
        try:
            return json.loads(text)
        except:
            # Last resort: look for any {...} pattern
            curly_match = re.search(r"\{[\s\S]*\}", text)
            if curly_match:
                try:
                    return json.loads(curly_match.group(0))
                except:
                    raise ValueError("Could not extract valid JSON from response")
            else:
                raise ValueError("No JSON-like structure found in response")
    
    def process_query(self, query: str, agent_type: str = None) -> str:
        """
        Process a user query through the multi-agent workflow or directly with a specific agent.
        
        Args:
            query: User's natural language query
            agent_type: Optional specific agent to use, bypassing the workflow entirely
            
        Returns:
            Final response
        """
        # Create initial state
        state = AgentState(
            messages=[HumanMessage(content=query)],
            current_date=datetime.now().isoformat(),
            context={"original_query": query},
            internal_state={
                "aql_query_agent_internal_state": {
                    "agent_executor_tools": {},
                    "full_response": {},
                    "all_tools_eval": {"passed": [], "stats": []},
                    "topic_adherence_eval": {"passed": [], "reason": []}
                },
                "graph_analysis_agent_internal_state": {
                    "agent_executor_tools": {},
                    "full_response": {},
                    "all_tools_eval": {"passed": [], "stats": []},
                    "topic_adherence_eval": {"passed": [], "reason": []}
                },
                "patient_data_agent_internal_state": {
                    "agent_executor_tools": {},
                    "full_response": {},
                    "all_tools_eval": {"passed": [], "stats": []},
                    "topic_adherence_eval": {"passed": [], "reason": []}
                },
                "population_health_agent_internal_state": {
                    "agent_executor_tools": {},
                    "full_response": {},
                    "all_tools_eval": {"passed": [], "stats": []},
                    "topic_adherence_eval": {"passed": [], "reason": []}
                }
            },
            callback=self.callback_handler
        )
        
        try:
            # Direct agent invocation if specified
            if agent_type in ["aql_query_agent", "graph_analysis_agent", 
                              "patient_data_agent", "population_health_agent"]:
                
                # Log the direct invocation
                print(f"\n{'='*50}\nDirect invocation of {agent_type}\n{'='*50}")
                state["callback"].write_agent_name(f"{agent_type} (Direct)")
                
                # Map of agent types to their functions
                agent_functions = {
                    "aql_query_agent": aql_query_agent,
                    "graph_analysis_agent": graph_analysis_agent,
                    "patient_data_agent": patient_data_agent,
                    "population_health_agent": population_health_agent
                }
                
                # Get the agent function
                agent_function = agent_functions.get(agent_type)
                
                # Add task information to state for the agent
                state["internal_state"]["current_task"] = {
                    "description": f"Process the query using the {agent_type}",
                    "expected_output": "A comprehensive response to the user query",
                    "validation_criteria": "The response should directly address the user's query",
                    "query_type": agent_type.replace("_agent", "")
                }
                
                # Execute the agent directly
                agent_state = agent_function(state)
                
                # Debug: Print all messages in the state
                print(f"Agent returned {len(agent_state['messages'])} messages")
                for i, msg in enumerate(agent_state['messages']):
                    print(f"Message {i}: type={type(msg).__name__}, name={getattr(msg, 'name', 'unnamed')}")
                    if hasattr(msg, 'content'):
                        print(f"  Content preview: {msg.content[:50]}...")
                
                # Return the agent's response - improved handling of different message types
                if agent_state["messages"]:
                    # Find the most recent AI message in the messages list
                    for message in reversed(agent_state["messages"]):
                        if isinstance(message, AIMessage):
                            print(f"Found AI message with content: {message.content[:100]}...")
                            return message.content
                    
                    # If no AIMessage found, try to find any message with content
                    for message in reversed(agent_state["messages"]):
                        if hasattr(message, 'content') and message.content:
                            print(f"Found message with content of type {type(message).__name__}")
                            return message.content
                    
                    # Last resort: return the string representation of the last message
                    last_message = agent_state["messages"][-1]
                    print(f"No suitable message found, using last message of type: {type(last_message).__name__}")
                    if hasattr(last_message, 'content'):
                        return last_message.content
                    else:
                        return str(last_message)
                else:
                    print("No messages found in agent_state")
                    return "No response generated from the agent. Please try a different query or agent."
            
            # Use the full workflow for automatic agent selection
            else:
                final_state = self.workflow.invoke(state)
                
                # Debug: Print all messages in the final state
                print(f"Workflow returned {len(final_state['messages'])} messages")
                for i, msg in enumerate(final_state['messages']):
                    print(f"Message {i}: type={type(msg).__name__}, name={getattr(msg, 'name', 'unnamed')}")
                    if hasattr(msg, 'content'):
                        print(f"  Content preview: {msg.content[:50]}...")
                
                # Get the final response
                if final_state["messages"]:
                    # Find the most recent AI message
                    for message in reversed(final_state["messages"]):
                        if isinstance(message, AIMessage):
                            print(f"Found AI message with content: {message.content[:100]}...")
                            return message.content
                    
                    # If no AIMessage found, try to find any message with content
                    for message in reversed(final_state["messages"]):
                        if hasattr(message, 'content') and message.content:
                            print(f"Found message with content of type {type(message).__name__}")
                            return message.content
                    
                    # Last resort: return the string representation of the last message
                    last_message = final_state["messages"][-1]
                    print(f"No suitable message found, using last message of type: {type(last_message).__name__}")
                    if hasattr(last_message, 'content'):
                        return last_message.content
                    else:
                        return str(last_message)
                else:
                    return "No response generated"
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            
            # Try to provide a more helpful error message
            if "population_health_agent" in str(e):
                return "I encountered an error while analyzing population health data. This might be because your query involves anomaly patterns or claim behaviors, which should be handled by the AQL Query Agent instead. Please try again or rephrase your query."
            elif "graph_analysis_agent" in str(e):
                return "I encountered an error while analyzing graph relationships. This might be because your query involves data retrieval or statistical analysis, which should be handled by the AQL Query Agent instead. Please try again or rephrase your query."
            elif "patient_data_agent" in str(e):
                return "I encountered an error while analyzing patient data. This might be because your query involves population-level trends, which should be handled by the Population Health Agent instead. Please try again or rephrase your query."
            elif "aql_query_agent" in str(e):
                return "I encountered an error while querying the database. Please try again with a more specific query or check if the database is properly connected."
            else:
                return error_msg
    
    def interactive_session(self):
        """Run an interactive session for querying the medical graph."""
        print("\n" + "="*80)
        print("Medical Graph Multi-Agent System Interactive Session")
        print("="*80)
        print("Type 'exit' or 'quit' to end the session.")
        print("Type 'help' for example queries.")
        print("="*80)
        
        example_queries = [
            "What are the most common conditions in the database?",
            "Find all patients who have been diagnosed with hypertension and are on beta blockers",
            "Which providers have the highest centrality in the referral network?",
            "Give me a complete medical history for patient with ID ''",
            "Analyze treatment effectiveness for diabetes patients",
            "Find communities of providers who frequently collaborate on patient care",
            "What medication patterns are most common for heart disease patients?",
            "Map the relationship between medication costs and treatment effectiveness for asthma"
        ]
        
        while True:
            print("\n")
            query = input("Enter your query: ")
            
            if query.lower() in ['exit', 'quit']:
                print("Ending session. Goodbye!")
                break
                
            if query.lower() == 'help':
                print("\nExample queries:")
                for i, q in enumerate(example_queries):
                    print(f"{i+1}. {q}")
                continue
            
            print("\nProcessing query...\n")
            
            try:
                response = self.process_query(query)
                print("\n" + "="*80)
                print("FINAL RESPONSE:")
                print("="*80)
                print(response)
                print("="*80)
            except Exception as e:
                print(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    supervisor = MedicalGraphSupervisor()
    
    # Generate and save workflow visualization
    try:
        from langchain_core.runnables.graph import MermaidDrawMethod
        
        # Save the workflow diagram as a PNG file
        png_data = supervisor.workflow.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )
        
        # Write the PNG data to a file
        with open("workflow_diagram.png", "wb") as f:
            f.write(png_data)
        print("Workflow diagram saved as 'workflow_diagram.png'")
    except Exception as e:
        print(f"Could not generate workflow diagram: {e}")
    
    supervisor.interactive_session() 