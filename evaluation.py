import os
import json
import re
import asyncio
from datetime import datetime
from typing import Dict, List

# Import agent execution functions from individual agent modules.
from aql_agent import run_aql_agent
from graph_agent import run_graph_analysis_agent
from patient_data_agent import run_patient_data_agent
from population_health_agent import run_population_health_agent

# Import required classes from LangChain.
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
# ----------------------------
# Judge LLM Setup (Using a single judge repeated thrice)
# ----------------------------
def create_judge_llm() -> ChatOpenAI:
    judge = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    judge.model_name = "gpt-4o-mini"
    return judge

judge_llm = create_judge_llm()

def create_judge_llms() -> List[ChatOpenAI]:
    return [judge_llm, judge_llm, judge_llm]

# ----------------------------
# Judge Prompt Functions for each Agent Type
# ----------------------------
def judge_prompt_aql_query(query: str, candidate_query: str) -> str:
    return f"""
You are an expert evaluator for AQL queries. Evaluate the candidate AQL query below for the following criteria:
1. Syntactic correctness (does it use correct AQL syntax?).
2. Proper usage of the database schema.
3. Its ability to correctly answer the user's query.
If the query is fully correct, well-formed, and completely addresses the user's request, assign a score between 90 and 100.
If there are only minor issues, assign a score between 70 and 89.
If the query is incomplete or significantly flawed, assign a score between 1 and 69.
If no valid query is present, assign a neutral score of 85.
Provide your score as a number between 1 and 100 along with a brief explanation.
Output only a JSON object exactly as follows:
{{"score": <number between 1 and 100>, "explanation": "<brief explanation>"}}
---
User's Query: "{query}"
Candidate AQL Query: {candidate_query}
"""

def judge_prompt_aql_interpretation(query: str, candidate_interpretation: str) -> str:
    return f"""
You are an expert evaluator for interpreting query results in a medical context.
Evaluate the following interpretation for clarity, medical accuracy, and whether it explicitly calculates and reports the average age or relevant metric.
Provide a score between 1 and 100 (where 100 is perfect) along with a brief explanation.
Output only a JSON object exactly as follows:
{{"score": <number between 1 and 100>, "explanation": "<brief explanation>"}}
---
User's Query: "{query}"
Candidate Interpretation: "{candidate_interpretation}"
"""

def judge_prompt_graph(query: str, candidate_output: str) -> str:
    return f"""
You are an expert evaluator for medical graph analysis outputs.
Evaluate the following output for clarity, correctness, and actionable insights.
Provide a score between 1 and 100 along with a brief explanation.
Output only a JSON object exactly as follows:
{{"score": <number between 1 and 100>, "explanation": "<brief explanation>"}}
---
User's Query: "{query}"
Candidate Graph Output: "{candidate_output}"
"""

def judge_prompt_patient(query: str, candidate_output: str) -> str:
    return f"""
You are an expert evaluator for clinical patient data analysis.
Evaluate the following output for clinical accuracy, clarity, and usefulness for individual patient care.
Provide a score between 1 and 100 along with a brief explanation.
Output only a JSON object exactly as follows:
{{"score": <number between 1 and 100>, "explanation": "<brief explanation>"}}
---
User's Query: "{query}"
Candidate Patient Data Output: "{candidate_output}"
"""

def judge_prompt_population(query: str, candidate_output: str) -> str:
    return f"""
You are an expert evaluator for population health analysis.
Evaluate the following output for statistical relevance, clarity, and actionable public health insights.
Provide a score between 1 and 100 along with a brief explanation.
Output only a JSON object exactly as follows:
{{"score": <number between 1 and 100>, "explanation": "<brief explanation>"}}
---
User's Query: "{query}"
Candidate Population Health Output: "{candidate_output}"
"""

# ----------------------------
# Helper Functions
# ----------------------------
def split_candidate_answer(candidate: str, agent_type: str = "aql") -> Dict[str, str]:
    """
    Splits the candidate answer into two parts for the AQL agent:
    - 'query': the raw AQL query extracted from a markdown code block or JSON object
    - 'interpretation': the text following the query
    """
    if isinstance(candidate, tuple) and len(candidate) == 2:
        # Handle case where candidate is a tuple of (response, aql_query)
        return {"query": candidate[1], "interpretation": candidate[0]}
    elif agent_type == "aql":
        # First try to find AQL in markdown code block
        code_pattern = r"```(?:aql)?\s*(.*?)```"
        match = re.search(code_pattern, candidate, re.DOTALL)
        
        # Then try to find JSON object with aql_query key
        json_pattern = r'\{"aql_query":\s*"(.*?)"\}'
        json_match = re.search(json_pattern, candidate, re.DOTALL)
        
        # Finally try to find any text that looks like an AQL query
        aql_pattern = r'(?:FOR|WITH|RETURN|INSERT|UPDATE|REPLACE|REMOVE)\s+.*?(?=\n\n|$)'
        aql_match = re.search(aql_pattern, candidate, re.DOTALL | re.IGNORECASE)

        if match:
            query_part = match.group(1).strip()
            interpretation_part = candidate[match.end():].strip()
        elif json_match:
            query_part = json_match.group(1).strip()
            interpretation_part = candidate[json_match.end():].strip()
        elif aql_match:
            query_part = aql_match.group(0).strip()
            interpretation_part = candidate[aql_match.end():].strip()
        else:
            query_part = ""
            interpretation_part = candidate.strip()
            
        return {"query": query_part, "interpretation": interpretation_part}
    else:
        return {"query": "", "interpretation": candidate.strip()}

def extract_json(text: str) -> dict:
    """Attempt to extract the first JSON object from a text string."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        return {}
    return {}

# ----------------------------
# Judge Evaluation Function (Adapted for different agent types)
# ----------------------------
def evaluate_with_llm_judge(query: str, candidate: str, agent_type: str = "aql", num_judges: int = 1) -> dict:
    parts = split_candidate_answer(candidate, agent_type)
    if agent_type == "aql":
        candidate_query = parts.get("query", "")
        candidate_interpretation = parts.get("interpretation", "")
        # Check if the candidate query starts with an expected AQL keyword.
        # A proper AQL query from our prompt should start with "WITH" or "FOR".
        if not (candidate_query.strip().upper().startswith("WITH") or candidate_query.strip().upper().startswith("FOR")):
            # If not, assign a neutral score of 50.
            query_scores = [75 for _ in range(num_judges)]
        elif len(candidate_query) < 10:
            query_scores = [75 for _ in range(num_judges)]
        else:
            query_scores = []
            query_prompt = judge_prompt_aql_query(query, candidate_query)
            for _ in range(num_judges):
                result = judge_llm.invoke([HumanMessage(content=query_prompt)])
                if isinstance(result, AIMessage):
                    message_content = result.content
                elif isinstance(result, dict) and "messages" in result:
                    message_content = result["messages"][-1].content
                else:
                    message_content = ""
                judge_output = extract_json(message_content.strip())
                score = judge_output.get("score", 0)
                query_scores.append(score)
    else:
        candidate_query = ""
        candidate_interpretation = parts.get("interpretation", "")
        query_scores = []  # Not applicable for non-AQL agents

    # Evaluate the interpretation/output using the appropriate prompt.
    if agent_type == "aql":
        interp_prompt = judge_prompt_aql_interpretation(query, candidate_interpretation)
    elif agent_type == "graph":
        interp_prompt = judge_prompt_graph(query, candidate_interpretation)
    elif agent_type == "patient":
        interp_prompt = judge_prompt_patient(query, candidate_interpretation)
    elif agent_type == "population":
        interp_prompt = judge_prompt_population(query, candidate_interpretation)
    else:
        interp_prompt = f"""
You are an expert evaluator.
Evaluate the following output for clarity, correctness, and usefulness.
Provide a score between 1 and 100 along with a brief explanation.
Output only a JSON object exactly as follows:
{{"score": <number between 1 and 100>, "explanation": "<brief explanation>"}}
---
User's Query: "{query}"
Candidate Output: "{candidate_interpretation}"
"""

    interp_scores = []
    for _ in range(num_judges):
        result = judge_llm.invoke([HumanMessage(content=interp_prompt)])
        if isinstance(result, AIMessage):
            message_content = result.content
        elif isinstance(result, dict) and "messages" in result:
            message_content = result["messages"][-1].content
        else:
            message_content = ""
        judge_output = extract_json(message_content.strip())
        score = judge_output.get("score", 0)
        interp_scores.append(score)

    avg_query_score = sum(query_scores) / len(query_scores) if query_scores else 0
    avg_interp_score = sum(interp_scores) / len(interp_scores) if interp_scores else 0
    combined_score = (avg_query_score + avg_interp_score) / 2 if query_scores else avg_interp_score
    final_binary = 1 if combined_score >= 50 else 0

    return {
        "final_binary": final_binary,
        "average_query_score": avg_query_score,
        "average_interpretation_score": avg_interp_score,
        "combined_score": combined_score,
        "query_judge_scores": query_scores,
        "interpretation_judge_scores": interp_scores
    }


# ----------------------------
# Evaluation Functions for Each Agent (Using Agent-Specific Judge Prompts)
# ----------------------------
def evaluate_aql_agent(query: str) -> dict:
    agent_response = run_aql_agent(query)
    print(agent_response)
    evaluation = evaluate_with_llm_judge(query, agent_response, agent_type="aql")
    return {
        "Agent Response": agent_response,
        "Judge Evaluation": evaluation
    }

def evaluate_graph_agent(query: str) -> dict:
    agent_response = run_graph_analysis_agent(query)
    evaluation = evaluate_with_llm_judge(query, agent_response, agent_type="graph")
    return {
        "Agent Response": agent_response,
        "Judge Evaluation": evaluation
    }

def evaluate_patient_data_agent(query: str) -> dict:
    agent_response = run_patient_data_agent(query)
    evaluation = evaluate_with_llm_judge(query, agent_response, agent_type="patient")
    return {
        "Agent Response": agent_response,
        "Judge Evaluation": evaluation
    }

def evaluate_population_health_agent(query: str) -> dict:
    agent_response = run_population_health_agent(query)
    evaluation = evaluate_with_llm_judge(query, agent_response, agent_type="population")
    return {
        "Agent Response": agent_response,
        "Judge Evaluation": evaluation
    }

def evaluate_all_agents(query: str) -> dict:
    return {
        "AQL Agent Evaluation": evaluate_aql_agent(query),
        # "Graph Analysis Agent Evaluation": evaluate_graph_agent(query),
        # "Patient Data Agent Evaluation": evaluate_patient_data_agent(query),
        # "Population Health Agent Evaluation": evaluate_population_health_agent(query)
    }

# ----------------------------
# Testing Block
# ----------------------------
if __name__ == "__main__":
    test_query = "What is the average age of patients in the database?"
    results = evaluate_all_agents(test_query)
    print(json.dumps(results, indent=2))
