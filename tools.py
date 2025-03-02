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
import pandas as pd
import matplotlib.pyplot as plt
from typing import Annotated, Dict, TypedDict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel

# LangChain and LangGraph imports

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import re

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_community.graphs import ArangoGraph
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langchain_core.tools import tool

from typing import Annotated, Dict, TypedDict, Any ,Optional
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.chains import ArangoGraphQAChain
from pydantic import BaseModel
import json
import re
import time
import asyncio

# Import settings and graph cache module
from settings import OPENAI_API_KEY
from graph_cache import get_db, get_graph, get_arango_graph, clear_cache

# Configuration
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Use cached database connection
db = get_db()

# Use cached graph
G_adb = get_graph()

# Use cached ArangoGraph
arango_graph = get_arango_graph()

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

@tool
def text_to_aql_to_text(query: str):
    """This tool translates natural language queries about the SYNTHEA_P100 graph into AQL,
    executes the query, and returns the results in natural language.
    """
    try:
        # Get actual schema details from database
        schema_details = get_schema_details()

        # Create custom prompts with our schema information
        from langchain.prompts import PromptTemplate

        aql_generation_template = """You are an expert at converting natural language questions into AQL (ArangoDB Query Language) queries.

        GRAPH DETAILS:
        Name: 'SYNTHEA_P100'
        Nodes: 145,514
        Edges: 311,701
        Default node type: 'allergies'
        schema details
        database structure:

      Vertex Collections: ['providers', 'encounters', 'organizations', 'medications', 'patients', 'conditions', 'observations', 'careplans', 'supplies', 'payers', 'procedures', 'imaging_studies', 'devices', 'immunizations', 'allergies']
      Edge Collections: ['patients_to_medications', 'encounters_to_immunizations', 'patients_to_conditions', 'encounters_to_imaging_studies', 'patients_to_immunizations', 'patients_to_allergies', 'encounters_to_medications', 'encounters_to_allergies', 'encounters_to_devices', 'patients_to_procedures', 'patients_to_devices', 'payers_to_encounters', 'encounters_to_careplans', 'encounters_to_conditions', 'payers_to_medications', 'patients_to_careplans', 'providers_to_encounters', 'organizations_to_providers', 'encounters_to_procedures', 'patients_to_imaging_studies', 'encounters_to_observations', 'organizations_to_encounters', 'patients_to_encounters', 'encounters_to_supplies', 'patients_to_supplies', 'patients_to_observations']



        DATABASE STRUCTURE:
        1. Vertex Collections (Nodes):
        {adb_schema}

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
        {adb_schema}

        Original question: {user_input}
        AQL query used: {aql_query}
        Query result: {aql_result}

        Please provide a clear, natural language response to the original question based on the query results.
        Human: What does this result mean?
        Assistant: """

        # Initialize LLM
        llm = ChatOpenAI(
            temperature=0.2,
            model_name="gpt-4o"
        )

        # Initialize chain with default prompts
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
          aql
          {result['aql_query']}

          Query Results:
          json
          {json.dumps(result['aql_result'], indent=2)}
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

def test_queries():
    """Test the query tool with example queries"""
    test_queries = [
        "What collections are available in the database?",
        # "Show me a sample of patient records",
        # "Count the total number of records in each collection",
        # "What are the most common conditions in the database give names of that?",
        # "Find all allergies for patients over 65 years old",
        # "what is pagerank of the patient 7c2e78bd-52cf-1fce-acc3-0ddd93104abe"
        # "Find top 5 providers with unusually high claims for expensive procedures and then detect any anomalies or fraud claims",
    #     "What are the most common anomaly patterns in claim behaviors for high-cost procedures?",
        # "Find all the patients with having condition Stress (finding) along with thier age name and id"
       
     ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        result = text_to_aql_to_text(query)
        print(result)


if __name__ == "__main__":
    try:

        print("\nExploring database structure:")
        schema_details = get_schema_details()
        print(f"\nVertex Collections: {schema_details['vertex_collections']}")
        print(f"Edge Collections: {schema_details['edge_collections']}")

        # Run test queries
        print("\nRunning test queries...")
        test_queries()

    except Exception as e:
        print(f"Application error: {str(e)}")