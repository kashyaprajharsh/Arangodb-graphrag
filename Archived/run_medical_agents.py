#!/usr/bin/env python3
"""
Medical Graph Multi-Agent System Runner

This script provides a command-line interface to run the Medical Graph Multi-Agent System
in different modes: interactive session, single query, or batch queries.
"""

import argparse
import json
import sys
from typing import List, Optional

# Import our multi-agent system
from multi_agent_manager import MedicalGraphSupervisor

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Medical Graph Multi-Agent System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for different modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Run in interactive mode")
    
    # Single query mode
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("query_text", type=str, help="Query to run")
    
    # Batch mode
    batch_parser = subparsers.add_parser("batch", help="Run multiple queries from a file")
    batch_parser.add_argument("query_file", type=str, help="File containing queries (one per line)")
    batch_parser.add_argument("--output", type=str, help="Output file for results (JSON format)")
    
    # Test mode
    test_parser = subparsers.add_parser("test", help="Run predefined test queries")
    test_parser.add_argument("--agent", type=str, choices=["aql", "graph", "patient", "population", "all"], 
                           default="all", help="Specific agent to test")
    
    return parser.parse_args()

def run_single_query(supervisor: MedicalGraphSupervisor, query: str) -> str:
    """Run a single query and return the result"""
    print(f"\nProcessing query: {query}")
    print("-" * 80)
    
    try:
        result = supervisor.process_query(query)
        return result
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        return error_msg

def run_batch_queries(supervisor: MedicalGraphSupervisor, query_file: str, output_file: Optional[str] = None) -> dict:
    """Run multiple queries from a file and optionally save results"""
    results = {}
    
    try:
        with open(query_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        print(f"Running {len(queries)} queries from {query_file}...")
        
        for i, query in enumerate(queries):
            print(f"\nQuery {i+1}/{len(queries)}: {query}")
            print("-" * 80)
            
            result = supervisor.process_query(query)
            results[query] = result
            
            print(f"\nResult: {result[:100]}..." if len(result) > 100 else f"\nResult: {result}")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {output_file}")
            
        return results
        
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return {"error": str(e)}

def run_tests(supervisor: MedicalGraphSupervisor, agent_type: str = "all") -> dict:
    """Run predefined test queries for specific agent types"""
    test_queries = {
        "aql": [
            "Find all patients who have been diagnosed with hypertension and are on beta blockers",
            "What are the most common medications prescribed for diabetes patients?"
        ],
        "graph": [
            "Which providers have the highest centrality in the referral network?",
            "Find communities of providers who frequently collaborate on patient care"
        ],
        "patient": [
            "Give me a complete medical history for patient with ID 'f4640c72-6ea6-db89-e996-91c90af95544'",
            "Find patients similar to patient 'f4640c72-6ea6-db89-e996-91c90af95544' based on their conditions"
        ],
        "population": [
            "What are the most common conditions in the database and their frequencies?",
            "Analyze the typical treatment pathway for patients with COPD"
        ]
    }
    
    # Map command-line arguments to agent types
    agent_map = {
        "aql": "aql",
        "graph": "graph",
        "patient": "patient",
        "population": "population",
        "all": "all"
    }
    
    # Select which queries to run
    if agent_type == "all":
        queries_to_run = []
        for agent in test_queries:
            queries_to_run.extend(test_queries[agent])
    else:
        agent_key = agent_map.get(agent_type)
        queries_to_run = test_queries.get(agent_key, [])
    
    # Run the selected queries
    results = {}
    for i, query in enumerate(queries_to_run):
        print(f"\nTest {i+1}/{len(queries_to_run)}: {query}")
        print("-" * 80)
        
        result = supervisor.process_query(query)
        results[query] = result
        
        print("\n" + "="*80)
        print(f"RESULT FOR: {query}")
        print("="*80)
        print(result)
        print("="*80)
    
    return results

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create the supervisor
    print("Initializing Medical Graph Multi-Agent System...")
    supervisor = MedicalGraphSupervisor()
    
    # Run in the specified mode
    if args.mode == "interactive":
        supervisor.interactive_session()
    elif args.mode == "query":
        result = run_single_query(supervisor, args.query_text)
        print("\n" + "="*80)
        print("RESULT:")
        print("="*80)
        print(result)
        print("="*80)
    elif args.mode == "batch":
        run_batch_queries(supervisor, args.query_file, args.output)
    elif args.mode == "test":
        run_tests(supervisor, args.agent)
    else:
        # Default to interactive mode if no mode specified
        print("No mode specified, defaulting to interactive session.")
        supervisor.interactive_session()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1) 