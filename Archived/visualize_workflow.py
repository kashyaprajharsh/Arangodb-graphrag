"""
Medical Graph Workflow Visualization

This script demonstrates how to visualize the medical graph multi-agent workflow
using Mermaid diagrams in a Jupyter notebook.
"""

from multi_agent_manager import MedicalGraphSupervisor

def visualize_workflow():
    """
    Create and visualize the medical graph workflow.
    
    This function initializes the MedicalGraphSupervisor and displays
    the workflow graph using Mermaid.
    """
    print("Initializing Medical Graph Supervisor...")
    supervisor = MedicalGraphSupervisor()
    
    print("Generating workflow visualization...")
    supervisor.display_workflow_graph()
    
    print("\nTo save the visualization to a file, use:")
    print("supervisor.save_workflow_graph('output_path.png')")

if __name__ == "__main__":
    print("This script is designed to be run in a Jupyter notebook.")
    print("To use it in a notebook, run:")
    print("\n```python")
    print("from visualize_workflow import visualize_workflow")
    print("visualize_workflow()")
    print("```\n")
    print("Alternatively, you can run the following code directly in a notebook cell:")
    print("\n```python")
    print("from multi_agent_manager import MedicalGraphSupervisor")
    print("supervisor = MedicalGraphSupervisor()")
    print("supervisor.display_workflow_graph()")
    print("```") 