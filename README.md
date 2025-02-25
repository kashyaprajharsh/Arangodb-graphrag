# Medical Graph Multi-Agent System

A powerful multi-agent architecture for analyzing medical graph data stored in ArangoDB. This system uses specialized agents to perform different types of analysis on the SYNTHEA_P100 healthcare dataset.

## Architecture

The system consists of the following components:

1. **Specialized Agents**:
   - **AQL Query Agent**: Translates natural language to AQL queries and executes them
   - **Graph Analysis Agent**: Applies NetworkX algorithms to the medical graph
   - **Patient Data Agent**: Analyzes individual patient histories and data
   - **Population Health Agent**: Performs population-level health analysis

2. **Orchestration**:
   - The `MedicalGraphSupervisor` coordinates the agents
   - Queries are analyzed to determine which agent(s) should handle them
   - For complex queries, multiple agents can work together
   - Results are synthesized into coherent responses

3. **User Interface**:
   - Professional Streamlit web interface for interacting with the system
   - Interactive query interface with example queries
   - Advanced data visualizations (networks, charts, tables)
   - Comprehensive documentation and agent overviews

## Requirements

- Python 3.8+
- ArangoDB
- NetworkX
- LangGraph/LangChain
- OpenAI API key
- Streamlit (for UI)
- Plotly (for visualizations)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd arangodb-cugraph
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your ArangoDB connection and API keys in `medical_graph_agents.py`

## Usage

The system can be run in several modes:

### Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

This launches a professional web interface where you can:
- Submit queries through an interactive interface
- Select specific agents for analysis
- View visualizations of results
- Access documentation and examples

### Command Line Interface

#### Interactive Mode

```bash
python run_medical_agents.py interactive
```

This starts an interactive session where you can enter queries and receive responses.

#### Single Query Mode

```bash
python run_medical_agents.py query "Which providers have the highest centrality in the referral network?"
```

This runs a single query and displays the result.

#### Batch Mode

```bash
python run_medical_agents.py batch queries.txt --output results.json
```

This runs multiple queries from a file (one per line) and optionally saves the results.

#### Test Mode

```bash
python run_medical_agents.py test --agent graph
```

This runs predefined test queries for a specific agent type.

## Example Queries

- "What are the most common conditions in the database?"
- "Find all patients who have been diagnosed with hypertension and are on beta blockers"
- "Which providers have the highest centrality in the referral network?"
- "Give me a complete medical history for patient with ID 'f4640c72-6ea6-db89-e996-91c90af95544'"
- "Analyze treatment effectiveness for diabetes patients"
- "Find communities of providers who frequently collaborate on patient care"
- "What medication patterns are most common for heart disease patients?"
- "Map the relationship between medication costs and treatment effectiveness for asthma"

## File Structure

- `medical_graph_agents.py`: Defines the specialized agents and tools
- `multi_agent_manager.py`: Implements the supervisor and orchestration
- `run_medical_agents.py`: Command-line interface for running the system
- `streamlit_app.py`: Web-based user interface built with Streamlit
- `requirements.txt`: Project dependencies
- `agent.py`: Original code (for reference)

## Extending the System

To add new capabilities:

1. Create new tools in `medical_graph_agents.py`
2. Add new agent types if needed
3. Update the supervisor in `multi_agent_manager.py` to route to the new agents
4. Enhance the UI in `streamlit_app.py` to showcase the new capabilities

## License

[Specify your license here]

## Acknowledgments

This system uses the SYNTHEA_P100 dataset, a synthetic healthcare dataset generated using Synthea. 