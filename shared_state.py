import streamlit as st
from mem0 import Memory

def get_memory():
    """Get the shared memory instance from Streamlit session state"""
    if "memory" not in st.session_state:
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                }
            },
            "custom_prompt": """
            Please extract entities containing healthcare-related information such as patient details, diagnoses, treatments, medications, medical encounters, and any relevant healthcare events. Additionally, please handle analytical queries regarding healthcare statistics, such as averages, percentages, and cost breakdowns. Each input should be processed to return structured data that can be stored in a system or used to answer specific queries.
            """
        }
        st.session_state.memory = Memory.from_config(config_dict=config)
    return st.session_state.memory 