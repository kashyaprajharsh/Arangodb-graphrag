"""
Configuration settings for the ArangoDB-CuGraph application.
This module loads environment variables from a .env file and provides them as settings.
"""

import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

# ArangoDB Settings
ARANGO_HOST = os.getenv("ARANGO_HOST", "https://86525f194b77.arangodb.cloud:8529")
ARANGO_USERNAME = os.getenv("ARANGO_USERNAME", "root")
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD", "")
ARANGO_DB = os.getenv("ARANGO_DB", "_system")
ARANGO_VERIFY = os.getenv("ARANGO_VERIFY", "True").lower() in ("true", "1", "t")

# OpenAI API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Graph Cache Settings
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # Default: 1 hour

# Graph Settings
GRAPH_NAME = os.getenv("GRAPH_NAME", "SYNTHEA_P100")
DEFAULT_NODE_TYPE = os.getenv("DEFAULT_NODE_TYPE", "allergies")

# Validate required settings
if not ARANGO_PASSWORD:
    logger.warning("ARANGO_PASSWORD is not set in environment variables")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set in environment variables")

# Log configuration status
logger.info(f"Configuration loaded from environment variables")
logger.info(f"Using ArangoDB host: {ARANGO_HOST}")
logger.info(f"Using graph: {GRAPH_NAME}") 