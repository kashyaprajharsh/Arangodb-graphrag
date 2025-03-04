import os
import networkx as nx
import nx_arangodb as nxadb
from arango import ArangoClient
from langchain_community.graphs import ArangoGraph
import time
import logging
from settings import ARANGO_HOST, ARANGO_USERNAME, ARANGO_PASSWORD, ARANGO_DB, ARANGO_VERIFY, CACHE_TTL, GRAPH_NAME, DEFAULT_NODE_TYPE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

# Global variables to store cached objects
_db = None
_G_adb = None
_arango_graph = None
_last_loaded = None
_cache_ttl = CACHE_TTL  # Cache time-to-live in seconds (from settings)

def get_db():
    """Get or initialize the ArangoDB connection"""
    global _db
    
    if _db is None:
        logger.info("Initializing ArangoDB connection...")
        _db = ArangoClient(hosts=ARANGO_HOST).db(
            ARANGO_DB, username=ARANGO_USERNAME, password=ARANGO_PASSWORD, verify=ARANGO_VERIFY
        )
        logger.info("ArangoDB connection established")
    
    return _db

def get_graph():
    """Get or initialize the NetworkX graph with caching"""
    global _G_adb, _last_loaded
    
    current_time = time.time()
    
    # Initialize if not exists or if cache has expired
    if _G_adb is None or _last_loaded is None or (current_time - _last_loaded > _cache_ttl):
        logger.info("Initializing or refreshing graph cache...")
        
        db = get_db()
        
        # Try to enable GPU acceleration if available
        try:
            nx.config.backends.arangodb.use_gpu = True
            logger.info("GPU acceleration enabled for graph operations")
        except ImportError:
            nx.config.backends.arangodb.use_gpu = False
            logger.info("GPU acceleration disabled, using CPU for graph operations")
        
        start_time = time.time()
        _G_adb = nxadb.Graph(name=GRAPH_NAME, db=db)
        _last_loaded = current_time
        
        # Check if graph exists
        if _G_adb:
            logger.info(f"Graph '{GRAPH_NAME}' loaded successfully in {time.time() - start_time:.2f} seconds")
            # Set default node type
            try:
                _G_adb.default_node_type = DEFAULT_NODE_TYPE
                logger.info(f"Default node type set to '{DEFAULT_NODE_TYPE}'")
            except Exception as e:
                logger.warning(f"Could not set default node type: {str(e)}")
        else:
            logger.error(f"Failed to load graph '{GRAPH_NAME}'")
    else:
        logger.debug("Using cached graph (loaded {:.2f} minutes ago)".format((current_time - _last_loaded) / 60))
    
    return _G_adb

def get_arango_graph():
    """Get or initialize the ArangoGraph object with caching"""
    global _arango_graph, _last_loaded
    
    current_time = time.time()
    
    # Initialize if not exists or if cache has expired
    if _arango_graph is None or _last_loaded is None or (current_time - _last_loaded > _cache_ttl):
        logger.info("Initializing or refreshing ArangoGraph cache...")
        
        db = get_db()
        _arango_graph = ArangoGraph(db)
        
        if not _last_loaded:
            _last_loaded = current_time
            
        logger.info("ArangoGraph initialized successfully")
    else:
        logger.debug("Using cached ArangoGraph")
    
    return _arango_graph

def clear_cache():
    """Clear all cached objects to force reinitialization"""
    global _G_adb, _arango_graph, _last_loaded
    
    _G_adb = None
    _arango_graph = None
    _last_loaded = None
    
    logger.info("Graph cache cleared") 