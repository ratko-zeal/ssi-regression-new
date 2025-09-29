"""
App configuration.
"""
import os
from datetime import date
from pathlib import Path

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5")

# Features
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"
ENABLE_FILE_SEARCH = os.getenv("ENABLE_FILE_SEARCH", "false").lower() == "true"

# Vector / file search IDs (placeholders)
FILE_SEARCH_INDEX_ID = os.getenv("FILE_SEARCH_INDEX_ID", "vector-index-id")

# Data locations (default to streamlit_app/data)
_base_dir = Path(__file__).resolve().parent.parent
_default_data_dir = _base_dir / "data"
_default_knowledge_dir = _base_dir / "knowledge"

_env_data_dir = os.getenv("DATA_DIR")
if _env_data_dir and _env_data_dir.strip():
    DATA_DIR = _env_data_dir
else:
    DATA_DIR = str(_default_data_dir)

_env_build_dir = os.getenv("BUILD_DIR")
if _env_build_dir and _env_build_dir.strip():
    BUILD_DIR = _env_build_dir
else:
    BUILD_DIR = str(Path(DATA_DIR) / "build")

_env_knowledge_dir = os.getenv("KNOWLEDGE_DIR")
if _env_knowledge_dir and _env_knowledge_dir.strip():
    KNOWLEDGE_DIR = _env_knowledge_dir
else:
    KNOWLEDGE_DIR = str(_default_knowledge_dir)

# Metadata
AS_OF_DATE = os.getenv("AS_OF_DATE", str(date.today()))
VERSION = os.getenv("VERSION", "")  # empty means null in outputs
