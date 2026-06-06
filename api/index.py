import sys
from pathlib import Path

# Fix python resolution path inside Render container
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import your FastAPI instance
from main import app

# EXPLICITLY expose 'app' to the global scope for Uvicorn
app = app