import sys
from pathlib import Path

# Make `import backend...` work regardless of the cwd pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
