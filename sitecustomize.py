"""
Automatically activates the random-tensor monkey-patch.

Python imports this file on start-up *if* it is importable
(see: docs.python.org → “sitecustomize”).
"""

import sys, pathlib

# Ensure the project root is on sys.path so `import src` resolves correctly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importing src.utils triggers the torch.randn patch
try:
    import src.utils        # noqa: F401  (side-effect only)
except Exception as e:
    # Fail softly so we don't prevent the interpreter from starting
    print("[sitecustomize] WARNING: could not import src.utils →", e)