"""HalluGuard — Hugging Face Spaces entry point.

This thin wrapper imports and launches the Gradio app from app/app.py.
HF Spaces expects app.py at the repository root.
"""

import os
import sys

# Add project root to path so src/ imports work
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Load env vars (secrets are injected by HF Spaces automatically)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.app import build_app

demo = build_app()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
