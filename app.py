"""Entry point for a Hugging Face *Gradio* Space (free CPU Basic hardware).

Hugging Face runs `python app.py` and expects the app on port 7860. OpsPilot is a
FastAPI app with the Gradio UI mounted at "/", so this file starts it with Uvicorn.
For local development use `uvicorn app.main:app --app-dir backend ...` instead."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

# A Space sits behind Hugging Face's proxy: read the visitor's address from it.
os.environ.setdefault("TRUST_PROXY", "true")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# Hugging Face rejects binary files pushed without Git LFS, so the deploy leaves out
# the sample PDFs. Rebuild them on first start.
SAMPLE_DIR = ROOT / "backend" / "app" / "sample_data"
if not list(SAMPLE_DIR.glob("*.pdf")):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_samples.py")], check=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
