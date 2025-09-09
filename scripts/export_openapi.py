
import sys, pathlib, json
from fastapi.openapi.utils import get_openapi

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from indexer.app import app

schema = get_openapi(
    title=getattr(app, "title", "AI↔Obsidian Indexer"),
    version=getattr(app, "version", "0.1.0"),
    routes=app.routes,
    description=getattr(app, "description", "Local indexing & RAG API for Obsidian"),
)
out_dir = ROOT / "api"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "openapi.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
print("Wrote", out_dir / "openapi.json")
